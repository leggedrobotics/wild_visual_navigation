import torch.nn.functional as F
from torch_geometric.data import Data
import torch
from typing import Optional
from wild_visual_navigation.utils import ConfidenceGenerator
from torch import nn
from torchmetrics import ROC, AUROC


class TraversabilityLoss(nn.Module):
    def __init__(
        self,
        w_trav,
        w_reco,
        w_temp,
        anomaly_balanced,
        model,
        method,
        confidence_std_factor,
        trav_cross_entropy=False,
        log_enabled: bool = False,
        log_folder: str = "/tmp",
    ):
        # TODO remove trav_cross_entropy default param when running in online mode
        super(TraversabilityLoss, self).__init__()
        self._w_trav = w_trav

        self._w_reco = w_reco
        self._w_temp = w_temp
        self._model = model
        self._anomaly_balanced = anomaly_balanced
        self._trav_cross_entropy = trav_cross_entropy
        if self._trav_cross_entropy:
            self._trav_loss_func = F.binary_cross_entropy
        else:
            self._trav_loss_func = F.mse_loss

        self._confidence_generator = ConfidenceGenerator(
            std_factor=confidence_std_factor,
            method=method,
            log_enabled=log_enabled,
            log_folder=log_folder,
        )

        self._anomaly_detection_only = self._w_trav == 0
        if self._anomaly_detection_only:
            self._anomaly_balanced = False
            self._anomaly_threshold = None
            # Determine optimal thershould for anomaly detection
            # Store all the reconsturction loss per segments
            # Compute the AUC over the full dataset
            # For each Threshold compute the AUROC and select the one with the highest value
            self._prediction_buffer = []
            self._target_prop_buffer = []
            self._target_gt_buffer = []
            print("Warning: Loss function will override the network traversability prediciton!")

            # self._roc.reset()
            # self._roc = ROC(task="binary")

    def reset(self):
        if self._anomaly_balanced:
            self._confidence_generator.reset()

        if self._anomaly_detection_only:
            self._prediction_buffer = []
            self._target_prop_buffer = []
            self._target_gt_buffer = []

    def compute_anomaly_detection_threshold(self):
        if self._anomaly_detection_only:
            # Normalize prediction to 0-1
            pred = torch.cat(self._prediction_buffer, dim=0)
            max_val = pred.max()
            pred = pred / max_val
            # Low loss = traversable; High loss = not traversable
            pred = 1 - pred

            target_gt = torch.cat(self._target_gt_buffer, dim=0)
            target_prop = torch.cat(self._target_prop_buffer, dim=0)

            roc = ROC(task="binary", thresholds=5000)
            roc.to(pred.device)
            nr = 100
            for k in range(int(pred.shape[0] / nr) + 1):
                ma = (k + 1) * nr
                if ma > pred.shape[0]:
                    ma = pred.shape[0]
                roc.update(pred[k * nr : ma], target_gt[k * nr : ma].type(torch.long))

            fpr, tpr, thresholds = roc.compute()
            aurocs = torch.zeros_like(thresholds)
            for j, t in enumerate(thresholds.tolist()):
                auroc = AUROC(task="binary")
                auroc.to(pred.device)
                auroc.update((pred > t).type(torch.float32), target_gt.type(torch.long))
                aurocs[j] = auroc.compute()

            max_idx = torch.where(aurocs == aurocs.max())[0][0]

            auroc_prop = AUROC(task="binary")
            auroc_prop.to(pred.device)
            auroc_prop.update((pred > thresholds[max_idx]).type(torch.float32), target_prop.type(torch.long))

            self._anomaly_threshold = (1 - thresholds[max_idx]) * max_val
            self.reset()
            return {"auroc_gt": aurocs.max().item(), "auroc_prop": auroc_prop.compute().item()}
        else:
            return {}

    def forward(
        self,
        graph: Data,
        res: torch.Tensor,
        update_generator: bool = True,
        step: int = 0,
        log_step: bool = False,
        update_buffer: bool = False,
    ):
        # Compute reconstruction loss
        nr_channel_reco = graph.x.shape[1]
        loss_reco = F.mse_loss(res[:, -nr_channel_reco:], graph.x, reduction="none").mean(dim=1)

        with torch.no_grad():
            if update_generator:
                confidence = self._confidence_generator.update(
                    x=loss_reco, x_positive=loss_reco[graph.y_valid], step=step, log_step=log_step
                )
            else:
                confidence = self._confidence_generator.inference_without_update(x=loss_reco)

        if self._anomaly_detection_only and update_buffer:
            self._prediction_buffer.append(loss_reco.clone().detach())
            self._target_gt_buffer.append(graph.y_gt.clone().detach())
            self._target_prop_buffer.append(graph.y.clone().detach())

        label = graph.y[:]
        if self._trav_cross_entropy:
            label = label.type(torch.long)
            loss_trav_raw = self._trav_loss_func(
                res[:, :-nr_channel_reco].squeeze()[:, 0], label.type(torch.float32), reduction="none"
            )
        else:
            loss_trav_raw = self._trav_loss_func(res[:, :-nr_channel_reco].squeeze(), label, reduction="none")

        loss_trav_raw_labeled = loss_trav_raw[graph.y_valid]
        loss_trav_raw_not_labeled = loss_trav_raw[~graph.y_valid]

        # Scale the loss
        loss_trav_raw_not_labeled_weighted = loss_trav_raw_not_labeled * (1 - confidence)[~graph.y_valid]

        if self._anomaly_balanced:
            loss_trav_confidence = (loss_trav_raw_not_labeled_weighted.sum() + loss_trav_raw_labeled.sum()) / (
                graph.y.shape[0]
            )
        else:
            loss_trav_confidence = loss_trav_raw.mean()

        # Compute temoporal loss
        if self._w_temp > 0:
            with torch.no_grad():
                # Create graph for previous image
                ptr = [torch.tensor(0, device=graph.x.device)]
                graph.edge_index_previous
                for b in graph.x_previous_count:
                    ptr.append(ptr[-1] + b)
                    graph.edge_index_previous[ptr[-2] : ptr[-1]] += ptr[-2]
                ptr = torch.stack(ptr)
                graph_aux = Data(x=graph.x_previous, edge_index=graph.edge_index_previous, ptr=ptr)

                # Inference the previous image
                aux_res = self._model(graph_aux)

                # This part is tricky:
                # 1. Correspondences across each graph is stored as a list where [:,0] points to the previous graph segment
                #    and [:,1] to the respective current graph segment.
                # 2. We use the graph_ptrs to correctly increment the indexes such that we can do a graph operation to
                #    to compute the MSE.
                current_indexes = []
                previous_indexes = []
                for j, (ptr, aux_ptr) in enumerate(zip(graph.ptr[:-1], graph_aux.ptr[:-1])):
                    current_indexes.append(graph[j].correspondence[:, 1] + ptr)
                    previous_indexes.append(graph[j].correspondence[:, 0] + aux_ptr)
                previous_indexes = torch.cat(previous_indexes)
                current_indexes = torch.cat(current_indexes)
            loss_temp = F.mse_loss(res[current_indexes, 0], aux_res[previous_indexes, 0])
        else:
            loss_temp = torch.zeros_like(loss_trav_confidence)

        loss_reco_mean = loss_reco[graph.y_valid].mean()
        # Compute total loss
        loss = self._w_trav * loss_trav_confidence + self._w_reco * loss_reco_mean + self._w_temp * loss_temp

        res_updated = res.clone().detach()
        if self._anomaly_detection_only:
            if self._anomaly_threshold is not None:
                res_updated[:, 0] = (loss_reco < self._anomaly_threshold).type(torch.float32)

        return (
            loss,
            {
                "loss_reco": loss_reco_mean,
                "loss_trav": loss_trav_raw.mean(),
                "loss_temp": loss_temp.mean(),
                "loss_trav_confidence": loss_trav_confidence,
                "confidence": confidence,
            },
            res_updated,
        )
