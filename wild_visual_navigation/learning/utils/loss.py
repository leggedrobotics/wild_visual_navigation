import torch.nn.functional as F
from torch_geometric.data import Data
import torch
from typing import Optional
from wild_visual_navigation.utils import ConfidenceGenerator
from torch import nn


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
        w_trav_start: Optional[float] = None,
        w_trav_increase: Optional[float] = None,
    ):
        # TODO remove trav_cross_entropy default param when running in online mode
        super(TraversabilityLoss, self).__init__()
        if w_trav_start is None:
            self._w_trav_start = w_trav
            self._w_trav = w_trav
        else:
            assert w_trav_start <= w_trav
            self._w_trav = w_trav_start
            self._w_trav_start = w_trav
        self._w_trav_increase = w_trav_increase

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

    def schedule_w_trav(self):
        if not self._w_trav_increase is None:
            self._w_trav += self._w_trav_increase
            self._w_trav = min(self._w_trav_start, self._w_trav)

    def reset(self):
        pass

    def forward(
        self,
        graph: Data,
        res: torch.Tensor,
        update_generator: bool = True,
        step: int = 0,
        log_step: bool = False,
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
        return loss, {
            "loss_reco": loss_reco_mean,
            "loss_trav": loss_trav_raw.mean(),
            "loss_temp": loss_temp.mean(),
            "loss_trav_confidence": loss_trav_confidence,
            "confidence": confidence,
        }
