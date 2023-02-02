import torch.nn.functional as F
from torch_geometric.data import Data
import torch
from typing import Optional
from wild_visual_navigation.utils import ConfidenceGenerator
from torch import nn
from torchmetrics import ROC, AUROC, Accuracy


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

    def reset(self):
        if self._anomaly_balanced:
            self._confidence_generator.reset()

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

        loss_temp = torch.zeros_like(loss_trav_confidence)

        loss_reco_mean = loss_reco[graph.y_valid].mean()
        # Compute total loss
        loss = self._w_trav * loss_trav_confidence + self._w_reco * loss_reco_mean + self._w_temp * loss_temp

        res_updated = res
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
