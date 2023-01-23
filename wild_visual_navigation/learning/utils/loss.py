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
        method="running_mean",
        false_negative_weight=1.0,
        confidence_std_factor=1.0,
        log_enabled: bool = False,
        log_folder: str = "/tmp",
        w_trav_start: Optional[float] = None,
        w_trav_increase: Optional[float] = None,
    ):
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
        self._false_negative_weight = false_negative_weight

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
        if self._anomaly_balanced:
            self._confidence_generator.reset()

    def forward(
        self,
        batch: Data,
        res: torch.Tensor,
        batch_aux: Optional[Data] = None,
        update_generator: bool = True,
        step: int = 0,
        log_step: bool = False,
    ):
        # Compute reconstruction loss
        loss_reco = F.mse_loss(res[:, 1:], batch.x, reduction="none").mean(dim=1)

        with torch.no_grad():
            if update_generator:
                confidence = self._confidence_generator.update(
                    x=loss_reco, x_positive=loss_reco[batch.y_valid], step=step, log_step=log_step
                )
            else:
                confidence = self._confidence_generator.inference_without_update(x=loss_reco)

        loss_trav_raw = F.mse_loss(res[:, 0], batch.y[:], reduction="none")
        loss_trav_raw_labeled = loss_trav_raw[batch.y_valid]
        loss_trav_raw_not_labeled = loss_trav_raw[~batch.y_valid]

        # Scale the loss
        loss_trav_raw_not_labeled_weighted = loss_trav_raw_not_labeled * (1 - confidence)[~batch.y_valid]

        if self._anomaly_balanced:
            loss_trav_confidence = (loss_trav_raw_not_labeled_weighted.sum() + loss_trav_raw_labeled.sum()) / (
                batch.y.shape[0]
            )
        else:
            loss_trav_confidence = loss_trav_raw.mean()

        # Compute temoporal loss
        if self._w_temp > 0 and batch_aux is not None:
            with torch.no_grad():
                aux_res = self._model(batch_aux)
                # This part is tricky:
                # 1. Correspondences across each graph is stored as a list where [:,0] points to the previous graph segment
                #    and [:,1] to the respective current graph segment.
                # 2. We use the batch_ptrs to correctly increment the indexes such that we can do a batch operation to
                #    to compute the MSE.
                current_indexes = []
                previous_indexes = []
                for j, (ptr, aux_ptr) in enumerate(zip(batch.ptr[:-1], batch_aux.ptr[:-1])):
                    current_indexes.append(batch[j].correspondence[:, 1] + ptr)
                    previous_indexes.append(batch[j].correspondence[:, 0] + aux_ptr)
                previous_indexes = torch.cat(previous_indexes)
                current_indexes = torch.cat(current_indexes)

            loss_temp = F.mse_loss(res[current_indexes, 0], aux_res[previous_indexes, 0])
        else:
            loss_temp = torch.zeros_like(loss_trav_confidence)

        # Compute total loss
        loss = (
            self._w_trav * loss_trav_confidence
            + self._w_reco * loss_reco[batch.y_valid].mean()
            + self._w_temp * loss_temp
        )
        return loss, {
            "loss_reco": loss_reco[batch.y_valid].mean(),
            "loss_trav": loss_trav_raw.mean(),
            "loss_temp": loss_temp.mean(),
            "loss_trav_confidence": loss_trav_confidence,
        }


def compute_loss(batch: Data, res: torch.Tensor, model: torch.nn.Module, batch_aux: Optional[Data] = None):
    # batch.y_valid
    loss_trav = F.mse_loss(res[:, 0], batch.y[:])
    nc = 1
    loss_reco = F.mse_loss(res[batch.y_valid][:, nc:], batch.x[batch.y_valid])

    if loss_cfg["temp"] > 0 and batch_aux is not None:
        with torch.no_grad():
            aux_res = model(batch_aux)
            # This part is tricky:
            # 1. Correspondences across each graph is stored as a list where [:,0] points to the previous graph segment
            #    and [:,1] to the respective current graph segment.
            # 2. We use the batch_ptrs to correctly increment the indexes such that we can do a batch operation to
            #    to compute the MSE.

            current_indexes = []
            previous_indexes = []
            for j, (ptr, aux_ptr) in enumerate(zip(batch.ptr[:-1], batch_aux.ptr[:-1])):
                current_indexes.append(batch[j].correspondence[:, 1] + ptr)
                previous_indexes.append(batch[j].correspondence[:, 0] + aux_ptr)
            previous_indexes = torch.cat(previous_indexes)
            current_indexes = torch.cat(current_indexes)

        loss_temp = F.mse_loss(res[current_indexes, 0], aux_res[previous_indexes, 0])
    else:
        loss_temp = torch.zeros_like(loss_trav)

    loss = loss_cfg["trav"] * loss_trav + loss_cfg["reco"] * loss_reco + loss_cfg["temp"] * loss_temp
    return loss, {"loss_reco": loss_reco, "loss_trav": loss_trav, "loss_temp": loss_temp}
