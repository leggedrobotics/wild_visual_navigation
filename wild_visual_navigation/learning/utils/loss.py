import torch.nn.functional as F
from torch_geometric.data import Data
import torch
from typing import Optional
from wild_visual_navigation.utils import ConfidenceGenerator
from torch import nn


class TraversabilityLoss(nn.Module):
    def __init__(self, w_trav, w_reco, w_temp, anomaly_blanced, model, false_negative_weight=1.0):
        super(TraversabilityLoss, self).__init__()
        self._w_trav = w_trav
        self._w_reco = w_reco
        self._w_temp = w_temp
        self._model = model
        self._anomaly_balanced = anomaly_blanced
        self._false_negative_weight = false_negative_weight

        if self._anomaly_balanced:
            self._confidence_generator = ConfidenceGenerator()

    def reset(self):
        if self._anomaly_balanced:
            self._confidence_generator.reset()

    def forward(self, batch: Data, res: torch.Tensor, batch_aux: Optional[Data] = None):
        # Compute reconstruction loss
        loss_reco = F.mse_loss(res[batch.y_valid][:, 1:], batch.x[batch.y_valid])

        # Compute traversability loss
        if self._anomaly_balanced:
            loss_reco_raw = F.mse_loss(res[:, 1:], batch.x, reduction="none").mean(dim=1)
            with torch.no_grad():
                confidence = self._confidence_generator.update(loss_reco_raw)

            m = batch.y == 0

            loss_trav_raw_labled = F.mse_loss(res[batch.y_valid, 0], batch.y[batch.y_valid], reduction="none")
            loss_trav_raw_not_labled = F.mse_loss(res[~batch.y_valid, 0], batch.y[~batch.y_valid], reduction="none")

            # Scale the loss
            loss_trav_raw_not_labled = loss_trav_raw_not_labled * (1 - confidence)[~batch.y_valid]
            loss_trav_raw_labled = loss_trav_raw_labled * self._false_negative_weight

            loss_trav_out = (loss_trav_raw_not_labled.sum() + loss_trav_raw_labled.sum()) / (m.numel())
            loss_trav_raw = F.mse_loss(res[:, 0], batch.y[:])
        else:
            loss_trav_raw = F.mse_loss(res[:, 0], batch.y[:])
            loss_trav_out = loss_trav_raw

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
            loss_temp = torch.zeros_like(loss_trav_out)

        # Compute total loss
        loss = self._w_trav * loss_trav_out + self._w_reco * loss_reco + self._w_temp * loss_temp
        return loss, {
            "loss_reco": loss_reco,
            "loss_trav": loss_trav_raw,
            "loss_temp": loss_temp,
            "loss_trav_confidence": loss_trav_out,
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
