import torch.nn.functional as F
from torch_geometric.data import Data
import torch
from typing import Optional


def compute_loss(
    batch: Data, res: torch.Tensor, loss_cfg: any, model: torch.nn.Module, batch_aux: Optional[Data] = None
):
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
