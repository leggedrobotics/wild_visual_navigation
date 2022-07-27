import torch.nn.functional as F
import torch


def get_confidence(pred: torch.tensor, x: torch.Tensor, scaled: bool = True):
    res = F.mse_loss(pred, x, reduction="none").mean(1)
    res -= res.min()
    res /= res.max()
    return 1 - res
