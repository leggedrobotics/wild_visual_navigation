import torch.nn.functional as F
from torch_geometric.data import Data
import torch
from typing import Optional
from torch import nn

from BaseWVN.utils import ConfidenceGenerator

class PhyLoss(nn.Module):
    def __init__(
        self,
        method: str,
        confidence_std_factor: float,
        log_enabled: bool,
        log_folder: str,):
        super(PhyLoss,self).__init__()
        
        self._method=method
        
        self._confidence_generator=ConfidenceGenerator(
            method=method,
            std_factor=confidence_std_factor,
            log_enabled=log_enabled,
            log_folder=log_folder,
        )
    
    def reset(self):
        self._confidence_generator.reset()
        