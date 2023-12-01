import torch.nn.functional as F
import torch
from typing import Optional,Union
from torch import nn

from .confidence_generator import ConfidenceGenerator
from ..model import VD_dataset

class PhyLoss(nn.Module):
    def __init__(
        self,
        w_pred: float,
        w_reco: float,
        method: str,
        confidence_std_factor: float,
        log_enabled: bool,
        log_folder: str,
        **kwargs,):
        super(PhyLoss,self).__init__()
        self._w_pred=w_pred
        self._w_recon=w_reco
        self._method=method
        
        self._confidence_generator=ConfidenceGenerator(
            method=method,
            std_factor=confidence_std_factor,
            log_enabled=log_enabled,
            log_folder=log_folder,
        )
        self.loss_type=kwargs.get("reco_loss_type",'cosine') # mse or cosine
    def reset(self):
        self._confidence_generator.reset()
        
    def forward(
        self, dataset: Union[VD_dataset, tuple], res: Union[torch.Tensor,tuple], update_generator: bool = True, step: int = 0, log_step: bool = False,batch_idx:int=None
    ):  
        if isinstance(dataset, tuple):
            x_label, y_label = dataset
        elif isinstance(dataset, VD_dataset):
            x_label=dataset.get_x(batch_idx)
            y_label=dataset.get_y(batch_idx)
        else:
            raise ValueError("dataset must be a tuple or a VD_dataset")
        
        if isinstance(res, tuple):
            # res is a tuple of (target,pred) from RndMLP
            x_label=res[0]
            res=res[1]
        
        # Compute reconstruction loss
        nr_channel_reco = x_label.shape[1]
        # loss_reco = F.mse_loss(res[:, :nr_channel_reco], x_label, reduction="none").mean(dim=1)
        if self.loss_type == 'mse':
            # Mean Squared Error Loss
            loss_reco = F.mse_loss(res[:, :nr_channel_reco], x_label, reduction="none").mean(dim=1)
        elif self.loss_type == 'cosine':
            # Cosine Similarity Loss
            cosine_sim = F.cosine_similarity(res[:, :nr_channel_reco], x_label, dim=1)
            loss_reco = 1 - cosine_sim
        else:
            raise ValueError("Invalid loss type specified. Choose 'mse' or 'cosine'")
    
        with torch.no_grad():
            if update_generator:
                # since we only use the foothold as dataset, so x==x_positive
                confidence = self._confidence_generator.update(
                    x=loss_reco, x_positive=loss_reco, step=step, log_step=log_step
                )
            else:
                confidence = self._confidence_generator.inference_without_update(x=loss_reco)
        # need to normalize the last two dim of res seperately since their range is different
        normalized_y=self.normalize_tensor(y_label)
        normalized_res=self.normalize_tensor(res[:, nr_channel_reco:])
        loss_pred_raw=F.mse_loss(normalized_res, normalized_y, reduction="none").mean(dim=1)

        loss_final=self._w_pred*loss_pred_raw.mean()+self._w_recon*loss_reco.mean()
        
        return loss_final, confidence,{"loss_pred":loss_pred_raw.mean(),"loss_reco":loss_reco.mean()}
    
    def compute_confidence_only(self,res:Union[torch.Tensor,tuple],input:torch.Tensor):
        if isinstance(res,tuple):
            input=res[0]
            res=res[1]
        nr_channel_reco =input.shape[1]
        if self.loss_type == 'mse':
            loss_reco_raw = F.mse_loss(res[:, :nr_channel_reco], input, reduction="none")
            loss_reco=loss_reco_raw.mean(dim=1)
        elif self.loss_type == 'cosine':
            # Cosine Similarity Loss
            cosine_sim = F.cosine_similarity(res[:, :nr_channel_reco], input, dim=1)
            loss_reco = 1 - cosine_sim
            loss_reco_raw=None
        confidence=self._confidence_generator.inference_without_update(x=loss_reco)
        return confidence,loss_reco,loss_reco_raw
    
    def normalize_tensor(self,tensor):
        # Assuming tensor shape is [batch_size, 2]
        
        # Extract each dimension
        dim0 = tensor[:, 0]  # First dimension (already 0-1)--friction
        dim1 = tensor[:, 1]  # Second dimension (1-10)--stiffness

        # Normalize the second dimension
        min_val = 1.0  # Minimum value in the second dimension
        max_val = 10.0  # Maximum value in the second dimension
        dim1_normalized = (dim1 - min_val) / (max_val - min_val)

        # Combine the dimensions back into a tensor
        normalized_tensor = torch.stack((dim0, dim1_normalized), dim=1)

        return normalized_tensor
    
    # def update_node_confidence(self, node):
    #     reco_loss = F.mse_loss(node.prediction[:, :-2], node.features, reduction="none").mean(dim=1)
    #     node.confidence = self._confidence_generator.inference_without_update(reco_loss)