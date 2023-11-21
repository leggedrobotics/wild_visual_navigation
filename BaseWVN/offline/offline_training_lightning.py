import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime
from BaseWVN import WVN_ROOT_DIR
from BaseWVN.offline.model_helper import *
from BaseWVN.GraphManager import MainNode
from BaseWVN.utils import PhyLoss,FeatureExtractor,concat_feat_dict,plot_overlay_image,compute_phy_mask
from BaseWVN.model import VD_dataset,get_model
from BaseWVN.config.wvn_cfg import ParamCollection,save_to_yaml
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

class DecoderLightning(pl.LightningModule):
    def __init__(self,model,params:ParamCollection):
        super().__init__()
        self.model=model
        self.params=params
        loss_params=self.params.loss
        self.step=0

        self.test_img=load_one_test_image(params.offline.data_folder,params.offline.image_file)
        B,C,H,W=self.test_img.shape
        self.feat_extractor=FeatureExtractor(device=self.params.run.device,
                                             segmentation_type=self.params.feat.segmentation_type,
                                             input_size=self.params.feat.input_size,
                                             feature_type=self.params.feat.feature_type,
                                             interp=self.params.feat.interp,
                                             center_crop=self.params.feat.center_crop,
                                             original_width=W,
                                             original_height=H,)
        self.loss_fn=PhyLoss(w_pred=loss_params.w_pred,
                               w_reco=loss_params.w_reco,
                               method=loss_params.method,
                               confidence_std_factor=loss_params.confidence_std_factor,
                               log_enabled=loss_params.log_enabled,
                               log_folder=loss_params.log_folder)
        self.val_loss=0.0
        self.time=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        xs=xs.squeeze(0)
        ys=ys.squeeze(0)
        # chech if xs and ys both have shape of 2
        if len(xs.shape)!=2 or len(ys.shape)!=2:
            raise ValueError("xs and ys must have shape of 2")
        res=self.model(xs)
        loss,confidence,loss_dict=self.loss_fn((xs,ys),res,step=self.step)
        
        self.log('train_loss', loss)
        if batch_idx==0:
            self.step+=1
        return loss
    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        xs=xs.squeeze(0)
        ys=ys.squeeze(0)
        # chech if xs and ys both have shape of 2
        if len(xs.shape)!=2 or len(ys.shape)!=2:
            raise ValueError("xs and ys must have shape of 2")
        res=self.model(xs)
        loss,confidence,loss_dict=self.loss_fn((xs,ys),res,step=self.step,update_generator=False)
        if batch_idx==0 and self.step%20==0:
            output_phy_resized,trans_img,confidence=compute_phy_mask(self.test_img,self.feat_extractor,self.model,self.loss_fn,self.params.loss.confidence_threshold,True,self.step,time=self.time,param=self.params)
            pass
        self.log('val_loss', loss)
        self.val_loss=loss

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.optimizer.lr)
        return optimizer

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, data:List[VD_dataset]):
        self.data=[]
        for d in data:
            self.data = self.data+d.batches

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


def train_and_evaluate():
    """Train and evaluate the model."""
    param=ParamCollection()
    mode=param.offline.mode
    ckpt_parent_folder=os.path.join(WVN_ROOT_DIR,param.offline.ckpt_parent_folder)
    
    m=get_model(param.model).to(param.run.device)
    model=DecoderLightning(m,param)
    if mode=="train":
        # Initialize the Neptune logger
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MDVkNmYxYi1kZjZjLTRmNmEtOGQ5My0xZmE2YTc0OGVmN2YifQ==",
            project="swsychen/Decoder-MLP",
        )
        max_epochs=42
        data=load_data(param.offline.train_data)
        combined_dataset = BigDataset(data)
        train_size = int(0.8 * len(combined_dataset))
        test_size = len(combined_dataset) - train_size

        train_dataset = Subset(combined_dataset, range(0, train_size))
        val_dataset = Subset(combined_dataset, range(train_size, len(combined_dataset)))
        
        batch_size = 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        # batch in loader is a tuple of (xs, ys)
        # xs:(1, 100, feat_dim), ys:(1, 100, 2)
        sample_input, sample_output = next(iter(train_loader))
        device=sample_input.device
        feat_dim=sample_input.shape[-1]
        label_dim=sample_output.shape[-1]
        
        trainer = Trainer(accelerator="gpu", devices=[0], logger=neptune_logger, max_epochs=max_epochs)
        trainer.fit(model, train_loader, val_loader)
        torch.save({
                    "time": model.time,
                    "step" : model.step,
                    "model_state_dict": model.model.state_dict(),
                    "phy_loss_state_dict": model.loss_fn.state_dict(),
                    "loss": model.val_loss.item(),
                },
                os.path.join(ckpt_parent_folder,model.time,"last_checkpoint.pt"))
    else:
        checkpoint_path = find_latest_checkpoint(ckpt_parent_folder)
        if checkpoint_path:
            print(f"Latest checkpoint path: {checkpoint_path}")
        else:
            print("No checkpoint found.")
            return
        checkpoint = torch.load(checkpoint_path)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.loss_fn.load_state_dict(checkpoint["phy_loss_state_dict"])
        model.step = checkpoint["step"]
        model.time = checkpoint["time"]
        model.val_loss = checkpoint["loss"]
        model.model.eval()
        feat_extractor=FeatureExtractor(device=param.run.device,
                                            segmentation_type=param.feat.segmentation_type,
                                            input_size=param.feat.input_size,
                                            feature_type=param.feat.feature_type,
                                            interp=param.feat.interp,
                                            center_crop=param.feat.center_crop,)
        if param.offline.test_images:
            test_imgs=load_all_test_images(param.offline.data_folder)
            for name,img in test_imgs.items():
                B,C,H,W=img.shape
                feat_extractor.set_original_size(W,H)
                compute_phy_mask(img,feat_extractor,
                                model.model,
                                model.loss_fn,
                                param.loss.confidence_threshold,
                                True,
                                -1,
                                time=model.time,
                                image_name=name,)
        if param.offline.test_nodes:
            nodes=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.nodes_data))
            SAM_label_mask_generate(param,nodes)
        
        pass

def show_mask(mask, ax, random_color=False):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

if __name__ == "__main__":
    train_and_evaluate()
    