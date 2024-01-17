import torch
import os
import cv2
from BaseWVN import WVN_ROOT_DIR
from BaseWVN.utils import PhyLoss,FeatureExtractor
from BaseWVN.model import VD_dataset,get_model
from BaseWVN.config.wvn_cfg import ParamCollection
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer

class DecoderLightning(pl.LightningModule):
    def __init__(self,model,params:ParamCollection):
        super().__init__()
        self.model=model
        self.params=params
        param=self.params.loss
        self.step=0

        self.test_img=load_test_image("image","hiking.png")
        B,C,H,W=self.test_img.shape
        self.feat_extractor=FeatureExtractor(device=self.params.run.device,
                                             segmentation_type=self.params.feat.segmentation_type,
                                             input_size=self.params.feat.input_size,
                                             feature_type=self.params.feat.feature_type,
                                             interp=self.params.feat.interp,
                                             center_crop=self.params.feat.center_crop,
                                             original_width=W,
                                             original_height=H,)
        self.loss_fn=PhyLoss(w_pred=param.w_pred,
                               w_reco=param.w_reco,
                               method=param.method,
                               confidence_std_factor=param.confidence_std_factor,
                               log_enabled=param.log_enabled,
                               log_folder=param.log_folder)
    
        
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
        # TODO: IMAGE loading for inference and upload to neptune
        self.log('val_loss', loss)
        self.step+=1
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


def load_data(folder, file):
    """Load data from the data folder."""
    path=os.path.join(WVN_ROOT_DIR, folder, file)
    data=torch.load(path)
    return data

def load_test_image(folder, file):
    """ return img in shape (B,C,H,W) """
    image_path = os.path.join(WVN_ROOT_DIR, folder,file)
    np_img = cv2.imread(image_path)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    return img

def train_and_evaluate():
    """Train and evaluate the model."""
    # dataset loading
    param=ParamCollection()
    max_epochs=5
    folder='results/manager'
    file='graph_data.pt'
    data=load_data(folder, file)
    
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
    
    # model loading
    step=0
    total_loss=0
    model=get_model(param.model).to(device)
    
    test_img=load_test_image("image","hiking.png")
    B,C,H,W=test_img.shape
    feat_extractor=FeatureExtractor(device=param.run.device,
                                            segmentation_type=param.feat.segmentation_type,
                                            input_size=param.feat.input_size,
                                            feature_type=param.feat.feature_type,
                                            interp=param.feat.interp,
                                            center_crop=param.feat.center_crop,
                                            original_width=W,
                                            original_height=H,)
    loss_fn=PhyLoss(w_pred=param.loss.w_pred,
                            w_reco=param.loss.w_reco,
                            method=param.loss.method,
                            confidence_std_factor=param.loss.confidence_std_factor,
                            log_enabled=param.loss.log_enabled,
                            log_folder=param.loss.log_folder)
    optimizer=torch.optim.Adam(model.parameters(), lr=param.optimizer.lr)
    
    

    pass

if __name__ == "__main__":
    train_and_evaluate()
    