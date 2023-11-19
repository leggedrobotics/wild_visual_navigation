import torch
import os
import PIL.Image
import cv2
from BaseWVN import WVN_ROOT_DIR
from BaseWVN.utils import PhyLoss,FeatureExtractor,concat_feat_dict,plot_overlay_image
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
        loss_params=self.params.loss
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
        self.loss_fn=PhyLoss(w_pred=loss_params.w_pred,
                               w_reco=loss_params.w_reco,
                               method=loss_params.method,
                               confidence_std_factor=loss_params.confidence_std_factor,
                               log_enabled=loss_params.log_enabled,
                               log_folder=loss_params.log_folder)
    
        
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
        
        self.log('train_loss', loss,on_step=True,prog_bar=True)
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
        # TODO: to plot overlay, upsample the mask.Maybe wrap up the following into function
        if batch_idx==0:
            output_dir=os.path.join(WVN_ROOT_DIR,"results","overlay")
            if not os.path.exists(output_dir):
                # Create the directory
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            else:
                print(f"Directory already exists: {output_dir}")
            features, seg,trans_img,compressed_feats=self.feat_extractor.extract(self.test_img)
            feat_input,H,W=concat_feat_dict(compressed_feats)
            feat_input=feat_input.squeeze(0)
            output=self.model(feat_input)
            confidence=self.loss_fn.compute_confidence_only(output,feat_input)
            confidence=confidence.reshape(H,W)
            output_phy=output[:,-2:].reshape(H,W,2).permute(2,0,1)
            mask=confidence<self.params.loss.confidence_threshold
            mask = mask.unsqueeze(0).repeat(output_phy.shape[0], 1, 1)
            output_phy[mask] = torch.nan
            channel_num=output_phy.shape[0]
            for i in range(channel_num):
                overlay_img=plot_overlay_image(trans_img, overlay_mask=output_phy, channel=i,alpha=0.7)
                # Convert the numpy array to an image
                out_image = PIL.Image.fromarray(overlay_img)
                # Construct a filename
                filename = f"dense_prediction_channel_{i}.jpg"
                file_path = os.path.join(output_dir, filename)

                # Save the image
                out_image.save(file_path)
            pass
        self.log('val_loss', loss,on_step=True)
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
    # Initialize the Neptune logger
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MDVkNmYxYi1kZjZjLTRmNmEtOGQ5My0xZmE2YTc0OGVmN2YifQ==",
        project="swsychen/Decoder-MLP",
    )
    
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
    m=get_model(param.model).to(device)
    model=DecoderLightning(m,param)
    trainer = Trainer(accelerator="gpu", devices=[0], logger=neptune_logger, max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)
    pass

if __name__ == "__main__":
    train_and_evaluate()
    