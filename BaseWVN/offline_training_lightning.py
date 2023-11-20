import torch
import os
import cv2
import datetime
from BaseWVN import WVN_ROOT_DIR
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

        self.test_img=load_one_test_image("results/manager","image_buffer.pt")
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


def load_data(folder, file):
    """Load data from the data folder."""
    path=os.path.join(WVN_ROOT_DIR, folder, file)
    data=torch.load(path)
    return data

def load_one_test_image(folder, file):
    """ return img in shape (B,C,H,W) """
    image_path = os.path.join(WVN_ROOT_DIR, folder,file)
    if file.lower().endswith('.pt'):
        is_pt_file=True
    else:
        is_pt_file=False
    if not is_pt_file:
        np_img = cv2.imread(image_path)
        img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        img = img.permute(2, 0, 1)
        img = (img.type(torch.float32) / 255)[None]
    else:
        imgs=torch.load(image_path)
        time,img=next(iter(imgs.items()))
    return img

def load_all_test_images(folder):
    """ Load all images from a folder and return them  """
    if "manager" in folder:
        is_pt_file=True
    else:
        is_pt_file=False
    if not is_pt_file:
        images = {}

        for file in os.listdir(os.path.join(WVN_ROOT_DIR, folder)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                image_path = os.path.join(WVN_ROOT_DIR, folder, file)
                np_img = cv2.imread(image_path)
                img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                img = img.permute(2, 0, 1)
                img = (img.type(torch.float32) / 255)[None]
                images[file] = img
    else:
        images={}
        for file in os.listdir(os.path.join(WVN_ROOT_DIR, folder)):
            if file.lower().endswith('.pt') and file.lower().startswith('image'):
                image_path = os.path.join(WVN_ROOT_DIR, folder, file)
                imgs=torch.load(image_path)
                for time,img in imgs.items():    
                    images[time]=img
                    
                break
    return images

def find_latest_checkpoint(parent_dir):
    # List all folders in the parent directory
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]

    # Sort these folders based on datetime in their names
    try:
        sorted_folders = sorted(folders, key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"), reverse=True)
    except ValueError:
        # Handle folders that don't follow the datetime naming convention
        print("Error: Some folders do not follow the expected datetime naming convention.")
        return None

    # Select the latest folder
    latest_folder = sorted_folders[0] if sorted_folders else None

    if latest_folder:
        latest_folder_path = os.path.join(parent_dir, latest_folder)
        
        # Search for the 'last_checkpoint.pt' file in this folder
        last_checkpoint_path = os.path.join(latest_folder_path, 'last_checkpoint.pt')
        
        if os.path.exists(last_checkpoint_path):
            return last_checkpoint_path
        else:
            print("Last checkpoint not found in the latest folder.")
            return None
    else:
        print("No folders found in the parent directory.")
        return None


def train_and_evaluate():
    """Train and evaluate the model."""
    mode="test"
    parent_folder=os.path.join(WVN_ROOT_DIR,"results/overlay")
    param=ParamCollection()
    m=get_model(param.model).to(param.run.device)
    model=DecoderLightning(m,param)
    if mode=="train":
        # Initialize the Neptune logger
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MDVkNmYxYi1kZjZjLTRmNmEtOGQ5My0xZmE2YTc0OGVmN2YifQ==",
            project="swsychen/Decoder-MLP",
        )
        max_epochs=42
        folder='results/manager'
        file='train_data.pt'
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
        
        trainer = Trainer(accelerator="gpu", devices=[0], logger=neptune_logger, max_epochs=max_epochs)
        trainer.fit(model, train_loader, val_loader)
        torch.save({
                    "time": model.time,
                    "step" : model.step,
                    "model_state_dict": model.model.state_dict(),
                    "phy_loss_state_dict": model.loss_fn.state_dict(),
                    "loss": model.val_loss.item(),
                },
                os.path.join(parent_folder,model.time,"last_checkpoint.pt"))
    else:
        checkpoint_path = find_latest_checkpoint(parent_folder)
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
        test_imgs=load_all_test_images("results/manager")
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

        pass

if __name__ == "__main__":
    train_and_evaluate()
    