
# from BaseWVN.utils import *
# import torch
# width, height = 400, 300
# channel = 0
# img = torch.rand((1,3,height, width))

# # Create a test mask with values in the range specific to the channel
# if channel == 0:  # Friction channel
#     mask = torch.ones((1,height, width))*0.8
# else:  # Stiffness channel
#     mask = np.random.uniform(1, 10, (height, width))

# # Create overlay image
# overlay_img = plot_overlay_image(img, overlay_mask=mask, alpha=1.0, channel=channel)

# # Save image with color bar
# output_path = "test_overlay_with_colorbar.png"
# add_color_bar_and_save(overlay_img, channel, output_path)

import torch
import random
import numpy as np
import os
import rosbag
import cv2
import matplotlib.pyplot as plt
import datetime
from BaseWVN import WVN_ROOT_DIR
from BaseWVN.offline.model_helper import *
from BaseWVN.offline.dataset_cls import *
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

param=ParamCollection()
mode=param.offline.mode
ckpt_parent_folder=os.path.join(WVN_ROOT_DIR,param.offline.ckpt_parent_folder)

# m=get_model(param.model).to(param.run.device)
# model=DecoderLightning(m,param)
# if not param.offline.use_online_ckpt:
#     checkpoint_path = find_latest_checkpoint(ckpt_parent_folder)
# else:
#     checkpoint_path = os.path.join(WVN_ROOT_DIR,param.general.resume_training_path)
# if checkpoint_path:
#     print(f"Latest checkpoint path: {checkpoint_path}")
# else:
#     print("No checkpoint found.")

# checkpoint = torch.load(checkpoint_path)
# model.model.load_state_dict(checkpoint["model_state_dict"])
# model.loss_fn.load_state_dict(checkpoint["phy_loss_state_dict"])
# model.step = checkpoint["step"]
# model.time = checkpoint["time"] if not param.offline.use_online_ckpt else "online"
# model.val_loss = checkpoint["loss"]
# model.model.eval()
feat_extractor=FeatureExtractor(device=param.run.device,
                                    segmentation_type=param.feat.segmentation_type,
                                    input_size=param.feat.input_size,
                                    feature_type=param.feat.feature_type,
                                    interp=param.feat.interp,
                                    center_crop=param.feat.center_crop,)
nodes=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,param.offline.env,param.offline.nodes_datafile))    
node=nodes[0]
img=node.image.to(param.run.device)
img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
reproj_mask=node.supervision_signal_valid[0].unsqueeze(0).unsqueeze(0).to(param.run.device)
B,C,H,W=img.shape
feat_extractor.set_original_size(W,H)
_,_,trans_img,compressed_feats=feat_extractor.extract(img)

from skimage.segmentation import slic as Slic
from skimage.color import label2rgb

# slic = Slic(num_components=100, compactness=10)
# transform image to numpy
feat=next(iter(compressed_feats.values()))
feat=F.interpolate(feat.type(torch.float32),size=(H,W),mode='nearest')
feat=torch.cat((img,feat),dim=1)
feat=feat.permute(0,2,3,1)
feat_np = feat[0].cpu().numpy()
seg = Slic(np.ascontiguousarray(np.uint8(img_np*255.0)),n_segments=20, compactness=20,channel_axis=2)
seg_torch=torch.tensor(seg).unsqueeze(0).unsqueeze(0).to(param.run.device)
seg_color = label2rgb(seg, image=img_np, bg_label=0)
print('ok')
# Create a figure with two subplots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # First subplot for the original image
plt.imshow(img_np)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)  # Second subplot for the segmentation
plt.imshow(seg_color)
plt.title("SLIC Segmentation")
plt.axis('off')
plt.show()