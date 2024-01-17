import torch
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime
from BaseWVN import WVN_ROOT_DIR
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
val_nodes=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,'val',param.offline.env,param.offline.nodes_datafile))
train_nodes=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,'train',param.offline.env,param.offline.nodes_datafile))

num_train_nodes=len(train_nodes)
num_val_nodes=len(val_nodes)

train_dist=0
for i,t_node in enumerate(train_nodes):
    if i==0:
        continue
    else:
        train_dist+=t_node.distance_to(train_nodes[i-1])
val_dist=0
for i,v_node in enumerate(val_nodes):
    if i==0:
        continue
    else:
        val_dist+=v_node.distance_to(val_nodes[i-1])

info_filename=os.path.join(WVN_ROOT_DIR,param.offline.data_folder,'train',param.offline.env,"dataset_info.txt")

with open(info_filename,'w') as f:
    f.write(f"num_train_nodes: {num_train_nodes}\n")
    f.write(f"num_val_nodes: {num_val_nodes}\n")
    f.write(f"train_dist: {train_dist}\n")
    f.write(f"val_dist: {val_dist}\n")
print(f'Information saved to {info_filename}')