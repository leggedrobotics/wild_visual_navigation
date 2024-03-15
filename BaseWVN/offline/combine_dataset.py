import os
import torch

from offline_training_lightning import *
from BaseWVN.config.wvn_cfg import ParamCollection,save_to_yaml

param=ParamCollection()
new_env="vowhite_both"
os.makedirs(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,"train",new_env),exist_ok=True)

# # 1. combine nodes files
mode="train"
nodes_1=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_1st","train_nodes_new.pt"))
nodes_2=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_2nd",param.offline.nodes_datafile))
nodes=nodes_1+nodes_2
torch.save(nodes,os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,new_env,param.offline.nodes_datafile))

# 2. combine mask files
# --2.1. combine white board
white_board_1=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_1st","white_masks.pt"))
white_board_2=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_2nd","white_masks.pt"))
white_board=torch.cat([white_board_1,white_board_2],dim=0)  
torch.save(white_board,os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,new_env,"white_masks.pt"))

# --2.2. combine ground
ground_1=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_1st","ground_masks.pt"))
ground_2=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_2nd","ground_masks.pt"))
ground=torch.cat([ground_1,ground_2],dim=0)
torch.save(ground,os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,new_env,"ground_masks.pt"))

# --2.3. combine gt
gt_1=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_1st","gt_masks_SAM.pt"))
gt_2=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_2nd","gt_masks_SAM.pt"))
gt=torch.cat([gt_1,gt_2],dim=0)
torch.save(gt,os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,new_env,"gt_masks_SAM.pt"))

# 3. combine train_data
train_data_1=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_1st","train_data.pt"))
train_data_2=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_2nd",param.offline.train_datafile))
train_data=train_data_1+train_data_2
torch.save(train_data,os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,new_env,param.offline.train_datafile))

# 4. combine img buffer
img_buffer_1=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_1st",param.offline.image_file))
img_buffer_2=torch.load(os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,"vowhite_2nd",param.offline.image_file))
img_buffer={}
img_buffer= {**img_buffer_1, **img_buffer_2}
torch.save(img_buffer,os.path.join(WVN_ROOT_DIR,param.offline.data_folder,mode,new_env,param.offline.image_file))
