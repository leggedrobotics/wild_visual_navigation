import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime
from .. import WVN_ROOT_DIR
from ..GraphManager import MainNode
from ..utils import PhyLoss,FeatureExtractor,concat_feat_dict,plot_overlay_image,compute_phy_mask
from ..model import VD_dataset,get_model
from ..config.wvn_cfg import ParamCollection,save_to_yaml
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from segment_anything import SamPredictor, sam_model_registry
from seem_base import inference,init_model
def load_data(file):
    """Load data from the data folder."""
    path=os.path.join(WVN_ROOT_DIR, file)
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
    
def sample_furthest_points(true_coords, num_points_to_sample):
    """ 
    only support B=1 operation
    """
    B, num, _ = true_coords.shape
    copy=true_coords.clone().type(torch.float32)
    # Calculate all pairwise distances
    pairwise_distances = torch.cdist(copy[0], copy[0])

    if num_points_to_sample == 2:
        # For two points, simply find the pair with the maximum distance
        max_dist, indices = torch.max(pairwise_distances, dim=1)
        furthest_pair = indices[max_dist.argmax()], max_dist.argmax()
        return true_coords[0][list(furthest_pair),:].unsqueeze(0)

def SAM_label_mask_generate(param:ParamCollection,nodes:List[MainNode]):
    """ 
    Using segment anything model to generate gt label mask
    Return: gt_masks in shape (B=node_num,1,H,W)
    
    """
    gt_masks=[]
    sam = sam_model_registry[param.offline.SAM_type](checkpoint=param.offline.SAM_ckpt)
    sam.to(param.run.device)
    predictor = SamPredictor(sam)
    for node in nodes:
        img=node.image.to(param.run.device)
        reproj_mask=node.supervision_signal_valid[0]
        # Find the indices where reproj_mask is True
        true_indices = torch.where(reproj_mask)
        true_coords = torch.stack((true_indices[1], true_indices[0]),dim=1).unsqueeze(0) # (x, y) format
        B, num, _ = true_coords.shape
 
        num_points_to_sample = min(2, num)
        
        sampled_true_coords=sample_furthest_points(true_coords, num_points_to_sample)
        
        # sampled_true_coords = torch.zeros(B, num_points_to_sample, 2)
        # rand_indices = torch.randperm(num)[:num_points_to_sample]
        # sampled_true_coords=true_coords[:,rand_indices,:]
        true_coords=sampled_true_coords.to(param.run.device)
        true_coords_resized=predictor.transform.apply_coords_torch(true_coords,img.shape[-2:])
        points_labels=torch.ones((true_coords_resized.shape[0],true_coords_resized.shape[1]),dtype=torch.int64).to(param.run.device)
        
        # need to image--> H,W,C uint8 format
        input_img=(img.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)
        H,W,C=input_img.shape
        predictor.set_image(input_img)

        # resized_img=predictor.transform.apply_image_torch(img)
        # predictor.set_torch_image(resized_img,img.shape[-2:])
        masks, scores, _ = predictor.predict_torch(point_coords=true_coords_resized,point_labels=points_labels,multimask_output=True)
        _, max_score_indices = torch.max(scores, dim=1)
        gt_mask=masks[:,max_score_indices,:,:]
        gt_masks.append(gt_mask)
        torch.cuda.empty_cache()
        # plt.figure(figsize=(10,10))
        # plt.imshow(input_img)
        # show_mask(gt_mask.squeeze(0), plt.gca())
        # show_points(true_coords.squeeze(0), points_labels.squeeze(0), plt.gca())
        # plt.axis('off')
        # plt.show() 
    return torch.cat(gt_masks,dim=0)

def SEEM_label_mask_generate(param:ParamCollection,nodes:List[MainNode]):
    model=init_model().to(param.run.device)

    gt_masks=[]
    for node in nodes:
        img=node.image.to(param.run.device)
        img=(img*255.0).type(torch.uint8)
        reproj_mask=node.supervision_signal_valid[0].unsqueeze(0).unsqueeze(0)
        
        masks,texts=inference(model,img,reproj_mask)
        # if has multiple masks, add them into one
        mask=torch.sum(masks,dim=0)
        gt_masks.append(mask.unsqueeze(0).unsqueeze(0))
        
        plt.figure(figsize=(10,10))
        input_img=(img.squeeze(0).permute(1,2,0).cpu().numpy())
        plt.imshow(input_img)
        show_mask(mask.squeeze(0), plt.gca())
        show_mask(reproj_mask.squeeze(0), plt.gca(), random_color=True)
        plt.axis('off')
        plt.show()
        
        pass
    return torch.cat(gt_masks,dim=0)

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
