import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime
from .. import WVN_ROOT_DIR
from ..GraphManager import MainNode
from ..utils import PhyLoss,FeatureExtractor,concat_feat_dict,plot_overlay_image,compute_phy_mask,plot_image,plot_images_side_by_side,plot_images_in_grid,plot_tsne,compute_pred_phy_loss
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
    valid_folders = []

    # Iterate over each folder in the parent directory
    for folder in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, folder)):
            try:
                # Try parsing the folder name as a datetime object
                datetime.datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S")
                # If successful, add it to the valid_folders list
                valid_folders.append(folder)
            except ValueError:
                # Skip folders that don't follow the datetime naming convention
                continue

    # Sort the valid folders based on datetime in their names
    sorted_folders = sorted(valid_folders, key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"), reverse=True)


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
    
def sample_furthest_points(true_coords, num_points_to_sample,given_point=None):
    """ 
    only support B=1 operation
    """
    
    B, num, _ = true_coords.shape
    if B != 1:
        raise ValueError("Only B=1 operation is supported.")
    
    if given_point is None:
        copy=true_coords.clone().type(torch.float32)
        # Calculate all pairwise distances
        pairwise_distances = torch.cdist(copy[0], copy[0])

        if num_points_to_sample == 2:
            # For two points, simply find the pair with the maximum distance
            max_dist, indices = torch.max(pairwise_distances, dim=1)
            furthest_pair = indices[max_dist.argmax()], max_dist.argmax()
            return true_coords[0][list(furthest_pair),:].unsqueeze(0)
    else:
        if len(given_point.shape)!=2:
            raise ValueError("given_point should be in shape (1,2)")
        # Calculate distances between the given point and all points in true_coords
        distances = torch.cdist(given_point.type(torch.float32), true_coords[0].type(torch.float32))

        # Find the index of the furthest point
        max_dist_index = torch.argmax(distances)

        # Return the furthest point
        return true_coords[0][max_dist_index].unsqueeze(0).unsqueeze(0)



def SAM_label_mask_generate(param:ParamCollection,nodes:List[MainNode]):
    """ 
    Using segment anything model to generate gt label mask
    Return: gt_masks in shape (B=node_num,1,H,W)
    
    """
    gt_masks=[]
    cor_images=[]
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
 
        # combine the furtherst points with the randomly sampled points
        far_coords=sample_furthest_points(true_coords, 2).to(param.run.device)
        num_points_to_sample = min(10, num)
        # sampled_true_coords = torch.zeros(B, num_points_to_sample, 2)
        rand_indices = torch.randperm(num)[:num_points_to_sample]
        sampled_true_coords=true_coords[:,rand_indices,:].to(param.run.device)
        pairs=[]
        for i in range(num_points_to_sample):
            pair=sample_furthest_points(true_coords, 2, given_point=sampled_true_coords[:,i,:]).to(param.run.device)
            pairs.append(pair)
        pairs=torch.cat(pairs,dim=1)
        true_coords=sampled_true_coords.to(param.run.device)
        true_coords=torch.cat((true_coords,far_coords,pairs),dim=1)
        
        true_coords_resized=predictor.transform.apply_coords_torch(true_coords,img.shape[-2:])
        points_labels=torch.ones((true_coords_resized.shape[0],true_coords_resized.shape[1]),dtype=torch.int64).to(param.run.device)
        
        # need to image--> H,W,C uint8 format
        input_img=(img.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)
        H,W,C=input_img.shape
        predictor.set_image(input_img)
        cor_images.append(img)
        # resized_img=predictor.transform.apply_image_torch(img)
        # predictor.set_torch_image(resized_img,img.shape[-2:])
        gt_mask_pts=torch.zeros_like(reproj_mask.unsqueeze(0).unsqueeze(0)).type(torch.int)
        for i in range(true_coords_resized.shape[1]):
            current_point_coords = true_coords_resized[:, i, :].unsqueeze(1)
            current_label = points_labels[:, i].unsqueeze(0)
            masks, scores, _ = predictor.predict_torch(point_coords=current_point_coords,point_labels=current_label,multimask_output=True)
            
            for j in range(masks.shape[1]):
                _, max_score_indices = torch.max(scores, dim=1)
                gt_mask=masks[:,max_score_indices,:,:]
                mask_ratio_s=gt_mask.sum()/(H*W)
                if mask_ratio_s>0.70:
                    scores[:,max_score_indices]=0
                else:
                    break              
            gt_mask_pts+=gt_mask.type(torch.int)
            gt_mask_pts[gt_mask_pts>0]=1
        gt_mask=gt_mask_pts.type(torch.bool)
        # masks, scores, _ = predictor.predict_torch(point_coords=true_coords_resized,point_labels=points_labels,multimask_output=True)
        # _, max_score_indices = torch.max(scores, dim=1)
        # gt_mask=masks[:,max_score_indices,:,:]
        
        
        gt_masks.append(gt_mask)
        torch.cuda.empty_cache()
        
        # mask_ratio=gt_mask.sum()/(H*W)
        # plt.figure(figsize=(10,10))
        # plt.imshow(input_img)
        # print("Mask region ratio: ",mask_ratio.item())
        # show_mask(gt_mask.squeeze(0), plt.gca())
        # show_points(true_coords.squeeze(0), points_labels.squeeze(0), plt.gca())
        # plt.axis('off')
        # plt.show() 
    return torch.cat(gt_masks,dim=0),torch.cat(cor_images,dim=0)



def SEEM_label_mask_generate(param:ParamCollection,nodes:List[MainNode]):
    """ 
    Using SEEM to generate gt label mask
    Return: gt_masks in shape (B=node_num,1,H,W)
    
    """
    model=init_model().to(param.run.device)
    gt_masks=[]
    cor_images=[]
    for node in nodes:
        img=node.image.to(param.run.device)
        cor_images.append(img)
        img=(img*255.0).type(torch.uint8)
        reproj_mask=node.supervision_signal_valid[0]
        
        # Find the indices where reproj_mask is True
        true_indices = torch.where(reproj_mask)
        true_coords = torch.stack((true_indices[1], true_indices[0]),dim=1).unsqueeze(0) # (x, y) format
        B, num, _ = true_coords.shape
 
        # combine the furtherst points with the randomly sampled points
        far_coords=sample_furthest_points(true_coords, 2).to(param.run.device)
        num_points_to_sample = min(10, num)
        # sampled_true_coords = torch.zeros(B, num_points_to_sample, 2)
        rand_indices = torch.randperm(num)[:num_points_to_sample]
        sampled_true_coords=true_coords[:,rand_indices,:].to(param.run.device)
        pairs=[]
        for i in range(num_points_to_sample):
            pair=sample_furthest_points(true_coords, 2, given_point=sampled_true_coords[:,i,:]).to(param.run.device)
            pairs.append(pair)
        pairs=torch.cat(pairs,dim=1)
        true_coords=sampled_true_coords.to(param.run.device)
        true_coords=torch.cat((true_coords,far_coords,pairs),dim=1)
        
        # generate reproj_mask based on true_coords
        gt_mask_pts=torch.zeros_like(reproj_mask.unsqueeze(0).unsqueeze(0)).type(torch.int)
        for i in range(true_coords.shape[1]):
            input_mask=torch.zeros_like(reproj_mask.unsqueeze(0).unsqueeze(0)).type(torch.bool)
            x,y=true_coords[0,i,:]
            input_mask[0,0,y,x]=True
            masks,texts=inference(model,img,input_mask)
            # if has multiple masks, add them into one
            gt_mask=(torch.sum(masks,dim=0)>0).unsqueeze(0).unsqueeze(0)
            gt_mask_pts+=gt_mask.type(torch.int)
            gt_mask_pts[gt_mask_pts>0]=1
        gt_mask=gt_mask_pts.type(torch.bool)
        gt_masks.append(gt_mask)
        torch.cuda.empty_cache()
        # masks,texts=inference(model,img,reproj_mask)
        # # if has multiple masks, add them into one
        # mask=torch.sum(masks,dim=0)>0
        # gt_masks.append(mask.unsqueeze(0).unsqueeze(0))
        
        # plt.figure(figsize=(10,10))
        # input_img=(img.squeeze(0).permute(1,2,0).cpu().numpy())
        # plt.imshow(input_img)
        # H,W,C=input_img.shape
        # mask_ratio=gt_mask.sum()/(H*W)
        # print("Mask region ratio: ",mask_ratio.item())
        # show_mask(gt_mask.squeeze(0), plt.gca())
        # show_mask(reproj_mask.squeeze(0), plt.gca(), random_color=True)
        # plt.axis('off')
        # plt.show()
    return torch.cat(gt_masks,dim=0),torch.cat(cor_images,dim=0)


def conf_mask_generate(param:ParamCollection,
                      nodes:List[MainNode],
                      feat_extractor:FeatureExtractor,
                      model:pl.LightningModule,
                      gt_masks:torch.Tensor,
                      ):
    """ 
    Here we use the model to generate confidence mask for each node
    Also the loss_recon is used to compute uncertainty histograms
    And the loss of physical param prediction (mean+std) tested on all recorded nodes
    
    
    """
    conf_masks=[]
    reproj_masks=[]
    losses=[]
    ori_imgs=[]
    all_fric_losses = []
    all_stiff_losses = []
    folder_path=os.path.join(WVN_ROOT_DIR,param.offline.ckpt_parent_folder,model.time)
    for i,node in enumerate( nodes):
        img=node.image.to(param.run.device)
        reproj_mask=node.supervision_signal_valid[0].unsqueeze(0).unsqueeze(0).to(param.run.device)
        reproj_masks.append(reproj_mask)
        ori_imgs.append(img)
        B,C,H,W=img.shape
        feat_extractor.set_original_size(W,H)
        res_dict=compute_phy_mask(img,feat_extractor,
                                model.model,
                                model.loss_fn,
                                param.loss.confidence_threshold,
                                param.loss.confidence_mode,
                                param.offline.plot_nodes,
                                i,
                                time=model.time,
                                image_name="node"+str(node.timestamp),
                                param=param,
                                label_mask=node._supervision_mask,)
        conf_mask=res_dict['conf_mask']
        loss_reco=res_dict['loss_reco']
        pred_phy_mask=res_dict['output_phy']
        ori_phy_mask= node._supervision_mask.to(param.run.device)
        # calculate phy loss
        phy_loss_dict=compute_pred_phy_loss(img,conf_mask,pred_phy_mask=pred_phy_mask,ori_phy_mask=ori_phy_mask)
        all_fric_losses.append(phy_loss_dict['fric_loss_raw'])
        all_stiff_losses.append(phy_loss_dict['stiff_loss_raw'])
        if param.offline.plot_hist:
            calculate_uncertainty_plot(loss_reco,conf_mask,reproj_mask,os.path.join(folder_path,'hist',f'node_{i}_uncertainty_histogram.png'))
            calculate_uncertainty_plot(loss_reco,gt_masks[i,:,:,:].unsqueeze(0),reproj_mask,os.path.join(folder_path,'hist/gt',f'node_{i}_gt_uncertainty_histogram.png'))
        conf_masks.append(conf_mask)
        losses.append(loss_reco)
        torch.cuda.empty_cache()
        
        loss_reco_raw=res_dict['loss_reco_raw']
        conf_mask_raw=res_dict['conf_mask_raw']
        if param.offline.plot_tsne:
            plot_tsne(conf_mask_raw, loss_reco_raw, title=f'node_{i}_t-SNE with Confidence Highlighting',path=os.path.join(folder_path,'tsne'))
    all_reproj_masks=torch.cat(reproj_masks,dim=0)
    torch.cuda.empty_cache()
    all_losses=torch.cat(losses,dim=0)
    all_conf_masks=torch.cat(conf_masks,dim=0)
    if param.offline.plot_hist:
        calculate_uncertainty_plot(all_losses,all_conf_masks,all_reproj_masks,os.path.join(folder_path,'hist','all_uncertainty_histogram.png'))
    
    all_fric_losses = torch.cat(all_fric_losses)
    all_stiff_losses = torch.cat(all_stiff_losses)
    fric_mean = torch.mean(all_fric_losses)
    fric_std = torch.std(all_fric_losses)
    stiff_mean = torch.mean(all_stiff_losses)
    stiff_std = torch.std(all_stiff_losses)

    #Save the Results to a Text File
    file_path = os.path.join(folder_path, 'overall_pred_loss_statistics.txt')
    with open(file_path, 'a') as file:
        file.write(f"{param.general.name}\n")
        file.write(f"Overall Friction Error Mean: {round(fric_mean.item(),3)}, Standard Deviation: {round(fric_std.item(),3)}\n")
        file.write(f"Overall Stiffness Error Mean: {round(stiff_mean.item(),3)}, Standard Deviation: {round(stiff_std.item(),3)}\n")

    print("Overall loss statistics saved to overall_loss_statistics.txt")

    torch.cuda.empty_cache()
    return {"all_conf_masks":all_conf_masks,
            "ori_imgs":torch.cat(ori_imgs,dim=0),
            "loss_fric_mean+std":(fric_mean,fric_std),
            "loss_stiff_mean+std":(stiff_mean,stiff_std),}

def calculate_uncertainty_plot(all_losses:torch.Tensor,all_conf_masks:torch.Tensor,all_reproj_masks:torch.Tensor=None,save_path=None):
    """ 
    Calculate a histogram of the uncertainty values (losses) from reproj_masks(should be very certain)
    and from conf_masks
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # bin_size = 0.05
    num_bins = 40
    # Flatten the tensors to get the loss values of all pixels
    flattened_losses = all_losses.flatten().detach().cpu().numpy()
    flattened_conf_masks = all_conf_masks.flatten().detach().cpu().numpy().astype(bool)
    if all_reproj_masks is not None:
        flattened_reproj_masks = all_reproj_masks.flatten().detach().cpu().numpy().astype(bool)

    # Use boolean indexing to filter losses
    conf_mask_losses = flattened_losses[flattened_conf_masks]
    if all_reproj_masks is not None:
        reproj_mask_losses = flattened_losses[flattened_reproj_masks]

    unconf_mask_losses = flattened_losses[~flattened_conf_masks]
    # Sample a fixed number of points from all losses to balance the scale
    # sample_size = min(10000, len(flattened_losses))
    # losses_sampled = np.random.choice(flattened_losses, sample_size, replace=False)

    # sample_size = min(3000, len(flattened_losses))
    # conf_sampled = np.random.choice(conf_mask_losses, sample_size, replace=False)
    # sample_size = min(1000, len(flattened_losses))
    # reproj_sampled = np.random.choice(reproj_mask_losses, sample_size, replace=False)
    if all_reproj_masks is not None:
        # Define the bin edges based on the minimum and maximum loss values and the desired bin size
        min_loss = min(flattened_losses.min(), conf_mask_losses.min(), reproj_mask_losses.min())
        max_loss = max(flattened_losses.max(), conf_mask_losses.max(), reproj_mask_losses.max())
    else:
        min_loss = min(flattened_losses.min(), conf_mask_losses.min())
        max_loss = max(flattened_losses.max(), conf_mask_losses.max())
    # bins = np.arange(min_loss, max_loss + bin_size, bin_size)
    bins = np.linspace(min_loss, max_loss, num_bins)
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    
    # Histogram for all losses in grey
    # plt.hist(flattened_losses, bins, color='grey', alpha=1.0, label='Sampled All Losses',density=False)
    plt.hist(unconf_mask_losses, bins, color='red', alpha=0.7, label='Un-confident Mask',density=False)
    # Histogram for conf_mask_losses in orange
    plt.hist(conf_mask_losses, bins, color='orange', alpha=0.7, label='Confidence Mask',density=False)

    if all_reproj_masks is not None:
        # Histogram for reproj_mask_losses in blue
        plt.hist(reproj_mask_losses, bins, color='blue', alpha=0.5, label='Reprojection Mask',density=False)


    # Add labels and title
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Uncertainty')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    # plt.show()
    plt.close()




def plot_masks_compare(gt_masks:torch.Tensor,conf_masks:torch.Tensor,images:torch.Tensor,file_path,layout_type='side_by_side',param=None):
    """
    Plot ground truth masks, confidence masks, and images side by side and save to file.

    Args:
    gt_masks (torch.Tensor): Ground truth masks (B, 1, H, W).
    conf_masks (torch.Tensor): Confidence masks (B, 1, H, W).
    images (torch.Tensor): Corresponding images (B, 3, H, W).
    file_path (str): Path to save the file.
    """

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    all_img_list=[]
    
    for i in range(gt_masks.shape[0]):
        img=plot_image(images[i].squeeze(0))
        output_gt=plot_overlay_image(images[i].unsqueeze(0),alpha=0.7,
                           overlay_mask=gt_masks[i],
                           channel=0,
                           cmap='coolwarm',
                           )
        output_conf=plot_overlay_image(images[i].unsqueeze(0),alpha=0.7,
                           overlay_mask=conf_masks[i],
                           channel=0,
                           cmap='coolwarm'
                           )
        img_list=[img,output_gt,output_conf]
        title_list=['Original Image','GT Mask','Confidence Mask']
        if layout_type == 'side_by_side':
            plot_images_side_by_side(img_list, title_list, save_path=os.path.join(file_path, 'node' + str(i) + '.png'))
        elif layout_type == 'grid':
            all_img_list.extend(img_list)
        torch.cuda.empty_cache()
        
    if layout_type == 'grid':
    # Assuming all images have the same dimensions
        rows=5
        cols = 3  # as there are 3 types of images
        num_images_per_chunk = rows * cols
        for i in range(0, len(all_img_list), num_images_per_chunk):
            chunk = all_img_list[i:i + num_images_per_chunk]
            chunk_titles = title_list * rows
            plot_images_in_grid(chunk, chunk_titles, rows,cols, save_path=os.path.join(file_path, 'chunk' + str(int(i/cols)) + '.png'), show_plot=False,param=param)
            print("Saved fig for chunk", int(i/cols))
        
   
def masks_stats(gt_masks:torch.Tensor,conf_masks:torch.Tensor, output_file='stats.txt',name="debug"):
    """ 
    Here we calculate the over-confidence and under-confidence statistics tested on all nodes
    """
    
    H,W=gt_masks.shape[-2:]
    delta=conf_masks.type(torch.int)-gt_masks.type(torch.int)
    with open(output_file, 'a') as file:
        # Calculate over-confidence
        diff_mask = torch.clamp(delta, min=0)
        ones_count = diff_mask.type(torch.float32).sum(dim=[2, 3])
        total_elements = H * W
        deviation = ones_count / total_elements * 100.0
        over_conf_mean = deviation.mean().item()
        over_conf_std = deviation.std().item()
        file.write(name+"\n")
        print(f'Average Over-confidence: {round(over_conf_mean,3)}%, Std. Dev: {round(over_conf_std,3)}%')
        over_conf_stats = f'Average Over-confidence: {round(over_conf_mean, 3)}%, Std. Dev: {round(over_conf_std, 3)}%\n'
        file.write(over_conf_stats)
        # Calculate under-confidence
        diff_mask = torch.clamp(delta, max=0)
        m_ones_count = diff_mask.type(torch.float32).sum(dim=[2, 3])
        m_deviation = -m_ones_count / total_elements * 100.0
        under_conf_mean = m_deviation.mean().item()
        under_conf_std = m_deviation.std().item()
        print(f'Average Under-confidence: {round(under_conf_mean,3)}%, Std. Dev: {round(under_conf_std,3)}%')
        under_conf_stats = f'Average Under-confidence: {round(under_conf_mean, 3)}%, Std. Dev: {round(under_conf_std, 3)}%\n'
        file.write(under_conf_stats)
    return {"over_conf_mean":over_conf_mean,
            "over_conf_std":over_conf_std,
            "under_conf_mean":under_conf_mean,
            "under_conf_std":under_conf_std,}
        
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
