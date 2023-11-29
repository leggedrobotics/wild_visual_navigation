import os
import torchvision.transforms.functional as F
from PIL import Image
import torch
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import json
import re
import matplotlib.pyplot as plt
from BaseWVN import WVN_ROOT_DIR
from BaseWVN.config.wvn_cfg import ParamCollection

api_key = '2fea959a3e8dc0ae5cd55e40d5f60c8e0bdc70d7'
client = SegmentsClient(api_key)
dataset_identifier = "swsychen/val_snow"
name = "v0.1"
client.add_release(dataset_identifier, name)
def save_imperfect_images(image_batch, mask_batch, imperfect_indexes, save_folder):
    """
    Save images with imperfect masks to a folder.

    :param image_batch: Batch of images (B, C, H, W)
    :param mask_batch: Corresponding batch of masks (B, 1, H, W)
    :param imperfect_indexes: List of indexes of imperfect masks
    :param save_folder: Folder to save the images
    """
    os.makedirs(save_folder, exist_ok=True)

    for idx in imperfect_indexes:
        image = image_batch[idx]
        # Convert from tensor to PIL image
        image_pil = F.to_pil_image(image)
        image_pil.save(os.path.join(save_folder, f"imperfect_image_{idx}.png"))

def download_and_replace_masks(mask_batch, corrected_mask_paths):
    """
    Replace the original imperfect masks with the corrected ones.

    :param mask_batch: The original batch of masks (B, 1, H, W)
    :param corrected_mask_paths: List of file paths to the corrected mask images
    :param imperfect_indexes: List of indexes of the masks to replace
    """
    
    release = client.get_release(dataset_identifier,name)
    dataset = SegmentsDataset(release, labelset='ground-truth')
    for sample in dataset:
        # Extract index from sample name (assuming format 'imperfect_image_INDEX.png')
        match = re.search(r'imperfect_image_(\d+)\.png', sample['name'])
        if not match:
            print(f"Could not extract index from sample name: {sample['name']}")
            continue  
        idx = int(match.group(1))
        # Show the instance segmentation label
        corrected_mask_tensor = F.to_tensor(sample['segmentation_bitmap']).type(torch.bool)  # Add channel dimension
        # Replace the corresponding mask in mask_batch
        if idx < len(mask_batch):
            mask_batch[idx] = corrected_mask_tensor
        else:
            print(f"Index {idx} is out of bounds for the mask batch.")
    output_path=os.path.join(corrected_mask_paths, 'gt_masks_SAM.pt')
    torch.save(mask_batch, output_path)
    print(f"Saved corrected masks to {output_path}")

if __name__ == "__main__":
    param=ParamCollection()
    output_dir = os.path.join(WVN_ROOT_DIR, param.offline.data_folder,'val',param.offline.env)

    gt_masks_path = os.path.join(output_dir, 'gt_masks_SAM.pt')
    img_path=os.path.join(output_dir, 'mask_img.pt')

    gt_masks = torch.load(gt_masks_path)
    mask_imgs=torch.load(img_path)
    imperfect_indexes = [1,2,5,7,8,16,18,20,21,22,23,24,25,26,35,36,39,41,44,45]
    # imperfect_indexes = [78,79]
    # imperfect_indexes = [0,1,2,3,4,13,14,15,16,18,19,20,21,25,26,27,28,29,30,31,32,33,34,35,36,39,56,59,60,61,62,63,64,65,75,80,81,82]
    # imperfect_indexes = [5,6,7,8,9,10,11,23,24,26,31,34,35,36,37,38,39,48,49,51,57,63,64,65,66,67,68,69,80]
    # imperfect_indexes = [17,18,19,20,24,54,56,67]
    # imperfect_indexes = [37,38]
    # save for manual correction
    # save_imperfect_images(mask_imgs, gt_masks, imperfect_indexes, os.path.join(output_dir, 'imperfect_images'))
    
    download_and_replace_masks(gt_masks,os.path.join(output_dir, 'imperfect_images'))
    