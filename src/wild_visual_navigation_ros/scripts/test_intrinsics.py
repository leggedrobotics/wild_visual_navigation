import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms as T
import matplotlib.pyplot as plt
from BaseWVN.utils.image_projector import ImageProjector
import cv2

def scale_intrinsic(K:torch.tensor,ratio_x,ratio_y,crop_offset_x,crop_offset_y):
    """ 
    scale the intrinsic matrix, first resize than crop!!
    """
    # dimension check of K
    if K.shape[2]!=4 or K.shape[1]!=4:
        raise ValueError("The dimension of the intrinsic matrix is not 4x4!")
    K_scaled = K.clone()
    K_scaled[:,0,0]=K[:,0,0]*ratio_x
    K_scaled[:,0,2]=K[:,0,2]*ratio_x-crop_offset_x
    K_scaled[:,1,1]=K[:,1,1]*ratio_y
    K_scaled[:,1,2]=K[:,1,2]*ratio_y-crop_offset_y
    
    return K_scaled

def transform_points(points, ratio_x, ratio_y, offset_x, offset_y):
    # Apply scaling
    scaled_points = points.clone()
    scaled_points[:, 0] *= ratio_x
    scaled_points[:, 1] *= ratio_y

    # Apply cropping offset
    scaled_points[:, 0] -= offset_x
    scaled_points[:, 1] -= offset_y

    return scaled_points

if __name__=="__main__":
    sample_dict=torch.load('check.pt')
    K_record=sample_dict['K']
    img_record=sample_dict['img']
    pose_cam=sample_dict['pose_cam_in_world']
    foot_poses=sample_dict['foot_plane']

    D=torch.tensor([0.4278669514845462, 0.11762807275762316, -0.4286132864490927, 0.46840427154501907])
    
    K_original=torch.tensor([566.0691858833568, 0.0, 753.2129743107714, 0.0, 568.6610245261098, 542.0153345189508, 0.0, 0.0, 1.0]).to(K_record.device)
    K_ori=torch.eye(4).to(K_record.device)
    K_ori[:3,:3]=K_original.reshape(3,3)
    K_ori=K_ori.unsqueeze(0)
    color = torch.ones((3,), device=K_record.device)
    transform = T.Compose([
                T.Resize((1078,1428),'bilinear'),
                T.CenterCrop((910,910)),
                T.ConvertImageDtype(torch.float),
            ])
    ratio_x=1428/1440.0
    ratio_y=1078/1080.0
    crop_offset_x=(1428-910)/2
    crop_offset_y=(1078-910)/2
    K_scaled=scale_intrinsic(K_ori,ratio_x,ratio_y,crop_offset_x,crop_offset_y)
    if torch.allclose(K_scaled,K_record):
        print('K_scaled is correct')
    
    im_ori=ImageProjector(K_ori,1078,1428)
    _,_,projected_points, valid_points=im_ori.project_and_render(pose_cam.unsqueeze(0).type(torch.float32), foot_poses, color)
    sample_point_ori=projected_points[0]
    transformed_points=transform_points(sample_point_ori,ratio_x,ratio_y,crop_offset_x,crop_offset_y)
    im=ImageProjector(K_scaled,910,910)
    _,_,projected_points, valid_points=im.project_and_render(pose_cam.unsqueeze(0).type(torch.float32), foot_poses, color)
    sample_point_scaled=projected_points[0]
    if torch.allclose(transformed_points,sample_point_scaled):
        print('transformed_points is correct')
    
    img_np = img_record.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # If your image is in RGB, convert it to BGR for OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # Your camera matrix and distortion coefficients
    K_np = K_scaled[0,:3,:3].cpu().numpy().reshape(3, 3)
    D_np = D.cpu().numpy()

    # Undistort the image
    undistorted_img = cv2.undistort(img_np, K_np, D_np)

    # Display the image
    plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    plt.title("Undistorted Image")
    plt.axis('off')
    plt.show()
    pass