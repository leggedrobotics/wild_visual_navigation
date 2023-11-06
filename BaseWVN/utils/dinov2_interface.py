from BaseWVN import WVN_ROOT_DIR
import torch
from torchvision import transforms as T
import time
from typing import Tuple
class Dinov2Interface:
    def __init__(
        self,
        device: str,
        input_size: int = 448,
        original_size: Tuple[int, int] = (1920, 1280),
        input_interp: str = "bilinear",
        model_type: str = "vit_small",
        **kwargs
    ):
       
        self._model_type = model_type
        # Initialize DINOv2
        if self._model_type == "vit_small":
            self.model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.embed_dim = 384
        elif self._model_type == "vit_base":
            self.model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.embed_dim = 768
        elif self._model_type == "vit_large":
            self.model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.embed_dim = 1024
        elif self._model_type == "vit_huge":
            self.model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            self.embed_dim = 1536
        self.patch_size = 14
        # Send to device
        self.model.to(device)
        self.device = device

        self._input_size = input_size
        self._input_interp = input_interp
        self.original_width = original_size[0]
        self.original_height = original_size[1]
        # Interpolation type
        if self._input_interp == "bilinear":
            self.interp = T.InterpolationMode.BILINEAR
        elif self._input_interp == "nearest":
            self.interp = T.InterpolationMode.NEAREST
        elif self._input_interp == "bicubic":
            self.interp = T.InterpolationMode.BICUBIC
        self.center_crop=kwargs.get('center_crop', True)
        self.transform=self._create_transform()

    def _create_transform(self):
        # Calculate aspect ratio preserving size
        aspect_ratio = self.original_width / self.original_height
        if aspect_ratio > 1:  # If width > height, scale by height (1280 -> 448)
            new_height = self._input_size
            new_width = int(new_height * aspect_ratio)
            # check if new_width is a multiple of self.patch_size
            if new_width % self.patch_size != 0:
                new_width=new_width-new_width%self.patch_size
        else:  # If height >= width, scale by width (1920 -> 448)
            new_width = self._input_size
            new_height = int(new_width / aspect_ratio)
            # check if new_height is a multiple of self.patch_size
            if new_height % self.patch_size != 0:
                new_height=new_height-new_height%self.patch_size

        # Resize and then center crop to the expected input size
        transform = T.Compose([
            T.Resize((new_height, new_width),self.interp),
            T.CenterCrop(self._input_size) if self.center_crop else T.CenterCrop((new_height, new_width)),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.new_height=new_height
        self.new_width=new_width
        return transform
    
    @torch.no_grad()
    def inference(self,img:torch.tensor):
        # check if it has a batch dim or not
        if img.dim()==3:
            img=img.unsqueeze(0)
             
        # Resize and normalize
        img = self.transform(img)
        # Send to device
        img = img.to(self.device)
        # print("After transform shape is:",img.shape)
        # Inference
        feat = self.model.forward_features(img)["x_norm_patchtokens"]
        return feat
    
    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.model.to(device)
        self.device = device
    
if __name__=="__main__":
    import os
    import cv2


    # Initialize the interface with the desired device and parameters
    # Load an image and convert it to a PyTorch tensor
    image_relative_path = 'image/hiking_2.jpeg'  # Update to the relative path of your image
    feat_relative_path = 'image/hiking_2_feat.png'
    # Use os.path.join to get the full path of the image
    image_path = os.path.join(WVN_ROOT_DIR, image_relative_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np_img = cv2.imread(image_path)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    
    input_size = 1260
    # If you have a GPU with CUDA support, use 'cuda', otherwise 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # W,H for original image size input
    dinov2_interface = Dinov2Interface(device=device, model_type="vit_small", input_size=input_size, original_size=(img.shape[-1], img.shape[-2]), input_interp="bilinear",center_crop=False)

 
    start_time = time.time()
    # Perform inference
    features = dinov2_interface.inference(img)

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    inference_time = end_time - start_time

    # Print the resulting feature tensor and inference time
    print("Feature shape:", features.shape)
    print("Inference time with transform: {:.3f} seconds".format(inference_time))

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    features=features[0].cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(features)

    pca_features = pca.transform(features)
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
    # pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    feat_height = int(input_size / 14)
    feat_width = int(input_size/img.shape[-2]*img.shape[-1] / 14) 
    plt.imshow(pca_features.reshape(feat_height, feat_width, 3).astype(np.uint8))
    image_path = os.path.join(WVN_ROOT_DIR, feat_relative_path)
    plt.savefig(image_path)

        