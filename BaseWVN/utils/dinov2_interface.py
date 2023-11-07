from BaseWVN import WVN_ROOT_DIR
import torch
from torchvision import transforms as T
import time
from typing import Tuple
class Dinov2Interface:
    def __init__(
        self,
        device: str,
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
        self.patch_size = kwargs.get("patch_size", 14)
        # Send to device
        self.model.to(device)
        self.device = device

        self.transform=self._create_transform()

    def _create_transform(self):
        # Resize and then center crop to the expected input size
        transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
        B=feat.shape[0]
        C=feat.shape[2]
        H=int(img.shape[2]/self.patch_size)
        W=int(img.shape[3]/self.patch_size)
        feat=feat.permute(0,2,1)
        feat=feat.reshape(B,C,H,W)
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
    from PIL import Image

    # Initialize the interface with the desired device and parameters
    # Load an image and convert it to a PyTorch tensor
    image_relative_path = 'image/hiking.png'  # Update to the relative path of your image
    feat_relative_path = 'image/hiking_feat.png'
    # Use os.path.join to get the full path of the image
    image_path = os.path.join(WVN_ROOT_DIR, image_relative_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np_img = cv2.imread(image_path)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    
    # input_image = Image.open(image_path).convert('RGB')  # Convert to RGB if not already
    # input_tensor = T.ToTensor()(input_image)
    # img=input_tensor.unsqueeze(0)
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
    B,C,H,W=features.shape
    features=features.permute(0,2,3,1)
    features=features.reshape(B,H*W,C)
    features=features[0].cpu().numpy()
    n=3
    pca = PCA(n_components=n)
    pca.fit(features)

    pca_features = pca.transform(features)
    for i in range(n):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
    # pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    feat_height = int(input_size / 14)
    feat_width = int(input_size/img.shape[-2]*img.shape[-1] / 14) 
    plt.imshow(pca_features.reshape(feat_height, feat_width, n).astype(np.uint8))
    image_path = os.path.join(WVN_ROOT_DIR, feat_relative_path)
    plt.savefig(image_path)

        