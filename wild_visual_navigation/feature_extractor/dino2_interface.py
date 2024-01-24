from wild_visual_navigation import WVN_ROOT_DIR
import torch
from torchvision import transforms as T
import torch.nn.functional as F
import time
from typing import Tuple
from torch.cuda.amp import autocast


class Dino2Interface:
    def __init__(
            self,
            device: str,
            model_type: str = "vit_small",
            **kwargs
    ):

        self._model_type = model_type
        # Initialize DINOv2
        if self._model_type == "vit_small":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.embed_dim = 384
        elif self._model_type == "vit_base":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            self.embed_dim = 768
        elif self._model_type == "vit_large":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            self.embed_dim = 1024
        elif self._model_type == "vit_huge":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
            self.embed_dim = 1536
        self.patch_size = kwargs.get("patch_size", 14)
        # Send to device
        self.model.to(device)
        self.device = device

        self.transform = self._create_transform()

    def _create_transform(self):
        # Resize and then center crop to the expected input size
        transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform

    @torch.no_grad()
    def inference(self, img: torch.tensor):
        # check if it has a batch dim or not
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Resize and normalize
        img = self.transform(img)
        # Send to device
        img = img.to(self.device)
        # print("After transform shape is:",img.shape)
        # Inference
        with autocast():
            feat = self.model.forward_features(img)["x_norm_patchtokens"]
        B = feat.shape[0]
        C = feat.shape[2]
        H = int(img.shape[2] / self.patch_size)
        W = int(img.shape[3] / self.patch_size)
        feat = feat.permute(0, 2, 1)
        feat = feat.reshape(B, C, H, W)

        # resize and interpolate features
        B, D, H, W = img.shape
        new_size = (H, H)
        pad = int((W - H) / 2)
        feat = F.interpolate(feat, new_size, mode="bilinear", align_corners=True)
        feat = F.pad(feat, pad=[pad, pad, 0, 0])

        return feat

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.model.to(device)
        self.device = device
