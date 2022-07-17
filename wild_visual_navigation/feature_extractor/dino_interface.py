from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import torch.nn.functional as F
import torch
import wget
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms as T
from stego.src.train_segmentation import DinoFeaturizer

from PIL import Image


class DinoInterface:
    def __init__(self, device: str):
        self.dim = 90
        self.cfg = DictConfig(
            {
                "dino_patch_size": 8,
                "dino_feat_type": "feat",
                "model_type": "vit_base",  # vit_small
                "projection_type": "nonlinear",
                "pretrained_weights": None,
                "dropout": True,
            }
        )

        if self.cfg.pretrained_weights is None:
            self.cfg.pretrained_weights = self.download_pretrained_model(self.cfg)

        self.model = DinoFeaturizer(self.dim, self.cfg)
        self.model.to(device)
        self.device = device

        self.transform = T.Compose(
            [T.Resize(448, Image.NEAREST), T.CenterCrop(448), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def download_pretrained_model(self, cfg: DictConfig):
        """Loads model.

        Returns:
            model (stego.src.train_segmentation.DinoFeaturizer): Pretrained model
        """

        arch = cfg.model_type
        patch_size = cfg.dino_patch_size

        if arch == "vit_small" and patch_size == 16:
            model = "dino_deitsmall16_pretrain"
        elif arch == "vit_small" and patch_size == 8:
            model = "dino_deitsmall8_300ep_pretrain"
        elif arch == "vit_base" and patch_size == 16:
            model = "dino_vitbase16_pretrain"
        elif arch == "vit_base" and patch_size == 8:
            model = "dino_vitbase8_pretrain"
        else:
            raise ValueError("Unknown arch and patch size")

        url = f"{model}/{model}.pth"

        # Download model
        model_path = join(WVN_ROOT_DIR, "assets", "dino")
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url, model_dir=model_path
        )

        return join(model_path, f"{model}.pth")

    def get_feature_dim(self):
        return self.dim

    @torch.no_grad()
    def inference(self, img: torch.tensor, interpolate: bool = False):
        """Performance inference using DINO
        Args:
            img (torch.tensor, dtype=type.torch.float32, shape=(BS,3,H.W)): Input image

        Returns:
            features (torch.tensor, dtype=torch.float32, shape=(BS,D,H,W)): per-pixel D-dimensional features
        """
        assert 1 == img.shape[0]
        assert img.device.type == self.device

        # Extract features
        features = self.model(img)[1]

        # resize and interpolate features
        # features = F.interpolate(features, img.shape[-2:], mode="bilinear", align_corners=True)

        return features


def run_dino_interfacer():
    """Performance inference using stego and stores result as an image."""

    from PIL import Image
    from wild_visual_navigation.utils import get_img_from_fig
    import matplotlib.pyplot as plt
    from stego.src import unnorm, remove_axes
    import numpy as np
    import kornia as K

    # Create test directory
    os.makedirs(join(WVN_ROOT_DIR, "results", "test_dino_interfacer"), exist_ok=True)

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    di = DinoInterface(device=device)
    p = join(WVN_ROOT_DIR, "assets/images/forest_clean.png")
    img = K.io.load_image(p, desired_type=K.io.ImageLoadType.RGB8, device=device)
    img = (img.type(torch.float32) / 255)[None]

    # Inference with DINO
    feat_dino = di.inference(di.transform(img), interpolate=False)

    # Fix size of DINO features to match input image's size
    B, D, H, W = img.shape
    new_size = (H, H)
    pad = int((W - H) / 2)
    feat_dino = F.interpolate(feat_dino, new_size, mode="bilinear", align_corners=True)
    feat_dino = F.pad(feat_dino, pad=[pad, pad, 0, 0])

    # Plot result as in colab
    fig, ax = plt.subplots(10, 11, figsize=(5 * 11, 5 * 11))

    for i in range(10):
        for j in range(11):
            if i == 0 and j == 0:
                continue

            elif (i == 0 and j != 0) or (i != 0 and j == 0):
                ax[i][j].imshow(img.permute(0, 2, 3, 1)[0].cpu())
                ax[i][j].set_title("Image")
            else:
                n = (i - 1) * 10 + (j - 1)
                if n >= di.get_feature_dim():
                    break
                ax[i][j].imshow(feat_dino[0][n].cpu(), cmap=plt.colormaps.get("inferno"))
                ax[i][j].set_title("Features [0]")
    remove_axes(ax)

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(join(WVN_ROOT_DIR, "results", "test_dino_interfacer", "forest_clean_dino.png"))


if __name__ == "__main__":
    run_dino_interfacer()
