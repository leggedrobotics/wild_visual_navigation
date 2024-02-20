#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from os.path import join
import torch.nn.functional as F
import torch
from torchvision import transforms as T
from omegaconf import OmegaConf

from stego.backbones.backbone import get_backbone


class DinoInterface:
    def __init__(
        self,
        device: str,
        backbone: str = "dino",
        input_size: int = 448,
        backbone_type: str = "vit_small",
        patch_size: int = 8,
        projection_type: str = None,  # nonlinear or None
        dropout_p: float = 0,  # True or False
        pretrained_weights: str = None,
        cfg: OmegaConf = OmegaConf.create({}),
    ):
        # Load config
        if cfg.is_empty():
            self._cfg = OmegaConf.create(
                {
                    "backbone": backbone,
                    "backbone_type": backbone_type,
                    "input_size": input_size,
                    "patch_size": patch_size,
                    "projection_type": projection_type,
                    "dropout_p": dropout_p,
                    "pretrained_weights": pretrained_weights,
                }
            )
        else:
            self._cfg = cfg

        # Initialize DINO
        self._model = get_backbone(self._cfg)

        # Send to device
        self._model.to(device)
        self._device = device

        # Other
        normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._transform = T.Compose(
            [
                T.Resize(input_size, T.InterpolationMode.NEAREST),
                T.CenterCrop(input_size),
                normalization,
            ]
        )

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._model.to(device)
        self._device = device

    @torch.no_grad()
    def inference(self, img: torch.tensor):
        """Performance inference using DINO
        Args:
            img (torch.tensor, dtype=type.torch.float32, shape=(B,3,H.W)): Input image

        Returns:
            features (torch.tensor, dtype=torch.float32, shape=(B,D,H,W)): per-pixel D-dimensional features
        """

        # Resize image and normalize
        resized_img = self._transform(img).to(self._device)

        # Extract features
        features = self._model(resized_img)

        # resize and interpolate features
        B, D, H, W = img.shape
        new_features_size = (H, H)
        # pad = int((W - H) / 2)
        features = F.interpolate(features, new_features_size, mode="bilinear", align_corners=True)
        # features = F.pad(features, pad=[pad, pad, 0, 0])
        return features

    @property
    def input_size(self):
        return self._cfg.input_size

    @property
    def backbone(self):
        return self._cfg.backbone

    @property
    def backbone_type(self):
        return self._cfg.backbone_type

    @property
    def vit_patch_size(self):
        return self._cfg.patch_size


def run_dino_interfacer():
    """Performance inference using stego and stores result as an image."""

    from pytictac import Timer
    from wild_visual_navigation.visu import get_img_from_fig
    from wild_visual_navigation.utils.testing import load_test_image, make_results_folder
    import matplotlib.pyplot as plt
    from stego.utils import remove_axes

    # Create test directory
    outpath = make_results_folder("test_dino_interfacer")

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_test_image().to(device)
    img = F.interpolate(img, scale_factor=0.25)

    plot = False
    save_features = True

    # Settings
    size = 448
    model = "vit_small"
    patch = 8
    backbone = "dinov2"

    # Inference with DINO
    # Create DINO
    di = DinoInterface(
        device=device,
        backbone=backbone,
        input_size=size,
        backbone_type=model,
        patch_size=patch,
    )

    with Timer(f"DINO, input_size, {di.input_size}, model, {di.backbone_type}, patch_size, {di.vit_patch_size}"):
        feat_dino = di.inference(img)

    if save_features:
        for i in range(90):
            fig = plt.figure(frameon=False)
            fig.set_size_inches(2, 2)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(feat_dino[0][i].cpu(), cmap=plt.colormaps.get("inferno"))

            # Store results to test directory
            out_img = get_img_from_fig(fig)
            out_img.save(
                join(
                    outpath,
                    f"forest_clean_dino_feat{i:02}_{di.input_size}_{di.backbone_type}_{di.vit_patch_size}.png",
                )
            )
            plt.close("all")

    if plot:
        # Plot result as in colab
        fig, ax = plt.subplots(10, 11, figsize=(1 * 11, 1 * 11))

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
        plt.tight_layout()

        # Store results to test directory
        out_img = get_img_from_fig(fig)
        out_img.save(
            join(
                outpath,
                f"forest_clean_{di.backbone}_{di.input_size}_{di.backbone_type}_{di.vit_patch_size}.png",
            )
        )
        plt.close("all")


if __name__ == "__main__":
    run_dino_interfacer()
