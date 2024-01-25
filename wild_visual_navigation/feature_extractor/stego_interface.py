from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import torch.nn.functional as F
import torch
from torchvision import transforms as T
from omegaconf import OmegaConf

from stego import STEGO_ROOT_DIR
from stego.stego import Stego
from stego.data import create_cityscapes_colormap


class StegoInterface:
    def __init__(
        self,
        device: str,
        input_size: int = 448,
        model_path: str = f"{STEGO_ROOT_DIR}/models/stego_cocostuff27_vit_base_5_cluster_linear_fine_tuning.ckpt",
        run_crf: bool = True,
        run_clustering: bool = False,
        cfg: OmegaConf = OmegaConf.create({}),
    ):
        # Load config
        if cfg.is_empty():
            self._cfg = OmegaConf.create(
                {
                    "model_path": model_path,
                    "input_size": input_size,
                    "run_crf": run_crf,
                    "run_clustering": run_clustering,
                }
            )
        else:
            self._cfg = cfg

        self._model = Stego.load_from_checkpoint(self._cfg.model_path)
        self._model.eval().to(device)
        self._device = device

        # Colormap
        self._cmap = create_cityscapes_colormap()

        # Other
        normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._transform = T.Compose(
            [
                T.Resize(input_size, T.InterpolationMode.NEAREST),
                T.CenterCrop(input_size),
                normalization,
            ]
        )

        # Internal variables to access internal data
        self._features = None
        self._segments = None

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._model.to(device)
        self._device = device

    @torch.no_grad()
    def inference(self, img: torch.tensor):
        """Performance inference using stego
        Args:
            img (torch.tensor, dtype=type.torch.float32, shape=(BS,3,H.W)): Input image

        Returns:
            linear_probs (torch.tensor, dtype=torch.float32, shape=(BS,C,H,W)): Linear prediction
            cluster_probs (torch.tensor, dtype=torch.float32, shape=(BS,C,H,W)): Cluster prediction
        """
        # assert 1 == img.shape[0]

        # Resize image and normalize
        resized_img = self._transform(img).to(self._device)

        # Run STEGO
        self._code = self._model.get_code(resized_img)
        self._cluster_pred, self._linear_pred = self._model.postprocess(
            code=self._code,
            img=resized_img,
            use_crf_cluster=self._cfg.run_crf,
            use_crf_linear=self._cfg.run_crf,
            image_clustering=self._cfg.run_clustering,
        )

        # resize and interpolate features
        B, D, H, W = img.shape
        new_features_size = (H, H)
        # pad = int((W - H) / 2)
        self._code = F.interpolate(self._code, new_features_size, mode="bilinear", align_corners=True)
        self._cluster_pred = F.interpolate(self._cluster_pred[None].float(), new_features_size, mode="nearest").int()
        self._linear_pred = F.interpolate(self._linear_pred[None].float(), new_features_size, mode="nearest").int()

        return self._linear_pred[0], self._cluster_pred[0]

    @property
    def model(self):
        return self._model

    @property
    def cmap(self):
        return self._cmap

    @property
    def input_size(self):
        return self._cfg.input_size

    @property
    def linear_segments(self):
        return self._linear_pred

    @property
    def cluster_segments(self):
        return self._cluster_pred

    @property
    def features(self):
        return self._code


def run_stego_interfacer():
    """Performance inference using stego and stores result as an image."""

    from pytictac import Timer
    from wild_visual_navigation.visu import get_img_from_fig
    from stego.utils import remove_axes
    import matplotlib.pyplot as plt
    import cv2

    # Create test directory
    os.makedirs(join(WVN_ROOT_DIR, "results", "test_stego_interfacer"), exist_ok=True)

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    si = StegoInterface(
        device=device,
        input_size=448,
        run_crf=True,
        run_clustering=False,
    )

    p = join(WVN_ROOT_DIR, "assets/images/forest_clean.png")
    np_img = cv2.imread(p)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(np_img).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    img = F.interpolate(img, scale_factor=0.25)

    with Timer(f"Stego (input {si.input_size}"):
        linear_pred, cluster_pred = si.inference(img)

    # Plot result as in colab
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    ax[0].imshow(img[0].permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Image")
    ax[1].imshow(si.cmap[cluster_pred[0].cpu() % si.cmap.shape[0]])
    ax[1].set_title("Cluster Predictions")
    ax[2].imshow(si.cmap[linear_pred[0].cpu()])
    ax[2].set_title("Linear Probe Predictions")
    remove_axes(ax)

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(
        join(
            WVN_ROOT_DIR,
            "results",
            "test_stego_interfacer",
            f"forest_clean_stego_{si.input_size}.png",
        )
    )


if __name__ == "__main__":
    run_stego_interfacer()
