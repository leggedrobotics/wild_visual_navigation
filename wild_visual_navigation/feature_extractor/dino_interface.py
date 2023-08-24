from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import torch.nn.functional as F
import torch
from omegaconf import DictConfig
from torchvision import transforms as T
from stego.src.train_segmentation import DinoFeaturizer
from kornia.filters import filter2d


class DinoInterface:
    def __init__(
        self,
        device: str,
        input_size: int = 448,
        input_interp: str = "bilinear",
        model_type: str = "vit_small",
        patch_size: int = 8,
    ):
        self.dim = 384  # 90
        self.cfg = DictConfig(
            {
                "dino_patch_size": patch_size,
                "dino_feat_type": "feat",
                "model_type": model_type,
                "projection_type": None,    # "nonlinear"
                "pretrained_weights": None,
                "dropout": False,        # True
            }
        )

        # Pretrained weights
        if self.cfg.pretrained_weights is None:
            self.cfg.pretrained_weights = self.download_pretrained_model(self.cfg)

        # Initialize DINO
        self.model = DinoFeaturizer(self.dim, self.cfg)

        # Send to device
        self.model.to(device)
        self.device = device

        self._input_size = input_size
        self._input_interp = input_interp

        # Interpolation type
        if self._input_interp == "bilinear":
            interp = T.InterpolationMode.BILINEAR
        elif self._input_interp == "nearest":
            interp = T.InterpolationMode.NEAREST

        # Transformation for testing
        self.transform = T.Compose(
            [
                T.Resize(self._input_size, interp),
                T.CenterCrop(self._input_size),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.crop = T.Compose(
            [
                T.Resize(self._input_size, interp),
                T.CenterCrop(self._input_size),
            ]
        )

        # Just normalization
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.mean_kernel = torch.ones((1, 5, 5), device=device) / 25

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.model.to(device)
        self.device = device

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
        elif arch == "resnet":
            model = "dino_resnet50_pretrain"
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
        # assert 1 == img.shape[0]
        img = self.norm(img).to(self.device)

        # Extract features
        features = self.model(img)[1]

        # resize and interpolate features
        B, D, H, W = img.shape
        new_size = (H, H)
        pad = int((W - H) / 2)
        features = F.interpolate(features, new_size, mode="bilinear", align_corners=True)
        features = F.pad(features, pad=[pad, pad, 0, 0])
        # Optionally turn on image feature smoothing
        # features = filter2d(features, self.mean_kernel, "replicate")
        return features

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_interpolation(self):
        return self._input_interp

    @property
    def model_type(self):
        return self.cfg.model_type

    @property
    def vit_patch_size(self):
        return self.cfg.dino_patch_size


def run_dino_interfacer():
    """Performance inference using stego and stores result as an image."""

    from pytictac import Timer
    from wild_visual_navigation.visu import get_img_from_fig
    import matplotlib.pyplot as plt
    from stego.src import remove_axes
    import cv2

    # Create test directory
    os.makedirs(join(WVN_ROOT_DIR, "results", "test_dino_interfacer"), exist_ok=True)

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = join(WVN_ROOT_DIR, "assets/images/forest_clean.png")
    np_img = cv2.imread(p)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]

    plot = False
    save_features = True

    # Settings
    size = 448
    interp = "bilinear"
    model = "vit_small"
    patch = 8

    # Inference with DINO
    # Create DINO
    di = DinoInterface(device=device, input_size=size, input_interp=interp, model_type=model, patch_size=patch)

    with Timer(
        f"DINO, input_size, {di.input_size}, interp, {di.input_interpolation}, model, {di.model_type}, patch_size, {di.vit_patch_size}"
    ):
        try:
            feat_dino = di.inference(di.transform(img), interpolate=False)

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
                            WVN_ROOT_DIR,
                            "results",
                            "test_dino_interfacer",
                            f"forest_clean_dino_feat{i:02}_{di.input_size}_{di.input_interpolation}_{di.model_type}_{di.vit_patch_size}.png",
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
                        WVN_ROOT_DIR,
                        "results",
                        "test_dino_interfacer",
                        f"forest_clean_dino_{di.input_size}_{di.input_interpolation}_{di.model_type}_{di.vit_patch_size}.png",
                    )
                )
                plt.close("all")

        except Exception as e:
            pass


if __name__ == "__main__":
    run_dino_interfacer()
