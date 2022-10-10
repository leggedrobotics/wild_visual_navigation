from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import torch.nn.functional as F
import torch
import wget
from torchvision import transforms as T
from stego.src import LitUnsupervisedSegmenter

# from stego.src import get_transform
from stego.src import dense_crf


class StegoInterface:
    def __init__(self, device: str, input_size: int = 448, input_interp: str = "bilinear"):
        self.model = self.load()
        self.model.to(device)
        self.device = device
        self._input_size = input_size
        self._input_interp = input_interp

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

        # Internal variables to access internal data
        self._features = None
        self._segments = None

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.model.to(device)
        self.device = device

    def load(self):
        """Loads model.

        Returns:
            model (stego.src.train_segmentation.LitUnsupervisedSegmenter): Pretrained model
        """
        model_name = "cocostuff27_vit_base_5.ckpt"
        model_path = join(WVN_ROOT_DIR, "assets", "stego", model_name)
        if not os.path.exists(model_path):
            os.makedirs(join(WVN_ROOT_DIR, "assets", "stego"), exist_ok=True)
            saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
            saved_model_name = model_name
            wget.download(saved_model_url_root + saved_model_name, model_path)

        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        return model

    @torch.no_grad()
    def inference(self, img: torch.tensor):
        """Performance inference using stego
        Args:
            img (torch.tensor, dtype=type.torch.float32, shape=(BS,3,H.W)): Input image

        Returns:
            linear_probs (torch.tensor, dtype=torch.float32, shape=(BS,C,H,W)): Linear prediction
            cluster_probs (torch.tensor, dtype=torch.float32, shape=(BS,C,H,W)): Cluster prediction
        """
        assert 1 == img.shape[0]

        img = self.norm(img).to(self.device)
        code1 = self.model(img)
        code2 = self.model(img.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode="bilinear", align_corners=False)
        linear_probs = torch.log_softmax(self.model.linear_probe(code), dim=1)
        cluster_probs = self.model.cluster_probe(code, 2, log_probs=True)

        # Save segments
        self._code = code
        self._segments = linear_probs

        return linear_probs, cluster_probs

    @torch.no_grad()
    def inference_crf(self, img: torch.tensor):
        """
        Args:
            img (torch.tensor, dtype=type.torch.float32): Input image

        Returns:
            linear_pred (torch.tensor, dtype=torch.int64): Linear prediction
            cluster_pred (torch.tensor, dtype=torch.int64): Cluster prediction
        """

        linear_probs, cluster_probs = self.inference(img)
        single_img = img[0].cpu()
        self._linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        self._cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

        return self._linear_pred, self._cluster_pred

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_interpolation(self):
        return self._input_interp

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

    from wild_visual_navigation.utils import Timer
    from wild_visual_navigation.visu import get_img_from_fig
    import matplotlib.pyplot as plt
    from stego.src import unnorm, remove_axes
    import cv2

    # Create test directory
    os.makedirs(join(WVN_ROOT_DIR, "results", "test_stego_interfacer"), exist_ok=True)

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    si = StegoInterface(device=device, input_size=448, input_interp="bilinear")

    p = join(WVN_ROOT_DIR, "assets/images/forest_clean.png")
    np_img = cv2.imread(p)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]

    with Timer(f"Stego (input {si.input_size}, interp: {si.input_interpolation})"):
        linear_pred, cluster_pred = si.inference_crf(si.crop(img))

    # Plot result as in colab
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    ax[0].imshow(si.crop(img).permute(0, 2, 3, 1)[0].cpu())
    ax[0].set_title("Image")
    ax[1].imshow(si.model.label_cmap[cluster_pred])
    ax[1].set_title("Cluster Predictions")
    ax[2].imshow(si.model.label_cmap[linear_pred])
    ax[2].set_title("Linear Probe Predictions")
    remove_axes(ax)

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(
        join(
            WVN_ROOT_DIR,
            "results",
            "test_stego_interfacer",
            f"forest_clean_stego_{si.input_size}_{si.input_interpolation}.png",
        )
    )


if __name__ == "__main__":
    run_stego_interfacer()
