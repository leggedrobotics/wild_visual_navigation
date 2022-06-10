from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import torch.nn.functional as F
import torch
import wget

from stego.src import LitUnsupervisedSegmenter
from stego.src import get_transform
from stego.src import dense_crf


class StegoInterface:
    def __init__(self, device):
        self.model = self.load()
        self.model.to(device)
        self.device = device
        self.transform = get_transform(448, False, "center")

    def load(self):
        """Loads model.

        Returns:
            model (stego.src.train_segmentation.LitUnsupervisedSegmenter): Pretrained model
        """
        model_path = join(WVN_ROOT_DIR, "assets", "stego", "cocostuff27_vit_base_5.ckpt")
        if not os.path.exists(model_path):
            saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
            saved_model_name = "cocostuff27_vit_base_5.ckpt"
            wget.download(saved_model_url_root + saved_model_name, model_path)

        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        return model

    @torch.no_grad()
    def inference(self, img):
        """Performance inference using stego

        Args:
            img (np.array, dtype=np.uint8 or PIL.Image.Image): Input image

        Returns:
            linear_pred (torch.tensor, dtype=torch.int64): Linear prediction
            cluster_pred (torch.tensor, dtype=torch.int64): Cluster prediction
        """
        img = self.transform(img).unsqueeze(0).to(self.device)
        code1 = self.model(img)
        code2 = self.model(img.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode="bilinear", align_corners=False)
        linear_probs = torch.log_softmax(self.model.linear_probe(code), dim=1).cpu()

        cluster_probs = self.model.cluster_probe(code, 2, log_probs=True).cpu()

        single_img = img[0].cpu()
        linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

        return linear_pred, cluster_pred


def test_stego_interfacer():
    """Performance inference using stego and stores result as an image."""

    from PIL import Image
    from wild_visual_navigation.utils import get_img_from_fig
    import matplotlib.pyplot as plt
    from stego.src import unnorm, remove_axes
    import numpy as np

    # Create test directory
    os.makedirs(join(WVN_ROOT_DIR, "results", "test_stego_interfacer"), exist_ok=True)

    # Inference model
    si = StegoInterface(device="cuda")
    img = Image.open(join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    linear_pred, cluster_pred = si.inference(img)

    # Plot result as in colab
    fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 5))

    ax[0].imshow(unnorm(si.transform(img)).permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[1].imshow(si.model.label_cmap[cluster_pred])
    ax[1].set_title("Cluster Predictions")
    ax[2].imshow(si.model.label_cmap[linear_pred])
    ax[2].set_title("Linear Probe Predictions")
    remove_axes(ax)

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(join(WVN_ROOT_DIR, "results", "test_stego_interfacer", "forest_clean_stego.png"))


if __name__ == "__main__":
    test_stego_interfacer()
