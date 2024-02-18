#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
from os.path import join
from omegaconf import DictConfig
from torchvision import transforms as T
from collections import namedtuple, OrderedDict
import numpy as np
import os
import torch.nn.functional as F
import torch
import tensorrt as trt


class TrtModel:
    def __init__(self, engine_path: str, device: str = "cuda"):
        self.engine_path = engine_path
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)

        self.bindings = OrderedDict()
        self.named_binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.allocate_buffers()

        # Initialize execution context
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.engine.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = self.named_binding(name, dtype, shape, data, int(data.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def __call__(self, x: torch.tensor, batch_size=1):
        # Note: the bindings' entries depend on the model, in this case
        # they're defined in DINO's ONNX model
        self.binding_addrs["input"] = int(x.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        features = self.bindings["features"].data
        code = self.bindings["code"].data
        return features, code


class DinoTrtInterface:
    def __init__(
        self,
        trt_model_path: str = os.path.join(WVN_ROOT_DIR, "assets/dino/dino_exported.trt"),
        device: str = "cuda",
    ):
        self.device = device
        self.dim = 90
        self.cfg = DictConfig(
            {
                "dino_patch_size": 8,
                "dino_feat_type": "feat",
                "model_type": "vit_small",  # vit_small
                "projection_type": "nonlinear",
                "pretrained_weights": None,
                "dropout": True,
            }
        )

        self.model = TrtModel(trt_model_path, device=device)

        # Transformation for testing
        self.transform = T.Compose(
            [
                T.Resize(224, T.InterpolationMode.NEAREST),
                T.CenterCrop(224),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.model.to(device)
        self.device = device

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

        B, D, H, W = img.shape
        img = self.transform(img).to(self.device)

        # Extract DINO features using TensorRT model
        _, features = self.model(img)

        # resize and interpolate features
        new_size = (H, H)
        pad = int((W - H) / 2)
        features = F.interpolate(features, new_size, mode="bilinear", align_corners=True)
        features = F.pad(features, pad=[pad, pad, 0, 0])

        return features


def run_dino_trt_interfacer():
    """Performance inference using stego and stores result as an image."""

    from wild_visual_navigation.visu import get_img_from_fig
    from wild_visual_navigation.testing import load_test_image, get_dino_transform
    from wild_visual_navigation.utils.testing import make_results_folder
    import matplotlib.pyplot as plt
    from stego.src import remove_axes

    # Create test directory
    outpath = make_results_folder("test_dino_trt_interfacer")

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    di = DinoTrtInterface(device=device)
    img = load_test_image().to(device)
    transform = get_dino_transform()

    # Inference with DINO
    feat_dino = di.inference(transform(img), interpolate=False)

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
    img.save(join(outpath, "forest_clean_dino.png"))


if __name__ == "__main__":
    run_dino_trt_interfacer()
