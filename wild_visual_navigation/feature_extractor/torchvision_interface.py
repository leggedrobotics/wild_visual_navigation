from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join

from torch import nn
import torch.nn.functional as F
import torch

import torchvision.models as models
from torchvision import transforms as T
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

MODEL_DIRECTORY = {method_name: getattr(models, method_name) for method_name in dir(models)
    if callable(getattr(models, method_name))}

class TorchVisionInterface(nn.Module):
    def __init__(self, device, model_type, input_size: int = 488, pretrained : bool = True):
        super().__init__()
        model_func = MODEL_DIRECTORY[model_type]
        model = model_func(pretrained=pretrained)
        self.device = device
        model.to(device)
        
        self.find_endpoints = False
        if model_type == "resnet50":
            return_nodes = {
                "layer2.0.relu" :"feat1",
                "layer3.0.relu" : "feat2",
                "layer4.0.relu" : "feat3",
                "layer4.2.relu_2": "feat4"
            }
        elif model_type == "resnet18":
            return_nodes = {
                "layer1.1.relu_1": "feat1",
                "layer2.1.relu_1": "feat2",
                "layer3.1.relu_1": "feat3",
                "layer4.1.relu_1": "feat4"
            }
        elif model_type == "efficientnet_b7":
            return_nodes = {
                "features.2.0.block.0.2": "feat1",
                "features.3.0.block.0.2": "feat2",
                "features.4.0.block.0.2": "feat3",
                "features.6.0.block.0.2": "feat4",
                "features.8.2": "feat5"
            }
        elif model_type == "efficientnet_b4":
            return_nodes = {
                "features.2.0.block.0.2": "feat1",
                "features.3.0.block.0.2": "feat2",
                "features.4.0.block.0.2": "feat3",
                "features.6.0.block.0.2": "feat4",
                "features.8.2": "feat5"
            }
        elif model_type == "efficientnet_b0":
            return_nodes = {
                "features.2.0.block.0.2": "feat1",
                "features.3.0.block.0.2": "feat2",
                "features.4.0.block.0.2": "feat3",
                "features.6.0.block.0.2": "feat4",
                "features.8.2": "feat5"
            }

        else:
            train_nodes, eval_nodes = get_graph_node_names(model)
            return_nodes = {k:k for k in eval_nodes}
            self.find_endpoints = True
                
        self.model = create_feature_extractor(model, return_nodes=return_nodes)
        
        # Transformation       
        self.transform = T.Compose(
            [
                T.Resize(input_size, T.InterpolationMode.BILINEAR),
                T.CenterCrop(input_size),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.model.eval()
        
        
    @torch.no_grad()
    def forward(self, img: torch.tensor):
        feature_pyramid = self.model(self.transform(img.to(self.device)))
        
        if self.find_endpoints:
            old_dim = None
            old_k =None
            for k,v in feature_pyramid.items():
                if old_dim is None:
                    old_dim = v.shape[2]
                else:
                    if old_dim != v.shape[2]:
                        print(old_k)
                old_k = k
                old_dim = v.shape[2]
                    
        return feature_pyramid

    @torch.no_grad()
    def inference(self, img: torch.tensor, interpolate: bool = False):
        return self(img)

def run_torch_vision_model_interfacer():
    """Performance inference using stego and stores result as an image."""

    from wild_visual_navigation.utils import Timer
    from wild_visual_navigation.visu import get_img_from_fig
    import matplotlib.pyplot as plt
    from stego.src import remove_axes
    import cv2

    # Create test directory
    # os.makedirs(join(WVN_ROOT_DIR, "results", "test_torchvision_interfacer"), exist_ok=True)

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = join(WVN_ROOT_DIR, "assets/images/forest_clean.png")
    np_img = cv2.imread(p)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]

    plot = False
    save_features = True

    size = 448
    model_type = "resnet18"
        
    di = TorchVisionInterface( model_type = model_type, input_size= 488)
    di.to(device)
    img.to(device)
    res = di(img)
    print(res)

if __name__ == "__main__":
    run_torch_vision_model_interfacer()
