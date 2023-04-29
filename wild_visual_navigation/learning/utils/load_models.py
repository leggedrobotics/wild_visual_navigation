import torch

from ..model.rnvp import LinearRNVP
from ..model.mlp import BinaryClassification
from wild_visual_navigation.feature_extractor.feature_extractor import FeatureExtractor


class Timer:
    def __init__(self, name="") -> None:
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"Time {self.name}: ", self.toc(), "ms")

    def tic(self):
        self.start.record()

    def toc(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


def load_model_1cls(model_path, device, input_dim=384, coupling_topology=200, flow_n=10, batch_norm=True,
               mask_type='odds', conditioning_size=0, use_permutation=True, single_function=True):

    # Load model
    model = LinearRNVP(input_dim=input_dim, coupling_topology=[coupling_topology], flow_n=flow_n, batch_norm=batch_norm,
                          mask_type=mask_type, conditioning_size=conditioning_size,
                          use_permutation=use_permutation, single_function=single_function)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except:
        raise ValueError("This model configuration does not exist!")

    model.eval()
    model.to(device)

    return model


def load_model_2cls(model_path, device, feat_size, batch_size):
    # Load model
    model = BinaryClassification(feat_size, batch_size)

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except:
        raise ValueError("This model configuration does not exist!")

    print("Loaded model...")

    model.eval()
    model.to(device)

    return model


def create_extractor(device, superpixel_mode="slic", features="dino", input_size=448,
                     model_type="vit_small", patch_size=8, slic_num_components=2000):
    return FeatureExtractor(
        device, superpixel_mode, features, input_size, model_type=model_type,
        patch_size=patch_size, slic_num_components=slic_num_components)
