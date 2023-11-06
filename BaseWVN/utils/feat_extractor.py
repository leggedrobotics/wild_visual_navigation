import torch
from dinov2_interface import Dinov2Interface
import numpy as np
from torchvision import transforms as T

class FeatureExtractor:
    def __init__(
        self, device: str, segmentation_type: str = "pixel", feature_type: str = "dinov2", input_size: int = 448, **kwargs
    ):
        """Feature extraction from image

        Args:
            device (str): Compute device

        """
        self._device = device
        self._segmentation_type = segmentation_type
        self._feature_type = feature_type
        self._input_size = input_size
        # input size must be a multiple of 14
        assert self._input_size % 14 == 0, "Input size must be a multiple of 14"
        # Extract original_width and original_height from kwargs if present
        self.original_width = kwargs.get('original_width', 1920)  # Default to 1920 if not provided
        self.original_height = kwargs.get('original_height', 1280)  # Default to 1080 if not provided
        self.interp=kwargs.get('interp', 'bilinear')
        # feature extractor
        if self._feature_type == "dinov2":
            self.extractor=Dinov2Interface(device, input_size=self._input_size, original_width=self.original_width, original_height=self.original_height, input_interp=self.interp)
        else:
            raise ValueError(f"Extractor[{self._feature_type}] not supported!")
        
        # segmentation
        if self._segmentation_type == "pixel":
            pass
        elif self._segmentation_type == "slic":
            from fast_slic import Slic
            self.slic = Slic(
                num_components=kwargs.get("slic_num_components", 200), compactness=kwargs.get("slic_compactness", 10)
            )
        else:
            raise ValueError(f"Segmentation[{self._segmentation_type}] not supported!")
        


    
    @property
    def feature_type(self):
        return self._feature_type

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def segmentation_type(self):
        return self._segmentation_type

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._device = device
        self.extractor.change_device(device)