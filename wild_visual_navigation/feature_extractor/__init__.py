#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from .dino_interface import DinoInterface, run_dino_interfacer
from .torchvision_interface import TorchVisionInterface

# from .dino_trt_interface import DinoTrtInterface, TrtModel, run_dino_trt_interfacer
from .stego_interface import StegoInterface, run_stego_interfacer
from .segment_extractor import SegmentExtractor
from .feature_extractor import FeatureExtractor
