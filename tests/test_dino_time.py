# from wild_visual_navigation import WVN_ROOT_DIR
from pytictac import Timer
from wild_visual_navigation.feature_extractor import (
    DinoInterface,
    # TrtModel,
)

# from collections import namedtuple, OrderedDict
# from torchvision import transforms as T
# import os
import torch
from wild_visual_navigation.utils.testing import load_test_image, get_dino_transform

# import tensorrt as trt
# import numpy as np


def test_dino_interfacer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    di = DinoInterface(device)
    transform = get_dino_transform()

    img = load_test_image().to(device)

    #####################################################################################
    for i in range(5):
        im = img + torch.rand(img.shape, device=img.device) / 100
        di.inference(transform(im))

    #####################################################################################
    with Timer("BS1 Dino Inference: "):
        for i in range(5):
            im = img + torch.rand(img.shape, device=img.device) / 100
            with Timer("BS1 Dino Single: "):
                di.inference(transform(im))

    #####################################################################################
    # img = img.repeat(4, 1, 1, 1)
    # with Timer("BS4 Dino Inference: "):
    #     for i in range(2):
    #         im = img + torch.rand(img.shape, device=img.device) / 100
    #         with Timer("BS4 Dino Single: "):
    #             res = di.inference(di.transform(im))

    #####################################################################################
    # # Conversion from ONNX model (https://github.com/facebookresearch/dino)
    # from wild_visual_navigation.feature_extractor import DinoTrtInterface
    # exported_trt_file = "dino_exported.trt"
    # exported_trt_path = os.path.join(WVN_ROOT_DIR, "assets/dino", exported_trt_file)
    # di_trt = DinoTrtInterface(exported_trt_path, device)

    # with Timer("TensorRT Inference: "):
    #     im = img + torch.rand(img.shape, device=img.device) / 100
    #     di_trt.inference(di.transform(im).contiguous())

    #####################################################################################
    # Conversion using the torch_tensorrt library: https://github.com/pytorch/TensorRT
    # Note: 2022-08-18: doesn't work because it requires that all the DINO modules are
    # compatible with torch script. It's not the case because of the lack of typing, and
    # variable output size on each module
    #
    # -> RuntimeError: Expected a default value of type Tensor (inferred) on parameter
    # # "return_qkv".Because "return_qkv" was not annotated with an explicit type it
    # is assumed to be type 'Tensor'.:

    # import torch_tensorrt

    # script_model = torch.jit.script(di.model)
    # trt_model = torch_tensorrt.compile(script_model,
    #     inputs = [di.transform(im), # Provide example tensor for input shape or...
    #         torch_tensorrt.Input( # Specify input object with shape and dtype
    #             shape=[1, 3, 224, 224],
    #             # For static size shape=[1, 3, 224, 224]
    #             dtype=torch.half) # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool)
    #     ],
    #         enabled_precisions = {torch.half}, # Run with FP16
    # )

    # with Timer("TensorRT Inference: "):
    #     im = img + torch.rand(img.shape, device=img.device) / 100
    #     trt_model(di.transform(im))

    #####################################################################################
    # Conversion using the torch2rt library: https://github.com/NVIDIA-AI-IOT/torch2trt
    # Note: 2022-08-18: doesn't work because of problems with torch.Parameter
    # -> AttributeError: 'Parameter' object has no attribute '_trt'

    # from torch2trt import torch2trt

    # # convert to TensorRT feeding sample data as input
    # model_trt = torch2trt(di.model, [di.transform(im)])

    # with Timer("TensorRT Inference: "):
    #     im = img + torch.rand(img.shape, device=img.device) / 100
    #     y_trt = model_trt(di.transform(im))

    #####################################################################################


if __name__ == "__main__":
    test_dino_interfacer()
