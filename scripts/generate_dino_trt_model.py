# #############################################################################
# This scripts converts DINO into a TensorRT model
# It first converts the original model to ONNX
# Then it uses trtexec to convert the ONNX one to TensorRT
#
# This requries to have all the environment variables setup correctly:
# # CUDA
# export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# # TensorRT (e.g. installed in the home folder)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/username/TensorRT-8.4.3.1/lib
# 
# If used in a conda environment, the CuBLAS versions must match between the
# system and the environment, and TensorRT must be installed within the environment
# #############################################################################

from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import DinoInterface, DinoTrtInterface
from torchvision import transforms as T
import cv2
import os
import subprocess
import torch
import tensorrt as trt
import numpy as np
import os

if __name__ == "__main__":
    # Create original DINO model
    print("\nLoading DINO ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    di = DinoInterface(device)

    # Read test image
    np_img = cv2.imread(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]

    # Prepare transform to resize the image
    transform = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.NEAREST),
            T.CenterCrop(224),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Export model as ONNX
    print("\nGenerating ONNX file ...")
    exported_onnx_file = "dino_exported.onnx"
    exported_onnx_path = os.path.join(WVN_ROOT_DIR, "assets/dino", exported_onnx_file)
    example_input = transform(img)
    torch.onnx.export(
        di.model,
        example_input,
        exported_onnx_path,
        verbose=True,
        input_names=["input"],
        output_names=["features", "code"],
    )

    # Export model as tensorrt
    print("\nGenerating TensorRT file ...")
    exported_trt_file = "dino_exported.trt"
    exported_trt_path = os.path.join(WVN_ROOT_DIR, "assets/dino", exported_trt_file)

    cmd = f"trtexec --onnx={exported_onnx_path} --saveEngine={exported_trt_path} --useCudaGraph"
    subprocess.run(cmd, shell=True, check=True)

    # Test models
    print("\nTesting models ...")
    # Run one inference of DINO
    y_original = di.inference(di.transform(img))

    # Run one inference of DINO RT
    di_trt = DinoTrtInterface(exported_trt_path, device)
    y_trt = di_trt.inference(di.transform(img))

    # Compare error
    error = torch.nn.functional.mse_loss(y_original, y_trt, reduction="mean")
    print(f"Average error between original and TensorRT model: {error:.4f}")

    print("Done")
