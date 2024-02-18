from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import (
    DinoInterface,
    # DinoTrtInterface,
    # TrtModel,
)

# from collections import namedtuple, OrderedDict
# from torchvision import transforms as T
# import cv2
import os
from os.path import join
import numpy as np
import pandas as pd
import torch

# import tensorrt as trt
from tqdm import tqdm

import contextlib


@contextlib.contextmanager
def capture_output():
    import sys
    from io import StringIO

    oldout, olderr = sys.stdout, sys.stderr
    try:
        out = [StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


def run_dino_interfacer():
    """Performance inference using stego and stores result as an image."""

    from pytictac import Timer
    import cv2

    # Create test directory
    test_path = join(WVN_ROOT_DIR, "results", "test_dino_interfacer")
    os.makedirs(test_path, exist_ok=True)

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read images
    test_image_names = []
    test_images = []

    # Options
    SIZES = [112, 224, 448]
    INTERPS = ["nearest", "bilinear"]
    MODELS = ["vit_small", "vit_base"]
    PATCHES = [8, 16]
    TRIALS = 100
    N_IMAGES = 20

    for n in range(N_IMAGES):
        test_image = f"perugia_0{n:02}"
        p = join(WVN_ROOT_DIR, f"assets/images/{test_image}.png")
        np_img = cv2.imread(p)
        img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
        img = img.permute(2, 0, 1)
        img = (img.type(torch.float32) / 255)[None]

        test_image_names.append(test_image)
        test_images.append(img)

    # Make timer
    timer = Timer("DINO timer")
    full_timer = Timer("Full timer")

    # Samples
    samples = []

    full_timer.tic()
    # Inference with DINO
    for size in SIZES:
        for interp in INTERPS:
            for model in MODELS:
                for patch in PATCHES:
                    di = None
                    with capture_output():
                        # Create DINO
                        di = DinoInterface(
                            device=device,
                            input_size=size,
                            input_interp=interp,
                            model_type=model,
                            patch_size=patch,
                        )

                    for img, img_name in zip(test_images, test_image_names):
                        print(size, interp, model, patch, img_name)

                        for n in tqdm(range(TRIALS)):
                            with capture_output():
                                try:
                                    timer.tic()
                                    di.inference(di.transform(img), interpolate=False)
                                    t = timer.toc()
                                except Exception:
                                    t = np.nan

                                samples.append([size, interp, model, patch, img_name, n, t])

    t = full_timer.toc()
    print(f"Iterations took {t/1000.0} s/ {t/t/3600000.0} hs")

    # Make pandas dataframe
    df = pd.DataFrame(
        samples,
        columns=[
            "image_size",
            "interpolation",
            "model",
            "patch_size",
            "img_name",
            "sample",
            "time",
        ],
    )
    df.to_pickle(join(test_path, f"dino_time_settings_{TRIALS}iter.pkl"))
    df.to_csv(join(test_path, f"dino_time_settings_{TRIALS}iter.csv"))


if __name__ == "__main__":
    run_dino_interfacer()
