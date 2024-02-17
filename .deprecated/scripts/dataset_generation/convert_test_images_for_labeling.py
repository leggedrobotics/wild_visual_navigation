import numpy as np
from wild_visual_navigation.utils import ROOT_DIR
import os


env = "grassland"

os.mkdir(os.path.join(ROOT_DIR, "wvn_output/labeling", env))


ls = []
with open(f"/media/Data/Datasets/2022_Perugia/wvn_output/split/{env}_test.txt", "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        ls.append(line.strip())

import torch
from PIL import Image

for l in ls:
    p = os.path.join(ROOT_DIR, l)
    res = torch.load(p)
    img = Image.fromarray(np.uint8(res.permute(1, 2, 0).cpu().numpy() * 255))

    img.save(
        os.path.join(
            ROOT_DIR,
            "wvn_output/labeling",
            env,
            l.split("/")[-1].replace(".pt", ".png"),
        )
    )
