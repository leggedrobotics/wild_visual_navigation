from wild_visual_navigation.visu import LearningVisualizer
from wild_visual_navigation import WVN_ROOT_DIR
import os
import torch

IMAGE_SIZE = 500
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# supervision_mask = torch.load("/media/Data/Datasets/2022_Perugia/wvn_output/day3/2022-05-12T11:56:13_mission_0_day_3/supervision_mask/1652341966_7438905.pt")

p = [
    str(s)
    for s in Path("/media/Data/Datasets/2022_Perugia/wvn_output/day3/2022-05-12T11:56:13_mission_0_day_3/image").rglob(
        "*.pt"
    )
]
p.sort()

visu = LearningVisualizer(os.path.join(WVN_ROOT_DIR, "results/visu_dataset"), store=False)
# seg = (torch.nan_to_num(supervision_mask.nanmean(axis=0) ))
# res = visu.plot_detectron_cont(image, seg, max_seg = 1, alpha=0.5, tag = "one")


plt.ion()
fig1, ax1 = plt.subplots()
image = torch.load(p[0])
res = visu.plot_image(image)
axim1 = ax1.imshow(res)

for _p in p:
    image = torch.load(_p)
    res = visu.plot_image(image)

    print(".", end="")
    # matrix = np.random.randint(0, 100, size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    axim1.set_data(res)
    fig1.canvas.flush_events()
