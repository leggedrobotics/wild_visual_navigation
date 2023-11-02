from wild_visual_navigation.utils import perguia_dataset, ROOT_DIR
from pathlib import Path
import numpy as np
import os

# Definition
# training 80% first train
# validation 20% last validation
# test subsample of all other scenes
# {env}_{mode}.txt

percentage = 0.8
every_n_test = 50

scenes = {}
for d in perguia_dataset:
    scenes[d["env"]] = []

for d in perguia_dataset:
    p = [
        str(s).replace(ROOT_DIR + "/", "")
        for s in Path(ROOT_DIR, "wvn_output", d["name"].replace("mission_data/", ""), "image").rglob("*.pt")
    ]
    if d["mode"] == "train":
        p.sort()
        m = d["mode"]
        env = d["env"]
        s = len(p)
        print(s)

        with open(os.path.join(ROOT_DIR, "wvn_output/split", f"{env}_train.txt"), "w") as output:
            for k in p[1 : int(s * percentage)]:
                output.write(k + "\n")

        with open(os.path.join(ROOT_DIR, "wvn_output/split", f"{env}_val.txt"), "w") as output:
            for k in p[int(s * percentage) + 1 :]:
                output.write(k + "\n")

    if d["mode"] == "test":
        print(len(p))
        scenes[d["env"]] += p

for k, v in scenes.items():
    with open(os.path.join(ROOT_DIR, "wvn_output/split", f"{k}_test.txt"), "w") as output:
        for _k in v[::every_n_test]:
            # DONT WANT TO HAVE FRAME 0 given optical flow
            if _k == 0:
                continue

            output.write(_k + "\n")
