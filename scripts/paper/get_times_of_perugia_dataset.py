from pathlib import Path
from wild_visual_navigation.utils import load_env
import os

perugia_root = os.path.join(load_env()["perugia_root"], "wvn_output/split")
ls = [str(s) for s in Path(perugia_root).rglob("*.txt") if str(s).find("desc") == -1]
import numpy as np

for l in ls:
    pa = []
    with open(l, "r") as f:
        while True:

            line = f.readline()
            if not line:
                break

            pa.append(line.strip())
    folder = np.unique(np.array([s.split("/")[:-2] for s in pa]))

    mission = np.unique(np.array([s.split("/")[-3] for s in pa]))
    arr = np.array([float(s.split("/")[-1][:-3].replace("_", ".")) for s in pa])
    delta_t = 0
    for m in mission:
        mask = np.array([s.split("/")[-3] for s in pa]) == m
        delta_t += arr[mask][-1] - arr[mask][0]
        print(delta_t)
