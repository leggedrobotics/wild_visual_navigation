import os
from pathlib import Path
import torch
import numpy as np

if __name__ == "__main__":
    visu = True

    mission_folders = ["/media/Data/Datasets/2022_Perugia/wvn_output/day3/2022-05-12T11:56:13_mission_0_day_3"]

    for mission in mission_folders:
        assert os.path.isdir(os.path.join(mission, "image")), f"{mission} is not a valid mission folder misses image"
        assert os.path.isdir(
            os.path.join(mission, "supervision_mask")
        ), f"{mission} is not a valid mission folder misses supervision_mask"

        images = [str(s) for s in Path(mission, "image").rglob("*.pt")]
        supervision_masks = [str(s) for s in Path(mission, "supervision_mask").rglob("*.pt")]
        l1 = [s.split("/")[-1][:-3] for s in images]
        l2 = [s.split("/")[-1][:-3] for s in supervision_masks]
        all_keys = l1 + l2
        all_keys = np.unique(np.array(all_keys))
        valid = np.zeros((all_keys.shape[0],), dtype=bool)

        for j, k in enumerate(all_keys):
            if k in l1 and k in l2:
                valid[j] = True

        images_with_mask = all_keys[valid]

        timestamps = [s.split("/")[-1][:-3] for s in images]  # remove .pt

        t_0 = timestamps[0]
        delta_t = []
        for t in timestamps[1:]:
            delta_t.append(float(t.replace("_", ".")) - float(t_0.replace("_", ".")))
            t_0 = t

        print(delta_t)
