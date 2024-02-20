from wild_visual_navigation.utils import perguia_dataset, ROOT_DIR
import os
from pathlib import Path
import numpy as np

import yaml
import subprocess


def get_bag_info(rosbag_path: str) -> dict:
    # This queries rosbag info using subprocess and get the YAML output to parse the topics
    info_dict = yaml.safe_load(
        subprocess.Popen(["rosbag", "info", "--yaml", rosbag_path], stdout=subprocess.PIPE).communicate()[0]
    )
    return info_dict


ouput_dir = "/media/Data/Datasets/2022_Perugia/wvn_output"
imgs_per_s = 1.8
failed = []

for d in perguia_dataset:
    img_key = [
        str(s).split("/")[-1] for s in Path(ouput_dir, d["name"].replace("mission_data/", ""), "image").rglob("*.pt")
    ]
    supervision_mask_key = [
        str(s).split("/")[-1]
        for s in Path(ouput_dir, d["name"].replace("mission_data/", ""), "supervision_mask").rglob("*.pt")
    ]
    ls = img_key + supervision_mask_key
    ls.sort()
    all_keys = np.unique(np.array(ls))

    valid = np.zeros((all_keys.shape[0],), dtype=bool)

    for j, k in enumerate(all_keys):
        if k in img_key and k in supervision_mask_key:
            valid[j] = True

    keys = all_keys[valid].tolist()
    print(d["name"])
    print("   Keys: ", len(keys), ", Target Duration:", d["stop"] - d["start"])
    print("   Not valid:", (~valid).sum())
    delete_keys = all_keys[~valid].tolist()
    for dele in delete_keys:
        p = str(os.path.join(ouput_dir, d["name"].replace("mission_data/", ""), "image", dele))
        if os.path.exists(p):
            os.remove(p)
        p = str(
            os.path.join(
                ouput_dir,
                d["name"].replace("mission_data/", ""),
                "supervision_mask",
                dele,
            )
        )
        if os.path.exists(p):
            os.remove(p)

    res = get_bag_info(os.path.join(ROOT_DIR, d["name"] + "_wvn.bag"))

    if len(keys) != 0:
        time_to_early = (res["start"] + d["stop"]) - float(keys[-1][:-3].replace("_", "."))
        print(f"   Stopped to early by: {time_to_early}s")
    else:
        time_to_early = 99999999999
        print("   Not even a single key matched")

    if time_to_early > 10:
        failed.append(d["name"])
        print(d["env"])
    print(" ")
    print(" ")
