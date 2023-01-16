from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml
import copy
import yaml
import os

replace_key = "slic_sift"
base = os.path.join(WVN_ROOT_DIR, "cfg/exp/abblation/feature/slic_sift.yaml")
device = "cpu"
fes = [
    "slic100_dino448_8",
    "slic100_dino448_16",
    "slic100_dino224_8",
    "slic100_dino224_16",
    "slic100_dino112_8",
    "slic100_dino112_16",
    "slic100_sift",
    "slic100_efficientnet_b4",
    "slic100_efficientnet_b7",
    "slic100_resnet50",
    "slic100_resnet50_dino",
    "slic100_resnet18",
]
##############################################################
file = load_yaml(base)
for k in fes:
    dump = copy.deepcopy(file)
    dump["general"]["name"] = file["general"]["name"].replace(replace_key, k)
    dump["abblation_data_module"]["feature_key"] = file["abblation_data_module"]["feature_key"].replace(replace_key, k)

    with open(base.replace(replace_key, k), "w") as f:
        yaml.dump(dump, f, default_flow_style=False, sort_keys=False)
