from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml
import copy
import yaml
import os

replace_key = "slic_sift"
base = os.path.join(WVN_ROOT_DIR, "cfg/exp/abblation/feature/slic_sift.yaml")
device = "cpu"
####################################################################
# COPY PASTA CODE FROM FEATURE EXTRACTOR
####################################################################
fes = {}
fes["slic100_dino448_8"] = 0
fes["slic100_dino448_16"] = 0
fes["slic100_dino224_8"] = 0
fes["slic100_dino224_16"] = 0
fes["slic100_dino112_8"] = 0
fes["slic100_dino112_16"] = 0
fes["slic100_sift"] = 0
fes["slic100_efficientnet_b4"] = 0
fes["slic100_efficientnet_b7"] = 0
fes["slic100_resnet50"] = 0
fes["slic100_resnet18"] = 0
##############################################################
file = load_yaml(base)
for k in fes:
    dump = copy.deepcopy(file)
    dump["general"]["name"] = file["general"]["name"].replace(replace_key, k)
    dump["abblation_data_module"]["feature_key"] = file["abblation_data_module"]["feature_key"].replace(replace_key, k)

    with open(base.replace(replace_key, k), "w") as f:
        yaml.dump(dump, f, default_flow_style=False, sort_keys=False)
