from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml, load_env
import copy
import yaml
import os

base = os.path.join(WVN_ROOT_DIR, "cfg/exp/adaptation_time/template/forest99_test100.yaml")

path = os.path.join(WVN_ROOT_DIR, "cfg/exp/adaptation_time/test_models/")
##############################################################
file = load_yaml(base)
env = load_env()

os.path.join(WVN_ROOT_DIR, base)
for percentage in range(10, 101, 10):
    dump = copy.deepcopy(file)
    for step in range(0, 5000, 100):
        dump["general"]["name"] = file["general"]["name"].replace("99", str(percentage))[:-3] + str(step)
        p = dump["general"]["name"].rfind("_")
        reference = dump["general"]["name"][:p]
        dump["model"]["load_ckpt"] = os.path.join(WVN_ROOT_DIR, env["base"], reference, f"{step}.pt")
        percentage_name = reference.split("/")[-1]
        os.makedirs(os.path.join(path, percentage_name), exist_ok=True)
        with open(os.path.join(path, percentage_name, f"{step}.yaml"), "w") as f:
            yaml.dump(dump, f, default_flow_style=False, sort_keys=False)
