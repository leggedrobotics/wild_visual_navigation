from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml
import copy
import yaml
import os

base = os.path.join(WVN_ROOT_DIR, "cfg/exp/adaptation_time/template/forest99.yaml")

##############################################################
file = load_yaml(base)

for i in range(10, 101, 10):

    dump = copy.deepcopy(file)

    dump["general"]["name"] = file["general"]["name"].replace("99", str(i))
    dump["abblation_data_module"]["training_data_percentage"] = i

    with open(base.replace("99", str(i)).replace("template", "percentage_sweep"), "w") as f:
        yaml.dump(dump, f, default_flow_style=False, sort_keys=False)
