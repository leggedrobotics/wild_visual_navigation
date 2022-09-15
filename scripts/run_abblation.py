from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.learning.general import training_routine
from wild_visual_navigation.utils import override_params

import os
from pathlib import Path
import yaml

if __name__ == "__main__":
    cfgs = [
        str(s).replace(WVN_ROOT_DIR + "/cfg/exp/", "")
        for s in Path(os.path.join(WVN_ROOT_DIR, "cfg/exp/abblation")).rglob("*.yaml")
    ]
    results = {}

    for cfg in cfgs:
        experiment_params = ExperimentParams()
        exp_override = load_yaml(os.path.join(WVN_ROOT_DIR, "cfg/exp", cfg))
        override_params(experiment_params, exp_override)
        res, model_path, short_id, logger_project_name = training_routine(experiment_params)
        print(short_id, exp_override["general"]["name"])
        print(res)

        results[cfg.replace(".yaml", "")] = {
            "model_path": model_path,
            "logger_version": short_id,
            "name": exp_override["general"]["name"],
            "results": res,
        }

    with open(os.path.join(WVN_ROOT_DIR, "cfg/exp/abblation/all.yml"), "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
