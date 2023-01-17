from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.learning.general import training_routine
from wild_visual_navigation.utils import override_params
from wild_visual_navigation.learning.utils import load_env

import os
import torch
from pathlib import Path
import yaml
import time

if __name__ == "__main__":
    runs = list(range(1))
    force = True
    cfgs = [
        str(s).replace(WVN_ROOT_DIR + "/cfg/exp/", "")
        for s in Path(os.path.join(WVN_ROOT_DIR, "cfg/exp/adaptation_time/test_models")).rglob("*.yaml")
    ]
    scenes = ["forest"]

    results = {}
    for scene in scenes:
        for cfg in cfgs:
            for run in runs:
                experiment_params = ExperimentParams()
                exp_override = load_yaml(os.path.join(WVN_ROOT_DIR, "cfg/exp", cfg))

                # run single config multiple time for confidence
                if len(runs) > 1:
                    exp_override["general"]["name"] = exp_override["general"]["name"] + f"_{run}"

                # set correct name with respect to scene
                exp_override["general"]["name"] = exp_override["general"]["name"].replace("forest", scene)
                exp_override["logger"] = {"name": "skip"}
                override_params(experiment_params, exp_override)

                # set correct scene (force overwrite)
                experiment_params.ablation_data_module.env = scene

                env = load_env()
                if (
                    os.path.exists(
                        os.path.join(env["base"], exp_override["general"]["name"], "detailed_test_results.pkl")
                    )
                    and not force
                ):
                    print("Skip Experiment")
                    continue
                else:
                    print(
                        "Run Experiment! Path does not exist:",
                        os.path.join(env["base"], exp_override["general"]["name"]),
                    )

                res, model_path, short_id, logger_project_name = training_routine(experiment_params, seed=run)
                print(short_id, exp_override["general"]["name"])
                print(res)

                results_name = cfg.replace(".yaml", "")
                if len(runs) > 1:
                    results_name = results_name + f"_{run}"

                results[results_name] = {
                    "model_path": model_path,
                    "logger_version": short_id,
                    "name": exp_override["general"]["name"],
                    "results": res,
                }

    with open(os.path.join(WVN_ROOT_DIR, "cfg/exp/ablation/all.yml"), "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
