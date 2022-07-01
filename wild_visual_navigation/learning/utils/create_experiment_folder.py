import datetime
import os
import shutil
from pathlib import Path

from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def create_experiment_folder():
    # Set in name the correct model path
    if exp.get("timestamp", True):
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        model_path = os.path.join(env["base"], exp["name"])
        p = model_path.split("/")
        model_path = os.path.join("/", *p[:-1], str(timestamp) + "_" + p[-1])
    else:
        model_path = os.path.join(env["base"], exp["name"])
        shutil.rmtree(model_path, ignore_errors=True)

    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Only copy config files for the main ddp-task
    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    print(f"Copy {exp_cfg_path} to {model_path}/{exp_cfg_fn}")
    shutil.copy(exp_cfg_path, f"{model_path}/{exp_cfg_fn}")
    return model_path
