import datetime
import os
from pathlib import Path

from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def create_experiment_folder(exp, env):
    # Set in name the correct model path
    if exp.get("timestamp", True):
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        model_path = os.path.join(env["base"], exp["general"]["name"])
        p = model_path.split("/")
        model_path = os.path.join("/", *p[:-1], str(timestamp) + "_" + p[-1])
    else:
        model_path = os.path.join(env["base"], exp["general"]["name"])
        shutil.rmtree(model_path, ignore_errors=True)

    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

    return model_path
