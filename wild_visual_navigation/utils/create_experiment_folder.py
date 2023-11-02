import datetime
import os
import shutil
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from wild_visual_navigation.cfg import ExperimentParams
from typing import Union
from omegaconf import read_write


@rank_zero_only
def create_experiment_folder(exp: ExperimentParams) -> str:
    """Creates an experiment folder if rank=0 with optional unique timestamp.
    Inplace sets the correct model_path in the experiment cfg.

    Args:
        exp (ExperimentParams): Experiment cfg

    Returns:
        str: model_path of the run
    """
    name = exp.general.name
    timestamp = exp.general.timestamp
    folder = exp.general.folder

    # Set in name the correct model path
    if timestamp:
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        model_path = os.path.join(folder, name)
        p = model_path.split("/")
        model_path = os.path.join("/", *p[:-1], str(timestamp).replace(":", "-") + "_" + p[-1])
    else:
        model_path = os.path.join(folder, name)
        shutil.rmtree(model_path, ignore_errors=True)

    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

    return model_path
