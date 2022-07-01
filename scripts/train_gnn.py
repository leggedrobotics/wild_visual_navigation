import os
import sys

import argparse
import shutil

# Frameworks
import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin, DDPSpawnPlugin

# Costume Modules
from wild_visual_navigation import WVN_ROOT_DIR

from wild_visual_navigation.learning.utils import get_logger
from wild_visual_navigation.learning.lightning import LightningTrav
from wild_visual_navigation.learning.utils import load_yaml, load_env, create_experiment_folder
from wild_visual_navigation.learning.dataset import get_pl_graph_trav_module

__all__ = ["train"]


def train(exp_cfg_path):
    seed_everything(42)
    exp = load_yaml(exp_cfg_path)
    env = load_env()

    model_path = create_experiment_folder(exp, env, exp_cfg_path)
    exp["general"]["name"] = os.path.relpath(model_path, env["base"])

    logger = get_logger(exp, env)

    # SET GPUS
    if (exp["trainer"]).get("gpus", -1) == -1:
        nr = torch.cuda.device_count()
        print(f"Set GPU Count for Trainer to {nr}!")
        for i in range(nr):
            print(f"Device {i}: " + str(torch.cuda.get_device_name(i)))
        exp["trainer"]["gpus"] = -1

    # MODEL
    model = LightningTrav(exp=exp, env=env)

    # profiler
    if exp["trainer"].get("profiler", False) == "advanced":
        exp["trainer"]["profiler"] = AdvancedProfiler(dirpath=model_path, filename="profile.txt")
    else:
        exp["trainer"]["profiler"] = False

    # COLLECT CALLBACKS
    cb_ls = [LearningRateMonitor(**exp["lr_monitor"])]

    if exp["cb_early_stopping"]["active"]:
        early_stop_callback = EarlyStopping(**exp["cb_early_stopping"]["cfg"])
        cb_ls.appned(early_stop_callback)

    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None
    exp["trainer"]["gpus"] = gpus
    # add distributed plugin
    if torch.cuda.is_available():
        if len(gpus) > 1:
            if exp["trainer"]["accelerator"] == "ddp" or exp["trainer"]["accelerator"] is None:
                ddp_plugin = DDPPlugin(find_unused_parameters=exp["trainer"].get("find_unused_parameters", False))
            elif exp["trainer"]["accelerator"] == "ddp_spawn":
                ddp_plugin = DDPSpawnPlugin(find_unused_parameters=exp["trainer"].get("find_unused_parameters", False))
            elif exp["trainer"]["accelerator"] == "ddp2":
                ddp_plugin = DDP2Plugin(find_unused_parameters=exp["trainer"].get("find_unused_parameters", False))
            exp["trainer"]["plugins"] = [ddp_plugin]

    ddp_plugin = DDPSpawnPlugin(find_unused_parameters=exp["trainer"].get("find_unused_parameters", False))
    exp["trainer"]["plugins"] = ddp_plugin
    
    datamodule = get_pl_graph_trav_module(**exp["data_module"])
    trainer = Trainer(
        **exp["trainer"], default_root_dir=model_path, callbacks=cb_ls, logger=logger
    )
    res = trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="exp.yaml",
        help="Experiment yaml file.",
    )

    args = parser.parse_args()

    exp_cfg_path = args.exp
    print(WVN_ROOT_DIR, args.exp)
    if not os.path.isabs(exp_cfg_path):
        exp_cfg_path = os.path.join(WVN_ROOT_DIR, "cfg/exp", args.exp)
    print(exp_cfg_path)

    train(exp_cfg_path)
    torch.cuda.empty_cache()
