import os
import yaml
import dataclasses
import pickle

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
from wild_visual_navigation.learning.utils import get_logger
from wild_visual_navigation.learning.lightning import LightningTrav
from wild_visual_navigation.learning.utils import load_yaml, load_env, create_experiment_folder
from wild_visual_navigation.learning.dataset import get_pl_graph_trav_module, get_abblation_module
from wild_visual_navigation.cfg import ExperimentParams

__all__ = ["training_routine"]


def training_routine(experiment: ExperimentParams) -> torch.Tensor:
    seed_everything(42)
    exp = dataclasses.asdict(experiment)
    env = load_env()

    model_path = create_experiment_folder(exp, env)

    exp["general"]["name"] = os.path.relpath(model_path, env["base"])
    exp["general"]["model_path"] = model_path

    with open(os.path.join(model_path, "experiment_params.yaml"), "w") as f:
        yaml.dump(exp, f, default_flow_style=False)

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
    cb_ls = []
    if logger is not None:
        cb_ls.append(LearningRateMonitor(**exp["lr_monitor"]))

    if exp["cb_early_stopping"]["active"]:
        early_stop_callback = EarlyStopping(**exp["cb_early_stopping"]["cfg"])
        cb_ls.appned(early_stop_callback)

    if exp["ch_checkpoint"]["active"]:
        checkpoint_callback = ModelCheckpoint(dirpath=model_path, save_top_k=1, monitor="epoch", mode='max', save_last=True)
        cb_ls.appned(checkpoint_callback)

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

    # datamodule = get_pl_graph_trav_module(**exp["data_module"])
    datamodule = get_abblation_module(**exp["abblation_data_module"], perugia_root=env["perugia_root"])

    trainer = Trainer(**exp["trainer"], default_root_dir=model_path, callbacks=cb_ls, logger=logger)
    trainer.fit(model=model, datamodule=datamodule)

    res = trainer.test(model=model, datamodule=datamodule)[0]

    with open(os.path.join(model_path, "detailed_test_results.pkl"), "rb") as handle:
        out = pickle.load(handle)
    res["detailed_test_results"] = out
    
    model.logger.experiment["model_checkpoint"].upload_files(os.path.join(model_path, "last.ckpt"))
    model.logger.experiment["model_checkpoint"].upload_files(os.path.join(model_path, "detailed_test_results.pkl"))
    
    
    try:
        short_id = logger.experiment._short_id
        project_name = logger._project_name
    except Exception as e:
        project_name = "not_defined"
        short_id = 0
    return res, model_path, short_id, project_name
