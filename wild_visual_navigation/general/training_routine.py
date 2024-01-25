import os
import warnings
from os.path import join
from omegaconf import read_write

warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Frameworks
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler

# Costume Modules
from wild_visual_navigation.utils import get_logger
from wild_visual_navigation.lightning import LightningTrav
from wild_visual_navigation.utils import create_experiment_folder, save_omega_cfg
from wild_visual_navigation.dataset import get_ablation_module
from wild_visual_navigation.cfg import ExperimentParams

__all__ = ["training_routine"]


def training_routine(exp: ExperimentParams, seed=42) -> torch.Tensor:
    exp
    seed_everything(seed)

    if exp.general.log_to_disk:
        model_path = create_experiment_folder(exp)
    else:
        model_path = exp.general.model_path

    with read_write(exp):
        # Update model paths
        exp.general.model_path = model_path
        exp.general.name = os.path.relpath(model_path, exp.env.results)
        exp.trainer.default_root_dir = model_path
        exp.visu.learning_visu.p_visu = join(model_path, "visu")

        logger = get_logger(exp)

        # Set gpus
        exp.trainer.gpus = 1 if torch.cuda.is_available() else None

        # Profiler
        if exp.trainer.get("profiler", False) == "advanced":
            exp.trainer.profiler = AdvancedProfiler(dirpath=model_path, filename="profile.txt")

    # Callbacks
    cb_ls = []
    if logger is not None:
        cb_ls.append(LearningRateMonitor(**exp.lr_monitor))

    if exp.cb_early_stopping.active:
        early_stop_callback = EarlyStopping(**exp.cb_early_stopping.cfg)
        cb_ls.appned(early_stop_callback)

    if exp.cb_checkpoint.active:
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_path,
            save_top_k=1,
            monitor="epoch",
            mode="max",
            save_last=True,
        )
        cb_ls.append(checkpoint_callback)

    train_dl, val_dl, test_dl = get_ablation_module(
        **exp.ablation_data_module,
        perugia_root=exp.general.perugia_root,
        get_train_val_dataset=not exp.general.skip_train,
        get_test_dataset=not exp.ablation_data_module.val_equals_test,
    )

    # Set correct input feature dimension
    if train_dl is not None:
        sample = train_dl.dataset[0]
    else:
        sample = test_dl[0].dataset[0]
    input_feature_dimension = sample.x.shape[1]

    with read_write(exp):
        exp.model.simple_mlp_cfg.input_size = input_feature_dimension
        exp.model.simple_gcn_cfg.input_size = input_feature_dimension
        exp.model.double_mlp_cfg.input_size = input_feature_dimension
        exp.model.linear_rnvp_cfg.input_size = input_feature_dimension

    if exp.general.log_to_disk:
        save_omega_cfg(exp, os.path.join(model_path, "experiment_params.yaml"))

    # Model
    model = LightningTrav(exp=exp)
    if type(exp.model.load_ckpt) == str:
        ckpt = torch.load(exp.model.load_ckpt)
        try:
            res = model.load_state_dict(ckpt.state_dict, strict=False)
        except Exception:
            res = model.load_state_dict(ckpt, strict=False)
        print("Loaded model checkpoint:", res)
    trainer = Trainer(**exp.trainer, callbacks=cb_ls, logger=logger)

    if not exp.general.skip_train:
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if exp.ablation_data_module.val_equals_test:
        return model.accumulated_val_results, model

    # TODO Verify that this makes sense here
    test_envs = []
    for j, dl in enumerate(test_dl):
        if exp.loss.w_trav == 0:
            model._traversability_loss._anomaly_threshold = None

        model.nr_test_run = j
        res = trainer.test(model=model, dataloaders=dl)[0]
        test_envs.append(dl.dataset.env)

    return {k: v for k, v in zip(test_envs, model.accumulated_test_results)}, model
