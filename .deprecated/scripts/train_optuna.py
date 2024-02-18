import optuna
from argparse import ArgumentParser
import copy
import os
import neptune.integrations.optuna as optuna_utils
import torch

from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.general import training_routine
from wild_visual_navigation.utils import get_neptune_run
from omegaconf import read_write, OmegaConf


def objective(trial, params: ExperimentParams):
    exp = copy.deepcopy(params)

    with read_write(exp):
        # Parameter to sweep
        exp.optimizer.lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
        exp.loss.w_trav = trial.suggest_float("w_trav", 0.0, 1.0)
        exp.loss.w_temp = trial.suggest_float("w_temp", 0.0, 1.0)
        exp.loss.w_reco = trial.suggest_float("w_reco", 0.0, 1.0)
        exp.loss.anomaly_balanced = trial.suggest_categorical("anomaly_balanced", [True])

    res, _ = training_routine(exp)

    torch.cuda.empty_cache()
    return list(res.values())[0]["test_auroc_gt_image"]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--name", type=str, default="sweep_loss_function", help="Name of sweep")
    parser.add_argument("--n_trials", type=int, default=100, help="Number Trials")

    args = parser.parse_args()
    params = OmegaConf.structured(ExperimentParams)

    with read_write():
        params.general.name = os.path.join(args.name, params.general.name)

    run = get_neptune_run(
        neptune_project_name=params.logger.neptune_project_name,
        tags=["optuna", args.name],
    )

    neptune_callback = optuna_utils.NeptuneCallback(
        run,
        plots_update_freq=2,
        log_plot_slice=True,
        log_plot_contour=True,
    )

    params.logger.name = "skip"
    params.trainer.enable_checkpointing = False
    params.cb_checkpoint.active = False
    params.visu.train = 0
    params.visu.val = 0
    params.visu.test = 0
    params.trainer.check_val_every_n_epoch = 100000
    params.general.store_model_every_n_steps = None
    params.ablation_data_module.training_in_memory = True
    params.trainer.max_steps = 1000
    params.trainer.max_epochs = None
    params.general.log_to_disk = False
    params.trainer.progress_bar_refresh_rate = 0
    params.trainer.weights_summary = None
    params.trainer.enable_progress_bar = False

    binded_objective = lambda trial: objective(trial=trial, experiment=args.experiment)
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(binded_objective, n_trials=args.n_trials, callbacks=[neptune_callback])

    trial = study.best_trial
