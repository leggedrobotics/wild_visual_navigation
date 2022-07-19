import optuna
from simple_parsing import ArgumentParser
import copy
import dataclasses
import os
import neptune.new.integrations.optuna as optuna_utils
import torch

from wild_visual_navigation.learning.utils import ExperimentParams
from wild_visual_navigation.learning.general import training_routine
from wild_visual_navigation.learning.utils import get_neptune_run


def objective(trial, experiment: ExperimentParams):
    exp = copy.deepcopy(experiment)

    exp.optimizer.lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    exp.trainer.max_epochs = trial.suggest_int("max_epochs", 1, 10)

    res = training_routine(exp)

    torch.cuda.empty_cache()

    return res[0]["test_acc_epoch"]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_arguments(ExperimentParams, dest="experiment")
    parser.add_argument("--name", type=str, default="lr_exp_0", help="Name of sweep")
    parser.add_argument("--log_lightning_run", type=bool, default=False, help="Log Lightning Run")
    parser.add_argument("--n_trials", type=int, default=20, help="Number Trials")

    args = parser.parse_args()
    args.experiment.general.name = os.path.join(args.name, args.experiment.general.name)

    run = get_neptune_run(neptune_project_name=args.experiment.logger.neptune_project_name, tags=["optuna", args.name])

    neptune_callback = optuna_utils.NeptuneCallback(
        run,
        plots_update_freq=2,
        log_plot_slice=True,
        log_plot_contour=True,
    )
    if not args.log_lightning_run:
        args.experiment.logger.name = "skip"

    binded_objective = lambda trial: objective(trial=trial, experiment=args.experiment)
    study = optuna.create_study(direction="maximize")
    study.optimize(binded_objective, n_trials=args.n_trials, callbacks=[neptune_callback])

    trial = study.best_trial
