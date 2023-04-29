import optuna
from simple_parsing import ArgumentParser
import copy
import os
import neptune.new.integrations.optuna as optuna_utils
import torch

from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.learning.general import training_routine
from wild_visual_navigation.learning.utils import get_neptune_run
from wild_visual_navigation.utils import override_params
from wild_visual_navigation.learning.utils import load_yaml
from wild_visual_navigation import WVN_ROOT_DIR


def objective(trial, experiment: ExperimentParams):
    exp = copy.deepcopy(experiment)

    # exp.optimizer.lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    exp.loss.w_trav = trial.suggest_float("w_trav", 0.0, 1.0)
    exp.loss.w_temp = trial.suggest_float("w_temp", 0.0, 1.0)
    exp.loss.w_reco = trial.suggest_float("w_reco", 0.0, 1.0)
    exp.loss.anomaly_balanced = trial.suggest_categorical("anomaly_balanced", [True])

    # if not trial.suggest_categorical("use_temporal_consistency", [True, False]):
    # exp.loss.w_temp = 0.0
    # exp.trainer.max_epochs = trial.suggest_int("max_epochs", 1, 10)

    res, _ = training_routine(exp)

    torch.cuda.empty_cache()
    return list(res.values())[0]["test_auroc_gt_image"]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_arguments(ExperimentParams, dest="experiment")
    parser.add_argument("--exp", type=str, default="nan", help="Overwrite params")

    parser.add_argument("--name", type=str, default="sweep_loss_function", help="Name of sweep")
    parser.add_argument("--n_trials", type=int, default=100, help="Number Trials")

    args = parser.parse_args()

    p = os.path.join(WVN_ROOT_DIR, "cfg/exp", args.exp)
    if args.exp != "nan" and os.path.isfile(p):
        exp_override = load_yaml(p)
        args.experiment = override_params(args.experiment, exp_override)

    args.experiment.general.name = os.path.join(args.name, args.experiment.general.name)

    run = get_neptune_run(neptune_project_name=args.experiment.logger.neptune_project_name, tags=["optuna", args.name])

    neptune_callback = optuna_utils.NeptuneCallback(
        run,
        plots_update_freq=2,
        log_plot_slice=True,
        log_plot_contour=True,
    )

    args.experiment.logger.name = "skip"
    args.experiment.trainer.enable_checkpointing = False
    args.experiment.cb_checkpoint.active = False
    args.experiment.visu.train = 0
    args.experiment.visu.val = 0
    args.experiment.visu.test = 0
    args.experiment.trainer.check_val_every_n_epoch = 100000
    args.experiment.general.store_model_every_n_steps = None
    args.experiment.ablation_data_module.training_in_memory = True
    args.experiment.trainer.max_steps = 1000
    args.experiment.trainer.max_epochs = None
    args.experiment.general.log_to_disk = False
    args.experiment.trainer.progress_bar_refresh_rate = 0
    args.experiment.trainer.weights_summary = None
    args.experiment.trainer.enable_progress_bar = False

    binded_objective = lambda trial: objective(trial=trial, experiment=args.experiment)
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(binded_objective, n_trials=args.n_trials, callbacks=[neptune_callback])

    trial = study.best_trial
