import os
import torch
from pathlib import Path
import yaml
import time
import pickle
import copy
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.utils import load_yaml
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.learning.general import training_routine

if __name__ == "__main__":
    """Test how much time and data it takes for a model to convergee on a scene.
    Settings:
        - No log files are created by the model.
        - The validation dataloaders are abused as test dataloaders.
        - After every training epoch the test routine is called.
        - The results are accumulated and returned by the training routine.
        - Calling the testing is therefore not necessary.
        - This procedure is repeated over all scenes and percentage of data used from the training dataset.
    """

    number_training_runs = 10
    exp = ExperimentParams()
    exp.general.log_to_disk = False
    exp.trainer.max_steps = 10000
    exp.trainer.max_epochs = None
    exp.logger.name = "skip"
    exp.ablation_data_module.val_equals_test = True
    exp.trainer.profiler = None
    exp.trainer.enable_checkpointing = False
    exp.cb_checkpoint.active = False

    exp.verify_params()
    exp.visu.train = 0
    exp.visu.val = 0
    exp.visu.test = 0
    exp.general.model_path = os.path.join(WVN_ROOT_DIR, "scripts/ablations/time_adaptation")

    # If check_val_every_n_epoch in the current setting the test dataloader is used for validation.
    # All results during validation are stored and returned by the training routine.
    exp.trainer.check_val_every_n_epoch = 100000

    # Currently the model weights are stored every 10 steps.
    # This allows to reload the model and test it on the test dataloader.
    exp.general.store_model_every_n_steps = 100

    Path(exp.general.model_path).mkdir(parents=True, exist_ok=True)

    # Train model in various configurations and the validation results per epoch are returned in results_epoch.
    results_epoch = {}
    for scene in ["forest", "hilly", "grassland"]:
        exp.ablation_data_module.env = scene
        percentage_results = {}
        for percentage in range(10, 100, 10):
            exp.ablation_data_module.training_data_percentage = percentage
            run_results = {}
            for run in range(number_training_runs):
                exp.general.store_model_every_n_steps_key = f"ablation_time_adaptation_{scene}_{percentage}_{run}"
                res = training_routine(exp, seed=run)
                run_results[f"run_{run}"] = copy.deepcopy(res)
                torch.cuda.empty_cache()
            percentage_results[f"percentage_{percentage}"] = copy.deepcopy(run_results)
        results_epoch[scene] = copy.deepcopy(percentage_results)

    # Store epoch output to disk.
    p = os.path.join(WVN_ROOT_DIR, "scripts/ablations/time_adaptation/time_adaptation_epochs.pkl")
    try:
        os.remove(p)
    except OSError as error:
        pass
    with open(p, "wb") as handle:
        pickle.dump(results_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Test all stored models on the test dataloader and store the results.
    exp.general.skip_train = True
    exp.ablation_data_module.val_equals_test = False
    results_step = []
    for p in Path(exp.general.model_path).rglob("*.pt"):
        _, _, _, scene, percentage, run, steps = str(p).split("/")[-1].split("_")
        percentage, run, steps = int(percentage), int(run), int(steps.split(".")[0])
        exp.ablation_data_module.env = scene
        exp.model.load_ckpt = str(p)

        res = training_routine(exp, seed=run)
        results_step.append({"scene": scene, "percentage": percentage, "run": run, "steps": steps, "results": res})
        torch.cuda.empty_cache()

    # Store step output to disk.
    p = os.path.join(WVN_ROOT_DIR, "scripts/ablations/time_adaptation/time_adaptation_steps.pkl")
    try:
        os.remove(p)
    except OSError as error:
        pass

    with open(p, "wb") as handle:
        pickle.dump(results_step, handle, protocol=pickle.HIGHEST_PROTOCOL)
