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
from wild_visual_navigation.utils import override_params

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

    number_training_runs = 1

    exp = ExperimentParams()
    exp.general.log_to_disk = False
    exp.trainer.max_steps = 1000
    exp.trainer.max_epochs = None
    exp.logger.name = "skip"
    exp.ablation_data_module.val_equals_test = False
    exp.ablation_data_module.test_all_datasets = True
    exp.trainer.profiler = None
    exp.trainer.enable_checkpointing = False
    exp.cb_checkpoint.active = False

    exp.verify_params()
    exp.visu.train = 0
    exp.visu.val = 0
    exp.visu.test = 0
    exp.trainer.check_val_every_n_epoch = 100000
    exp.general.store_model_every_n_steps = None

    exp.general.model_path = os.path.join(WVN_ROOT_DIR, "scripts/ablations/network_ablation")

    Path(exp.general.model_path).mkdir(parents=True, exist_ok=True)

    directory = Path(os.path.join(WVN_ROOT_DIR, "cfg/exp/ablation/network"))
    cfg_paths = [str(p) for p in directory.rglob("*.yaml") if str(p).find("template") == -1]
    # Train model and get test results for every epoch.
    results_epoch = {}
    j = 0
    for scene in ["forest", "hilly", "grassland"]:
        model_results = {}
        exp.ablation_data_module.env = scene
        for p in cfg_paths:
            run_results = {}
            for run in range(number_training_runs):
                p = str(p)
                exp_override = load_yaml(p)
                override_params(exp, exp_override)
                res = training_routine(exp, seed=run)
                run_results[str(run)] = copy.deepcopy(res)
                j += 1
            model_results[p] = copy.deepcopy(run_results)
        results_epoch[scene] = copy.deepcopy(model_results)

    # Store epoch output to disk.
    p = os.path.join(WVN_ROOT_DIR, "scripts/ablations/network_ablation/network_ablation_test_results.pkl")
    
    try:
        os.remove(p)
    except OSError as error:
        pass
    
    with open(p, "wb") as handle:
        pickle.dump(results_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
