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

    number_training_runs = 1
    exp = ExperimentParams()
    exp.general.log_to_disk = False
    exp.trainer.max_steps = 5000
    exp.trainer.max_epochs = None
    exp.trainer.check_val_every_n_epoch = 1
    exp.logger.name = "skip"
    exp.abblation_data_module.val_equals_test = True
    exp.abblation_data_module.active = True
    exp.trainer.profiler = None
    exp.cb_checkpoint.active = False
    exp.general.store_model_every_n_steps = None
    exp.verify_params()
    exp.visu.train = 0
    exp.visu.val = 0
    exp.visu.test = 0

    results = {}
    for scene in ["forest", "hilly", "grassland"]:
        exp.abblation_data_module.env = scene
        percentage_results = {}
        for percentage in range(10, 100, 40):
            exp.abblation_data_module.training_data_percentage = percentage
            run_results = {}
            for run in range(number_training_runs):
                res = training_routine(exp, seed=run)
                run_results[f"run_{run}"] = copy.deepcopy(res)
            percentage_results[f"percentage_{percentage}"] = copy.deepcopy(run_results)
        results[scene] = copy.deepcopy(percentage_results)

        p = os.path.join(WVN_ROOT_DIR, "scripts/abblations/time_adaptation.pkl")
        os.system(f"rm {p}")
        with open(p, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
