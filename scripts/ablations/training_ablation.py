raise ValueError("TODO: Not tested with new configuration!")
import os
import torch
from pathlib import Path
import pickle
import copy
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.utils import load_yaml
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.general import training_routine
from wild_visual_navigation.utils import override_params
import argparse
import shutil

from dataclasses import asdict

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="network",
        help="Folder containing the ablation configs.",
    )
    parser.add_argument("--number_training_runs", type=int, default=1, help="Number of run per config.")
    parser.add_argument(
        "--test_all_datasets",
        dest="test_all_datasets",
        action="store_true",
        help="Test on all datasets.",
    )
    parser.add_argument(
        "--store_final_model",
        dest="store_final_model",
        action="store_true",
        help="store_final_model on all datasets.",
    )
    parser.add_argument(
        "--scenes",
        default="forest,hilly,grassland",
        type=str,
        help="List of scenes seperated by comma without spaces.",
    )
    parser.add_argument("--special_key", type=str, default="", help="Test on all datasets.")
    parser.set_defaults(test_all_datasets=False)
    parser.set_defaults(store_final_model=False)

    args = parser.parse_args()
    exp = ExperimentParams()
    stored_params = asdict(exp)

    def get_exp(args, model_path, p, scene, stored_params):
        exp = ExperimentParams()
        override_params(exp, copy.deepcopy(stored_params))

        exp.general.log_to_disk = False
        exp.trainer.max_steps = 1000
        exp.trainer.max_epochs = None
        exp.logger.name = "skip"
        exp.ablation_data_module.val_equals_test = False
        exp.ablation_data_module.test_all_datasets = args.test_all_datasets
        exp.trainer.profiler = None
        exp.trainer.enable_checkpointing = False
        exp.cb_checkpoint.active = False
        exp.ablation_data_module.training_in_memory = True
        exp.visu.train = 0
        exp.visu.val = 0
        exp.visu.test = 0
        exp.trainer.check_val_every_n_epoch = 100000
        exp.general.store_model_every_n_steps = None
        exp.general.model_path = model_path
        exp.trainer.progress_bar_refresh_rate = 0
        exp.trainer.weights_summary = None
        exp.trainer.enable_progress_bar = False
        exp.ablation_data_module.env = scene
        exp_override = load_yaml(p)
        override_params(exp, exp_override)
        exp.verify_params()
        return exp

    number_training_runs = args.number_training_runs
    folder = args.ablation_type
    special_key = args.special_key
    ws = os.environ.get("ENV_WORKSTATION_NAME", "default")
    model_path = os.path.join(WVN_ROOT_DIR, f"results/ablations/{folder}_ablation_{ws}")
    shutil.rmtree(model_path, ignore_errors=True)
    Path(model_path).mkdir(parents=True, exist_ok=True)

    directory = Path(os.path.join(WVN_ROOT_DIR, f"cfg/exp/ablation/{folder}"))
    cfg_paths = [str(p) for p in directory.rglob("*.yaml") if str(p).find("template") == -1]
    # Train model and get test results for every epoch.
    results_epoch = {}
    j = 0
    scenes = args.scenes.split(",")
    print(f"The configuration files in folder {folder} will be evaluated on {scenes}")
    for scene in scenes:
        print(f"Scene: {scene}")
        model_results = {}

        for p in cfg_paths:
            run_results = {}
            for run in range(number_training_runs):
                p = str(p)
                print(f"Run number {j}: Scene {scene}, Run: {run}, Config: {p}")
                exp = get_exp(args, model_path, p, scene, stored_params)
                res, model = training_routine(exp, seed=run)
                run_results[str(run)] = copy.deepcopy(res)
                j += 1

                if args.store_final_model:
                    p_ = p.split("/")[-1][:-5]
                    p_ = os.path.join(exp.general.model_path, f"model_{p_}_{scene}_{run}.pt")
                    torch.save(model.state_dict(), p_)
                    run_results[str(run)]["model"] = model.state_dict()
                    run_results[str(run)]["model_path"] = p_

            model_results[p] = copy.deepcopy(run_results)
        results_epoch[scene] = copy.deepcopy(model_results)

    # Store epoch output to disk.
    p = os.path.join(exp.general.model_path, f"{folder}_ablation_test_results{special_key}.pkl")

    try:
        os.remove(p)
    except OSError as error:
        print(error)

    with open(p, "wb") as handle:
        pickle.dump(results_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
