raise ValueError("TODO: Not tested with new configuration!")
import os
import torch
from pathlib import Path
import time
import pickle
import copy
import argparse
import shutil
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.general import training_routine

if __name__ == "__main__":
    """Test how much time and data it takes for a model to convergee on a scene.
    During training we store checkpoints of the models.
    After training we run for all model checkpoints the test routine.
    """
    exp = ExperimentParams()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_key", type=str, default="learning_curve", help="Name of the run.")
    parser.add_argument("--number_training_runs", type=int, default=5, help="Number of run per config.")
    parser.add_argument("--data_start_percentage", type=int, default=100)
    parser.add_argument("--data_stop_percentage", type=int, default=100)
    parser.add_argument("--data_percentage_increment", type=int, default=10)
    parser.add_argument("--test_all_datasets", dest="test_all_datasets", action="store_true", help="")
    parser.set_defaults(test_all_datasets=False)
    parser.add_argument(
        "--scenes",
        default="forest,hilly,grassland",
        type=str,
        help="List of scenes seperated by comma without spaces.",
    )
    parser.add_argument("--store_model_every_n_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)

    # python scripts/ablations/stepwise_ablation.py --output_key time_adaptation --number_training_runs 5 --data_start_percentage 100 --data_stop_percentage 100 --data_percentage_increment 10 --scenes forest,hilly,grassland --store_model_every_n_steps 100

    # python scripts/ablations/stepwise_ablation.py --output_key data_percentage --number_training_runs 1 --data_start_percentage 10 --data_stop_percentage 100 --data_percentage_increment 10 --scenes forest --store_model_every_n_steps 50 --max_steps 1000

    args = parser.parse_args()

    # Change Experiment Params
    output_key = args.output_key
    number_training_runs = args.number_training_runs
    data_start_percentage = args.data_start_percentage
    data_stop_percentage = args.data_stop_percentage
    data_percentage_increment = args.data_percentage_increment
    scenes = args.scenes.split(",")
    exp.test_all_datasets = args.test_all_datasets
    exp.general.store_model_every_n_steps = args.store_model_every_n_steps

    # Ensure deafult configuration
    exp.trainer.max_steps = args.max_steps
    exp.ablation_data_module.training_in_memory = True
    exp.trainer.check_val_every_n_epoch = 1000000
    exp.general.log_to_disk = False
    exp.trainer.max_epochs = None
    exp.logger.name = "skip"
    exp.ablation_data_module.val_equals_test = True
    exp.trainer.profiler = None
    exp.trainer.enable_checkpointing = False
    exp.cb_checkpoint.active = False
    exp.visu.train = 0
    exp.visu.val = 0
    exp.visu.test = 0
    exp.verify_params()
    ws = os.environ.get("ENV_WORKSTATION_NAME", "default")
    exp.general.model_path = os.path.join(exp.env.results, f"ablations/{output_key}_{ws}")

    # If check_val_every_n_epoch in the current setting the test dataloader is used for validation.
    # All results during validation are stored and returned by the training routine.

    # Currently the model weights are stored every n steps.
    # This allows to reload the model and test it on the test dataloader.
    train_and_delete = True
    if train_and_delete:
        shutil.rmtree(exp.general.model_path, ignore_errors=True)

        Path(exp.general.model_path).mkdir(parents=True, exist_ok=True)
        percent = range(
            data_start_percentage,
            data_stop_percentage + data_percentage_increment,
            data_percentage_increment,
        )

        with open(os.path.join(exp.general.model_path, "experiment_params.pkl"), "wb") as handle:
            pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Train model in various configurations and the validation results per epoch are returned in results_epoch.
        results_epoch = {}
        for scene in scenes:
            exp.ablation_data_module.env = scene
            percentage_results = {}
            for percentage in percent:
                exp.ablation_data_module.training_data_percentage = percentage
                run_results = {}
                for run in range(number_training_runs):
                    exp.general.store_model_every_n_steps_key = f"ablation_{output_key}_{scene}_{percentage}_{run}"
                    res, _ = training_routine(exp, seed=run)
                    run_results[f"run_{run}"] = copy.deepcopy(res)
                    torch.cuda.empty_cache()
                percentage_results[f"percentage_{percentage}"] = copy.deepcopy(run_results)
            results_epoch[scene] = copy.deepcopy(percentage_results)

        # Store epoch output to disk.
        p = os.path.join(exp.general.model_path, f"{output_key}_epochs.pkl")
        try:
            os.remove(p)
        except OSError as error:
            print(error)
        with open(p, "wb") as handle:
            pickle.dump(results_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Test all stored models on the test dataloader and store the results.
    exp.general.skip_train = True
    exp.ablation_data_module.val_equals_test = False
    results_step = []

    p_inter = os.path.join(exp.general.model_path, f"{output_key}_steps.pkl")
    total_paths = [str(s) for s in Path(exp.general.model_path).rglob("*.pt")]
    print("total_paths", len(total_paths))

    st = time.time()
    for j, p in enumerate(Path(exp.general.model_path).rglob("*.pt")):
        print(
            "XXXXXXXXXXXXXXXXXXXXX PROGRESS ",
            j,
            " of ",
            len(total_paths),
            " in ",
            time.time() - st,
            " seconds.",
        )
        res = str(p).split("/")[-1].split("_")
        scene, percentage, run, steps = res[-4], res[-3], res[-2], res[-1]

        percentage, run, steps = int(percentage), int(run), int(steps.split(".")[0])
        exp.ablation_data_module.env = scene
        exp.model.load_ckpt = str(p)

        res, _ = training_routine(exp, seed=run)
        results_step.append(
            {
                "scene": scene,
                "percentage": percentage,
                "run": run,
                "steps": steps,
                "results": res,
                "model_path": str(p),
            }
        )
        torch.cuda.empty_cache()

        if j % 1000 == 0:
            # Store step output to disk.
            try:
                os.remove(p_inter.replace(".pkl", f"_inter_{j}.pkl"))
            except OSError as error:
                print(error)

            with open(p_inter.replace(".pkl", f"_inter_{j}.pkl"), "wb") as handle:
                pickle.dump(results_step, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Store step output to disk.
    p = os.path.join(exp.general.model_path, f"{output_key}_steps.pkl")
    try:
        os.remove(p)
    except OSError as error:
        print(error)

    with open(p, "wb") as handle:
        pickle.dump(results_step, handle, protocol=pickle.HIGHEST_PROTOCOL)
