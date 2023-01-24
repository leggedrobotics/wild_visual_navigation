# Use the same config to load the data using the dataloader
from wild_visual_navigation.learning.dataset import get_ablation_module
from wild_visual_navigation.learning.utils import load_env
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.cfg import ExperimentParams
import torch
from torchmetrics import Accuracy, AUROC

from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import numpy as np
import copy
import os
import pickle
from pathlib import Path

env = load_env()
number_training_runs = 10
test_all_datasets = True

models = {
    "KNN_1": KNeighborsClassifier(n_neighbors=1, weights="uniform"),
    "KNN_3": KNeighborsClassifier(n_neighbors=3, weights="uniform"),
    "SVM_poly": SVC(kernel="poly", degree=2),
    "SVM_rbf": SVC(kernel="rbf"),
}

results_epoch = {}

for scene in ["forest", "hilly", "grassland"]:
    exp = ExperimentParams()
    ablation_data_module = {
        "batch_size": 1,
        "num_workers": 0,
        "env": scene,
        "feature_key": exp.ablation_data_module.feature_key,
        "test_equals_val": False,
        "val_equals_test": False,
        "test_all_datasets": test_all_datasets,
        "training_data_percentage": 100,
        "training_in_memory": True,
    }
    train_dataset, val_dataset, test_datasets = get_ablation_module(
        **ablation_data_module, perugia_root=env["perugia_root"]
    )
    test_scenes = [a.dataset.env for a in test_datasets]

    features = []
    labels = []
    for j, d in enumerate(train_dataset):
        features.append(d.x.cpu().numpy())
        labels.append(d.y.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    features_tests = []
    y_gts = []
    y_props = []

    for test_dataset in test_datasets:
        features_test = []
        y_gt = []
        y_prop = []
        for j, d in enumerate(test_dataset):
            features_test.append(d.x.cpu().numpy())
            y_gt.append(d.y_gt.cpu().numpy())
            y_prop.append(d.y.cpu().numpy())

        features_test = np.concatenate(features_test, axis=0)
        y_gt = np.concatenate(y_gt, axis=0)
        y_prop = np.concatenate(y_prop, axis=0)

        features_tests.append(features_test)
        y_gts.append(y_gt)
        y_props.append(y_prop)

    model_results = {}
    for model_name, v in models.items():
        run_results = {}
        for run in range(number_training_runs):

            test_auroc_gt = AUROC("binary")
            test_auroc_prop = AUROC("binary")

            v.fit(features, labels)

            test_reslts = {}
            for features_test, y_gt, y_prop, test_scene in zip(features_tests, y_gts, y_props, test_scenes):
                y_pred = v.predict(features_test)
                test_auroc_gt.update(torch.from_numpy(y_pred).type(torch.long), torch.from_numpy(y_gt).type(torch.long))
                test_auroc_prop.update(torch.from_numpy(y_pred), torch.from_numpy(y_prop).type(torch.long))

                res = {
                    "test_auroc_gt_image": test_auroc_gt.compute().item(),
                    "test_auroc_proprioceptive_image": test_auroc_prop.compute().item(),
                }
                print(
                    f"{model_name}:  {scene} -test {test_scene} ---- AUROC_GT {test_auroc_gt.compute():.3f} , AUROC_PROP {test_auroc_prop.compute():.3f}"
                )
                test_reslts[test_scene] = res

            run_results[str(run)] = copy.deepcopy(test_reslts)
        model_results[model_name] = copy.deepcopy(run_results)
    results_epoch[scene] = copy.deepcopy(model_results)

    # Store epoch output to disk.
    p = os.path.join(
        env["base"], "ablations/classicial_learning_ablation/classicial_learning_ablation_test_results.pkl"
    )

    Path(p).parent.mkdir(parents=True, exist_ok=True)

    try:
        os.remove(p)
    except OSError as error:
        pass

    with open(p, "wb") as handle:
        pickle.dump(results_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
