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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import copy
import os
import pickle
from pathlib import Path

env = load_env()
number_training_runs = 5
test_all_datasets = False

models = {
    # "KNN1": KNeighborsClassifier(n_neighbors=1, weights="uniform"),
    # "KNN3": KNeighborsClassifier(n_neighbors=3, weights="uniform"),
    "SVMpoly": SVC(kernel="poly", degree=2, probability=True),
    "SVMrbf": SVC(kernel="rbf", probability=True),
    "RandomForest50": RandomForestClassifier(),
    # "MLP": MLPClassifier(hidden_layer_sizes=(256,32), alpha=0, batch_size=800, learning_rate_init=0.001, max_iter=10000),
    # "MLP64": MLPClassifier(hidden_layer_sizes=(256,32), alpha=0, batch_size=64, learning_rate_init=0.001, max_iter=10000),
    # "MLP64_Slow": MLPClassifier(hidden_layer_sizes=(256,32), alpha=0, batch_size=64, learning_rate_init=0.0001, max_iter=10000)
    "RandomForest500": RandomForestClassifier(),
}

# n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_featu

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

            test_auroc_gt_image = AUROC("binary")
            test_auroc_prop_image = AUROC("binary")
            test_acc_gt_image = Accuracy("binary")
            test_acc_prop_image = Accuracy("binary")

            test_auroc_gt = AUROC("binary")
            test_auroc_prop = AUROC("binary")
            v.fit(features, labels)

            test_reslts = {}

            test_reslts_img = {}
            for test_dataset in test_datasets:
                for j, (d, test_scene) in enumerate(zip(test_dataset, test_scenes)):
                    y_pred = torch.from_numpy(v.predict_proba(d.x.cpu().numpy())[:, 1])
                    BS, H, W = d.label.shape
                    # project graph predictions and label onto the image
                    buffer_pred = d.label.clone().type(torch.float32).flatten()
                    buffer_prop = d.label.clone().type(torch.float32).flatten()
                    seg_pixel_index = (d.seg).flatten()

                    buffer_pred = y_pred[seg_pixel_index].reshape(BS, H, W)
                    buffer_prop = d.y[seg_pixel_index].reshape(BS, H, W)

                    test_auroc_gt_image(preds=buffer_pred, target=d.label.type(torch.long))
                    test_auroc_prop_image(preds=buffer_pred, target=buffer_prop.type(torch.long).cpu())

                    test_acc_gt_image(preds=buffer_pred, target=d.label.type(torch.long))
                    test_acc_prop_image(preds=buffer_pred, target=buffer_prop.type(torch.long).cpu())

                res = {
                    "test_auroc_gt_image": test_auroc_gt_image.compute().item(),
                    "test_auroc_self_image": test_auroc_prop_image.compute().item(),
                    "test_acc_gt_image": test_acc_gt_image.compute().item(),
                    "test_acc_self_image": test_acc_prop_image.compute().item(),
                }
                test_reslts_img[test_scene] = res
                print(
                    f"{model_name}:  {scene} -test {test_scene} ---- AUROC_GT_Image {test_auroc_gt_image.compute():.3f} , AUROC_SELF_Image {test_auroc_prop_image.compute():.3f}"
                )

            for features_test, y_gt, y_prop, test_scene in zip(features_tests, y_gts, y_props, test_scenes):
                y_pred = v.predict_proba(features_test)
                test_auroc_gt.update(torch.from_numpy(y_pred[:, 1]), torch.from_numpy(y_gt).type(torch.long))
                test_auroc_prop.update(torch.from_numpy(y_pred[:, 1]), torch.from_numpy(y_prop).type(torch.long))

                res = {
                    "test_auroc_gt_seg": test_auroc_gt.compute().item(),
                    "test_auroc_self_seg": test_auroc_prop.compute().item(),
                }
                res.update(test_reslts_img[test_scene])

                print(
                    f"{model_name}:  {scene} -test {test_scene} ---- AUROC_GT {test_auroc_gt.compute():.3f} , AUROC_SELF {test_auroc_prop.compute():.3f}"
                )

                test_reslts[test_scene] = res
            run_results[str(run)] = copy.deepcopy(test_reslts)
        model_results[model_name] = copy.deepcopy(run_results)
    results_epoch[scene] = copy.deepcopy(model_results)

    ws = os.environ["ENV_WORKSTATION_NAME"]
    # Store epoch output to disk.
    p = os.path.join(
        env["base"], f"ablations/classicial_learning_ablation_{ws}/classicial_learning_ablation_test_results.pkl"
    )
    print(results_epoch)
    Path(p).parent.mkdir(parents=True, exist_ok=True)

    try:
        os.remove(p)
    except OSError as error:
        pass

    with open(p, "wb") as handle:
        pickle.dump(results_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
