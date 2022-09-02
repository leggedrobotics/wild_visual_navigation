# Use the same config to load the data using the dataloader
from wild_visual_navigation.learning.dataset import get_abblation_module
from wild_visual_navigation.learning.utils import load_env
import torch
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from torchmetrics import Accuracy, AUROC

test_auroc_gt = AUROC()
test_auroc_prop = AUROC()
        
env = load_env()

for env_ in ["forest", "hilly", "grassland"]:
    abblation_data_module = {
        "batch_size": 1,
        "num_workers": 0,
        "env": env_,
        "feature_key": "slic_dino",
        "test_equals_val": False,
    }

    datamodule = get_abblation_module(**abblation_data_module, perugia_root=env["perugia_root"])
    knn = KNeighborsClassifier(n_neighbors=1, weights="uniform")

    features = []
    labels = []

    for j, d in enumerate( datamodule.train_dataset ):
        features.append( d[0].x.cpu().numpy() )
        labels.append( d[0].y.cpu().numpy() ) 
        
    features = np.concatenate( features, axis=0 )
    labels = np.concatenate( labels, axis=0)

    knn.fit( features, labels )

    features_test = []
    y_gt = []
    y_prop = []

    acc = Accuracy()
    for j, d in enumerate( datamodule.test_dataset ):
        features_test.append( d[0].x.cpu().numpy() ) 
        y_gt.append( d[0].y_gt.cpu().numpy() )
        y_prop.append( d[0].y.cpu().numpy() )

    features_test = np.concatenate( features_test, axis=0 )
    y_gt = np.concatenate( y_gt, axis=0)
    y_prop = np.concatenate( y_prop, axis=0)
    y_pred = knn.predict(features_test)

    test_auroc_gt.update( torch.from_numpy( y_pred ), torch.from_numpy( y_gt ).type(torch.long) )
    test_auroc_prop.update( torch.from_numpy( y_pred ), torch.from_numpy( y_prop ).type(torch.long) )
    acc.update( torch.from_numpy( y_pred ), torch.from_numpy( y_gt ).type(torch.long) )
    print(f"{env_} ---- AUROC_GT {test_auroc_gt.compute():.3f} , ACC_GT {acc.compute():.3f}, AUROC_PROP {test_auroc_prop.compute():.3f}")


