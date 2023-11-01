# Use the same config to load the data using the dataloader
from wild_visual_navigation.dataset import get_ablation_module
from wild_visual_navigation.utils import load_env
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.visu import LearningVisualizer
import torch
from torchmetrics import Accuracy, AUROC
from PIL import Image
import numpy as np
import copy
import os
import pickle
from pathlib import Path
import time

env = load_env()
test_all_datasets = True

for scene in ["forest", "hilly", "grassland"]:
    ablation_data_module = {
        "batch_size": 1,
        "num_workers": 0,
        "env": scene,
        "feature_key": "slic_dino",
        "test_equals_val": False,
        "val_equals_test": False,
        "test_all_datasets": test_all_datasets,
        "training_data_percentage": 100,
        "training_in_memory": False,
    }
    train_loader, val_loader, test_loader = get_ablation_module(
        **ablation_data_module, perugia_root=env["perugia_root"]
    )
    test_scenes = [a.dataset.env for a in test_loader]

    visualizer = LearningVisualizer()
    import matplotlib.pyplot as plt

    # You probably won't need this if you're embedding things in a tkinter plot...
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    obj = ax.imshow(np.zeros((100, 100, 3)))
    for t in test_loader:
        for j, d in enumerate(t):
            graph = d[0]
            b = 0
            center = graph.center[:]
            seg = graph.seg
            img = graph.img
            # res = visualizer.plot_traversability_graph_on_seg(
            #         graph.y_gt.type(torch.float32), seg[0], graph, center, img[0], not_log=True
            #     )

            res = visualizer.plot_detectron(img[0], graph.label[0].type(torch.long), not_log=True, max_seg=2)
            i1 = Image.fromarray(res)

            obj.set_data(i1)

            fig.canvas.draw()
            fig.canvas.flush_events()
            print(j)
            time.sleep(0.25)
