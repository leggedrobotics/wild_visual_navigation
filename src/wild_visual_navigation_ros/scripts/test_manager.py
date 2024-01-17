
from BaseWVN import WVN_ROOT_DIR
from pytorch_lightning import seed_everything
from torch_geometric.data import Data, Batch
from threading import Lock
import dataclasses
import os
import pickle
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

from BaseWVN.GraphManager import (
    BaseGraph,
    DistanceWindowGraph,
    MaxElementsGraph,
    MainNode,SubNode,Manager
)
from BaseWVN.utils import ImageProjector

to_tensor = transforms.ToTensor()
device='cuda'
manager_path=os.path.join(WVN_ROOT_DIR,'results/manager')
filename='try1.pkl'
os.makedirs(manager_path, exist_ok=True)
output_file = os.path.join(manager_path, filename)
if not filename.endswith('.pkl') and not filename.endswith('.pickle'):
    output_file += '.pkl'  # Append .pkl if not already present
manager:Manager = pickle.load(open(output_file, "rb"))

first_main_node=manager._main_graph.get_first_node()

# Set color
color = torch.ones((3,), device=device)




pass 
    