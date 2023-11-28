import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime
from BaseWVN import WVN_ROOT_DIR
from BaseWVN.config.wvn_cfg import ParamCollection,save_to_yaml
from typing import List
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from BaseWVN.offline.offline_training_lightning import train_and_evaluate

def generalization_test():
    """ 
    First train on hiking dataset, then do validation on snow dataset.
    All the results save to a folder named 'generalization_test'
    
    """