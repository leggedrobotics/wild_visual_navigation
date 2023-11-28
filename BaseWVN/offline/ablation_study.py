import matplotlib.pyplot as plt
import datetime
from BaseWVN.config.wvn_cfg import ParamCollection,save_to_yaml
from typing import List
from BaseWVN.offline.offline_training_lightning import train_and_evaluate
import copy
def generalization_test():
    """ 
    First train on hiking dataset, then do validation on snow dataset.
    All the results save to a folder named 'generalization_test'
    
    """
    hiking_dataset_folder='results/manager/hiking'
    snow_dataset_folder='results/manager/snow'
    
    param=ParamCollection()
    # param=copy.deepcopy(ori_param)
    param.offline.mode='train'
    param.offline.reload_model=False
    param.offline.use_online_ckpt=False
    param.offline.ckpt_parent_folder='results/generalization_test'
    param.offline.data_folder=hiking_dataset_folder
    
    param.general.name='1.train_on_hiking'
    train_and_evaluate(param)
    
    # param=ParamCollection()
    param.offline.mode='test'
    param.offline.reload_model=False
    param.offline.use_online_ckpt=False
    param.offline.ckpt_parent_folder='results/generalization_test'
    param.offline.data_folder=hiking_dataset_folder
    
    param.general.name='1.train_on_hiking'
    train_and_evaluate(param)
    
    # param2=ParamCollection()
    param.offline.use_online_ckpt=False
    param.offline.mode='test'
    param.offline.ckpt_parent_folder='results/generalization_test'
    param.offline.data_folder=snow_dataset_folder
    param.general.name='2.test_on_snow'
    train_and_evaluate(param)

if __name__ == "__main__":
    generalization_test()