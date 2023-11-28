import matplotlib.pyplot as plt
import datetime
from BaseWVN.config.wvn_cfg import ParamCollection,save_to_yaml
from typing import List
from BaseWVN.offline.offline_training_lightning import train_and_evaluate

hiking_dataset_folder='results/manager/hiking'
snow_dataset_folder='results/manager/snow'

def generalization_test():
    """ 
    First train on hiking dataset, then do validation on snow dataset.
    All the results save to a folder named 'generalization_test'
    
    """
    
    param=ParamCollection()
    param.offline.mode='train'
    param.offline.reload_model=False
    param.offline.use_online_ckpt=False
    param.offline.ckpt_parent_folder='results/generalization_test'
    param.offline.data_folder=hiking_dataset_folder
    
    param.general.name='1.train_on_hiking'
    train_and_evaluate(param)
    
    param.offline.mode='test'
    param.offline.reload_model=False
    param.offline.use_online_ckpt=False
    param.offline.ckpt_parent_folder='results/generalization_test'
    param.offline.data_folder=hiking_dataset_folder
    
    param.general.name='1.train_on_hiking'
    train_and_evaluate(param)
    
    param.offline.use_online_ckpt=False
    param.offline.mode='test'
    param.offline.ckpt_parent_folder='results/generalization_test'
    param.offline.data_folder=snow_dataset_folder
    param.general.name='2.test_on_snow'
    train_and_evaluate(param)

def memory_test():
    """ 
    First train on hiking dataset, then continue training on snow dataset. 
    At last, do validation on hiking dataset.
    All the results save to a folder named 'memory_test'
    
    """
    param=ParamCollection()
    param.offline.mode='train'
    param.offline.reload_model=False
    param.offline.use_online_ckpt=False
    param.offline.ckpt_parent_folder='results/memory_test'
    param.offline.data_folder=hiking_dataset_folder
    
    param.general.name='1.train_on_hiking'
    train_and_evaluate(param)
    
    param.offline.mode='test'
    param.offline.use_online_ckpt=False
    param.offline.ckpt_parent_folder='results/memory_test'
    param.offline.data_folder=hiking_dataset_folder
    
    param.general.name='1.train_on_hiking'
    train_and_evaluate(param)
    
    param.offline.use_online_ckpt=False
    param.offline.mode='train'
    param.offline.reload_model=True
    param.offline.ckpt_parent_folder='results/memory_test'
    param.offline.data_folder=snow_dataset_folder
    param.general.name='2.resume_train_on_snow'
    train_and_evaluate(param)
    
    param.offline.use_online_ckpt=False
    param.offline.mode='test'
    param.offline.ckpt_parent_folder='results/memory_test'
    param.offline.data_folder=snow_dataset_folder
    param.general.name='2.resume_train_on_snow'
    train_and_evaluate(param)
    
    param.offline.mode='test'
    param.offline.use_online_ckpt=False
    param.offline.ckpt_parent_folder='results/memory_test'
    param.offline.data_folder=hiking_dataset_folder
    param.general.name='3.retest_on_hiking'
    train_and_evaluate(param)
    

if __name__ == "__main__":
    # generalization_test()
    memory_test()