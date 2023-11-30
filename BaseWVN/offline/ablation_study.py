import matplotlib.pyplot as plt
import datetime
import torch
import numpy as np
from BaseWVN.config.wvn_cfg import ParamCollection,save_to_yaml
from typing import List
from BaseWVN.offline.offline_training_lightning import train_and_evaluate
import os 
from BaseWVN import WVN_ROOT_DIR

def run_scenario(scenario_name, ckpt_parent_folder,reload_model,use_online_ckpt,dataset_folder,test_only):
    """ 
    Return the test stats dict of the scenario.
    
    """
    param=ParamCollection()
    torch.cuda.empty_cache()
    if "hiking" in scenario_name:
        param.offline.env='hiking'
    elif "snow" in scenario_name:
        param.offline.env='snow'
    else:
        raise ValueError("scenario_name must contain 'hiking' or 'snow'")
    
    if not test_only:
        param.offline.mode='train'
        param.offline.reload_model=reload_model
        param.offline.use_online_ckpt=use_online_ckpt
        param.offline.ckpt_parent_folder=ckpt_parent_folder
        param.offline.data_folder=dataset_folder
        param.general.name=scenario_name
        train_and_evaluate(param)
        # if use_online_ckpt to train, after training, the new model is no longer in online folder
        use_online_ckpt=False
    torch.cuda.empty_cache()
    param.offline.mode='test'
    param.offline.reload_model=False
    param.offline.use_online_ckpt=use_online_ckpt
    param.offline.ckpt_parent_folder=ckpt_parent_folder
    param.offline.data_folder=dataset_folder
    param.general.name=scenario_name
    
    return train_and_evaluate(param)
    

def test(config:dict):
    """ 
    First train on hiking dataset, then do validation on snow dataset.
    All the results save to a folder named 'generalization_test'
    
    """
    number=config["number"]
    ckpt_parent_folder=config["ckpt_parent_folder"]
    for item in config["agenda"]:
        item['ckpt_parent_folder'] = ckpt_parent_folder
    agenda=config["agenda"]
    
    aggregate_stats = {scenario['name']: {'fric_mean': [], 'fric_std': [], 'stiffness_mean': [], 'stiffness_std': [], 'over_conf_mean': [], 'over_conf_std': [], 'under_conf_mean': [], 'under_conf_std': []} for scenario in agenda}

    for _ in range(number):  # Assuming you want to repeat the whole process 5 times
        for scenario in agenda:
            stats = run_scenario(scenario['name'], 
                                 scenario['ckpt_parent_folder'], 
                                 scenario['reload_model'], 
                                 scenario['use_online_ckpt'], 
                                 scenario['dataset_folder'], 
                                 scenario['test_only'])
            if stats:  # Stats will be None if the mode is 'train'
                for key in aggregate_stats[scenario['name']]:
                    aggregate_stats[scenario['name']][key].append(stats[key].detach().cpu().numpy() if isinstance(stats[key], torch.Tensor) else stats[key])

    # Calculate mean and std for each metric
    for scenario_name in aggregate_stats:
        for metric in aggregate_stats[scenario_name]:
            values = aggregate_stats[scenario_name][metric]
            aggregate_stats[scenario_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    time=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_stats_to_file(aggregate_stats,os.path.join(WVN_ROOT_DIR,ckpt_parent_folder,f'{time}_stats.txt'))
    return aggregate_stats
   
def save_stats_to_file(stats, filename):
    with open(filename, 'w') as file:
        for scenario, scenario_stats in stats.items():
            file.write(f"Scenario: {scenario}\n")
            for metric, values in scenario_stats.items():
                file.write("{}: Mean = {:.3f}, Std = {:.3f}\n".format(metric, values['mean'], values['std']))
            file.write("\n")

if __name__ == "__main__":
    from BaseWVN.offline.ablation_cfg import*
    test(generalization_cfg)
   