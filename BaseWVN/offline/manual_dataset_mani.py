import os
from offline_training_lightning import *
from BaseWVN.config.wvn_cfg import ParamCollection,save_to_yaml

def modify_dataset_from_nodes(param:ParamCollection,nodes:List[MainNode],add_mask):
    for i,node in enumerate(nodes[:3]):
        img=node.image.to(param.run.device)
        cur_mask=add_mask[i]
        expanded_mask = cur_mask.repeat(2, 1, 1)
        # Now apply the mask to set the desired positions to NaN
        # We use where to selectively replace values where the condition is True
        node.supervision_mask = torch.where(expanded_mask, torch.tensor(float('nan')).to(param.run.device), node.supervision_mask)
        node.supervision_signal_valid=torch.where(expanded_mask,False, node.supervision_signal_valid)

def resample_train_data(param:ParamCollection,nodes:List[MainNode]):
    res=[]
    for i in range(len(nodes)):
        for _ in range(2):
            batch_list=[mnode.query_valid_batch() for i, mnode in enumerate(nodes[0:i+1])]
            dataset=VD_dataset(batch_list,combine_batches=True,random_num=param.graph.random_sample_num)
            res.append(dataset)
    return res
param=ParamCollection()
param.offline.env="vowhite_1st"
# load train and val data from nodes_datafile (should include all pixels of supervision masks)
path=os.path.join(param.offline.data_folder,"train",param.offline.env,param.offline.nodes_datafile)
train_data_raw=load_data(path)
# 17*1*910*910
add_mask=torch.load(os.path.join(param.offline.data_folder,"train",param.offline.env,"white_board.pt"))

train_data=torch.load(os.path.join(param.offline.data_folder,"train",param.offline.env,"train_data.pt"))

modify_dataset_from_nodes(param,train_data_raw,add_mask)
new_train_data=resample_train_data(param,train_data_raw)
torch.save(new_train_data,os.path.join(param.offline.data_folder,"train",param.offline.env,"train_data_new.pt"))
torch.save(train_data_raw,os.path.join(param.offline.data_folder,"train",param.offline.env,"train_nodes_new.pt"))


