import torch
import os
from BaseWVN import WVN_ROOT_DIR
from BaseWVN.model import VD_dataset
from BaseWVN.config.wvn_cfg import ParamCollection
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import List

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, data:List[VD_dataset]):
        self.data=[]
        for d in data:
            self.data = self.data+d.batches
        
        pass
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


def load_data(folder, file):
    """Load data from the data folder."""
    path=os.path.join(WVN_ROOT_DIR, folder, file)
    data=torch.load(path)
    return data


def train_and_evaluate():
    """Train and evaluate the model."""
    folder='results/manager'
    file='graph_data.pt'
    data=load_data(folder, file)
    
    combined_dataset = BigDataset(data)
    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size

    train_dataset = Subset(combined_dataset, range(0, train_size))
    val_dataset = Subset(combined_dataset, range(train_size, len(combined_dataset)))
    
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # batch in loader is a tuple of (xs, ys)
    # xs:(1, 100, feat_dim), ys:(1, 100, 2)
    for batch in train_loader:
        
        print(batch)
    pass

if __name__ == "__main__":
    train_and_evaluate()
    