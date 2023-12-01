import torch
import random
from typing import List
from ..model import VD_dataset
class BigDataset(torch.utils.data.Dataset):
    def __init__(self, data:List[VD_dataset],sample_size: int = None):
        self.data=[]
        for d in data:
            self.data = self.data+d.batches
        
        # If a sample size is specified and is less than the total data size
        if sample_size and sample_size < len(self.data):
            self.data = random.sample(self.data, sample_size)

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    
class EntireDataset(torch.utils.data.Dataset):
    def __init__(self, data: VD_dataset):
        if len(data.batches) > 1:
            self.x = torch.cat([data.batches[i][0] for i in range(len(data.batches))], dim=0)
            self.y = torch.cat([data.batches[i][1] for i in range(len(data.batches))], dim=0)
        else:
            self.x, self.y = data.batches[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

