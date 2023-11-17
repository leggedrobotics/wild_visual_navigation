import torch

a=torch.tensor([1,2,3]).unsqueeze(0)
b=torch.tensor([4,5]).unsqueeze(0)

print(torch.cat((a,b),dim=1))
