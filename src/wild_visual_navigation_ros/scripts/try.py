import torch
import numpy as np
import os
from BaseWVN import WVN_ROOT_DIR
a=torch.tensor([1.0,-2,3]).unsqueeze(0)
b=torch.tensor([4,5,6]).unsqueeze(0)

print(torch.cat((a,b),dim=1))

c="debug"
d="debug+label"
if "debug" in d:
    print("yes")

a[0,0]=torch.nan
ab=torch.cat((a,b))
print(torch.amin(ab,dim=0))
print(torch.fmin(a,b))
print(torch.std(a))
# e=a.numpy()
# g=np.clip(e,0,1)
# v=[2,255,167,0]
# print(min(v[:-1]))

# cc=a.type(torch.float32)
# print(a.max())

# dd=np.array([1,2,3])
# ee=np.array([4,5,6])
# print(dd+ee)
# print(torch.zeros(0, 5, 3).shape)

# gg=[]
# if gg:
#     print("empty list is true")
# else:
#     print("empty list is false")
# pass
