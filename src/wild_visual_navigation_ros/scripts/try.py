import torch
import numpy as np
import os
from BaseWVN import WVN_ROOT_DIR
a=torch.tensor([1.0,2,3]).unsqueeze(0)
b=torch.tensor([4,5]).unsqueeze(0)

print(torch.cat((a,b),dim=1))

c="debug"
d="debug+label"
if "debug" in d:
    print("yes")

a[0,0]=torch.nan
e=a.numpy()
g=np.clip(e,0,1)
v=[2,255,167,0]
print(min(v[:-1]))

from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="/media/chen/UDisk1/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)
pass
