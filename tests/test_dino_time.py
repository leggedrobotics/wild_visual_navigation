from wild_visual_navigation.feature_extractor import DinoInterface
import cv2
import os 
import torch
from wild_visual_navigation.utils import Timer
from wild_visual_navigation import WVN_ROOT_DIR

def test_dino_interfacer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    di = DinoInterface(device)
    
    np_img = cv2.imread(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
   
    for i in range(10):
        im = img + torch.rand(img.shape, device=img.device)/100
        di.inference(di.transform( im )) 
    
    with Timer("Dino Inference: "):
        for i in range(30):
            im = img + torch.rand(img.shape, device=img.device)/100
            with Timer("Dino Single: "):
                res = di.inference(di.transform( im ))

if __name__ == "__main__":
    test_dino_interfacer()