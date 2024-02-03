import os
import cv2
import torch
from torchvision import transforms as T
from wild_visual_navigation import WVN_ROOT_DIR


def load_test_image():
    np_img = cv2.imread(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    return img


def get_dino_transform():
    transform = T.Compose(
        [
            T.Resize(448, T.InterpolationMode.NEAREST),
            T.CenterCrop(448),
        ]
    )
    return transform


def make_results_folder(name):
    path = os.path.join(WVN_ROOT_DIR, "results", name)
    os.makedirs(path, exist_ok=True)
    return path
