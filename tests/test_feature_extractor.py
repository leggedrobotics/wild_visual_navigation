from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation import WVN_ROOT_DIR

import torch
import os
from PIL import Image
from pathlib import PurePath, Path
import cv2


def test_feature_extractor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = FeatureExtractor(device)

    np_img = cv2.imread(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2,0,1)

    img = (img.type(torch.float32) / 255)[None]
    adj, feat, seg, center, img = fe.dino_slic(img.clone(), return_centers=True, return_image=True)

    p = PurePath(WVN_ROOT_DIR).joinpath("results", "test_feature_extractor", "forest_clean_graph.png")
    Path(p.parent).mkdir(parents=True, exist_ok=True)
    img.save(str(p))


if __name__ == "__main__":
    test_feature_extractor()
