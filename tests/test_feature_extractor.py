from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation import WVN_ROOT_DIR

import torch
import os
import kornia as K
from PIL import Image
from pathlib import PurePath, Path


def test_feature_extractor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = FeatureExtractor(device)

    img = K.io.load_image(
        os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"),
        desired_type=K.io.ImageLoadType.RGB8,
        device=device,
    )
    img = (img.type(torch.float32) / 255)[None]
    adj, feat, seg, center, img = fe.dino_slic(img.clone(), return_centers=True, return_image=True)

    p = PurePath(WVN_ROOT_DIR).joinpath("results", "test_feature_extractor", "forest_clean_graph.png")
    Path(p.parent).mkdir(parents=True, exist_ok=True)
    img.save(str(p))


if __name__ == "__main__":
    test_feature_extractor()
