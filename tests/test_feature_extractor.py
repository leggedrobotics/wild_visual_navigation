from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation import WVN_ROOT_DIR
import torch
from torchvision import transforms as T
from pathlib import PurePath, Path
import os
import cv2


def test_feature_extractor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = FeatureExtractor(device, segmentation_type="stego", feature_type="stego")

    transform = T.Compose(
        [
            T.Resize(448, T.InterpolationMode.NEAREST),
            T.CenterCrop(448),
        ]
    )

    np_img = cv2.imread(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)

    img = (img.type(torch.float32) / 255)[None]
    adj, feat, seg, center = fe.extract(transform(img.clone()))

    p = PurePath(WVN_ROOT_DIR).joinpath("results", "test_feature_extractor", "forest_clean_graph.png")
    Path(p.parent).mkdir(parents=True, exist_ok=True)
    img.save(str(p))


if __name__ == "__main__":
    test_feature_extractor()
