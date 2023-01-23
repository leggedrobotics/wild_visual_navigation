from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.visu import get_img_from_fig
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
from pathlib import PurePath, Path
import os
import cv2


def test_feature_extractor():
    segmentation_type = "slic"
    feature_type = "dino"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    K = torch.FloatTensor([[720, 0, 720, 0], [0, 720, 540, 0], [0, 0, 1, 0], [0, 0, 0, 1]])[None]
    # Image size
    H = 1080
    W = 1440
    new_H = 448
    new_W = 448
    ip = ImageProjector(K, H, W, new_h=new_H, new_w=new_W)
    fe = FeatureExtractor(device, segmentation_type=segmentation_type, feature_type=feature_type, input_size=(new_H, new_W))

    np_img = cv2.imread(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]

    img = ip.resize_image(img)
    adj, feat, seg, center = fe.extract(img.clone())

    p = PurePath(WVN_ROOT_DIR).joinpath(
        "results", "test_feature_extractor", f"forest_clean_graph_{segmentation_type}.png"
    )
    Path(p.parent).mkdir(parents=True, exist_ok=True)

    # Plot result as in colab
    fig, ax = plt.subplots(1, 2, figsize=(5 * 3, 5))

    ax[0].imshow(img.permute(0, 2, 3, 1)[0].cpu())
    ax[0].set_title("Image")
    ax[1].imshow(seg.cpu(), cmap=plt.colormaps.get("inferno"))
    ax[1].set_title("Segmentation")
    plt.tight_layout()

    # Store results to test directory
    img = get_img_from_fig(fig)
    img.save(str(p))


if __name__ == "__main__":
    test_feature_extractor()
