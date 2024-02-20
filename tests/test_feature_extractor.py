from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.visu import get_img_from_fig
from wild_visual_navigation.utils.testing import load_test_image, get_dino_transform, make_results_folder
from os.path import join
from pytictac import Timer
import matplotlib.pyplot as plt
import torch
import itertools


def test_feature_extractor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmentation_types = ["none", "grid", "slic", "random", "stego"]
    feature_types = ["dino", "dinov2", "stego"]
    backbone_types = ["vit_small", "vit_base"]  # "vit_small_reg", "vit_base_reg"]

    for seg_type, feat_type, back_type in itertools.product(segmentation_types, feature_types, backbone_types):
        if seg_type == "stego" and feat_type != "stego":
            continue

        with Timer(f"Running seg [{seg_type}], feat [{feat_type}], backbone [{back_type}]"):
            try:
                fe = FeatureExtractor(
                    device, segmentation_type=seg_type, feature_type=feat_type, backbone_type=back_type
                )
            except Exception:
                print("Not available")
                continue

        img = load_test_image().to(device)
        transform = get_dino_transform()
        outpath = make_results_folder("test_feature_extractor")

        # Compute
        edges, feat, seg, center, dense_feat = fe.extract(transform(img.clone()))

        # Plot result as in colab
        fig, ax = plt.subplots(1, 2, figsize=(5 * 3, 5))

        ax[0].imshow(transform(img).permute(0, 2, 3, 1)[0].cpu())
        ax[0].set_title("Image")
        ax[1].imshow(seg.cpu(), cmap=plt.colormaps.get("gray"))
        ax[1].set_title("Segmentation")
        plt.tight_layout()

        # Store results to test directory
        img = get_img_from_fig(fig)
        img.save(join(outpath, f"forest_clean_graph_{seg_type}_{feat_type}.png"))
        plt.close()


if __name__ == "__main__":
    test_feature_extractor()
