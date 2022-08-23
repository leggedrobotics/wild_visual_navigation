from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.visu import LearningVisualizer
import os
from pathlib import Path
import torch

if __name__ == "__main__":
    visu = True

    mission_folders = ["/media/Data/Datasets/2022_Perugia/wvn_output/day3/2022-05-12T11:56:13_mission_0_day_3"]

    fes = {}
    fes["none_dino"] = FeatureExtractor("cuda", segmentation_type="none", feature_type="dino")
    fes["none_sift"] = FeatureExtractor("cuda", segmentation_type="none", feature_type="sift")
    # fes["none_histogram"] = FeatureExtractor("cuda", segmentation_type="none", feature_type="histogram")

    fes["slic_dino"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="dino")
    fes["slic_sift"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="sift")
    # fes["slic_histogram"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="histogram")

    fes["grid_dino"] = FeatureExtractor("cuda", segmentation_type="grid", feature_type="dino")
    fes["grid_sift"] = FeatureExtractor("cuda", segmentation_type="grid", feature_type="sift")
    # fes["grid_histogram"] = FeatureExtractor("cuda", segmentation_type="grid", feature_type="histogram")

    fes["stego_dino"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="dino")
    fes["stego_sift"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="sift")
    # fes["stego_histogram"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="histogram")

    for mission in mission_folders:
        assert os.path.isdir(os.path.join(mission, "image")), f"{mission} is not a valid mission folder misses image"
        assert os.path.isdir(
            os.path.join(mission, "supervision_mask")
        ), f"{mission} is not a valid mission folder misses supervision_mask"

        stores = ["edges", "feat", "seg", "center"]

        visualizers = {}
        for name, _ in fes.items():
            if visu:
                visualizers[name] = LearningVisualizer(os.path.join(mission, "features", name))

            for s in stores:
                os.makedirs(os.path.join(mission, "features", name, s), exist_ok=True)

        images = [str(s) for s in Path(mission, "image").rglob("*.pt")]
        images.sort()

        for j, image in enumerate(images):
            key = image.split("/")[-1][:-3]  # remove .pt
            img = torch.load(image)
            for name, feature_extractor in fes.items():

                edges, feat, seg, center = feature_extractor.extract(img.clone()[None], return_centers=True)
                for (k, v) in zip(stores, [edges, feat, seg, center]):
                    filename = os.path.join(mission, "features", name, s, key + ".pt")
                    torch.save(v, filename)

            break
