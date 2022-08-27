from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.visu import LearningVisualizer
import os
from pathlib import Path
import torch
from torch_geometric.data import Data

if __name__ == "__main__":
    visu = True

    mission_names = [name for name in os.listdir("/media/Data/Datasets/2022_Perugia/wvn_output/day3")]

    mission_folders = [os.path.join("/media/Data/Datasets/2022_Perugia/wvn_output/day3/", m) for m in mission_names]

    fes = {}
    # fes["none_dino"] = FeatureExtractor("cuda", segmentation_type="none", feature_type="dino")
    # fes["none_sift"] = FeatureExtractor("cuda", segmentation_type="none", feature_type="sift")
    # fes["none_histogram"] = FeatureExtractor("cuda", segmentation_type="none", feature_type="histogram")

    fes["slic_dino"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="dino")
    fes["slic_sift"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="sift")
    # fes["slic_histogram"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="histogram")

    # fes["grid_dino"] = FeatureExtractor("cuda", segmentation_type="grid", feature_type="dino")
    # fes["grid_sift"] = FeatureExtractor("cuda", segmentation_type="grid", feature_type="sift")
    # # fes["grid_histogram"] = FeatureExtractor("cuda", segmentation_type="grid", feature_type="histogram")

    # fes["stego_dino"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="dino")
    # fes["stego_sift"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="sift")
    # fes["stego_histogram"] = FeatureExtractor("cuda", segmentation_type="slic", feature_type="histogram")

    for mission in mission_folders:
        assert os.path.isdir(os.path.join(mission, "image")), f"{mission} is not a valid mission folder misses image"
        assert os.path.isdir(
            os.path.join(mission, "supervision_mask")
        ), f"{mission} is not a valid mission folder misses supervision_mask"

        stores = ["seg", "center"]

        visualizers = {}
        for name, _ in fes.items():
            if visu:
                visualizers[name] = LearningVisualizer(os.path.join(mission, "features", name))

            for s in stores:
                os.makedirs(os.path.join(mission, "features", name, s), exist_ok=True)
            os.makedirs(os.path.join(mission, "features", name, "graph"), exist_ok=True)

        images = [str(s) for s in Path(mission, "image").rglob("*.pt")]
        images.sort()

        for j, image in enumerate(images):
            key = image.split("/")[-1][:-3]  # remove .pt
            img = torch.load(image)
            for name, feature_extractor in fes.items():

                edges, feat, seg, center = feature_extractor.extract(img.clone()[None], return_centers=True)
                # for (k, v) in zip(stores, [edges, feat, seg, center]):

                filename = os.path.join(mission, "features", name, "seg", key + ".pt")
                torch.save(seg, filename)

                filename = os.path.join(mission, "features", name, "center", key + ".pt")
                torch.save(center, filename)

                supervision_mask = torch.load(image.replace("image", "supervision_mask"))
                feature_segments = seg

                signal = supervision_mask.type(torch.float32)

                # If we have features, update supervision signal
                labels_per_segment = []
                for s in range(feature_segments.max() + 1):
                    # Get a mask indices for the segment
                    m = feature_segments == s
                    # Add the higehst number per segment
                    # labels_per_segment.append(signal[m].max())
                    labels_per_segment.append(signal[m].mean())

                # Prepare supervision signal
                torch_labels = torch.stack(labels_per_segment)
                # if torch_labels.sum() > 0:
                supervision_signal = torch.nan_to_num(torch_labels, nan=0)
                # Binary mask
                supervision_signal_valid = torch_labels > 0
                data = Data(x=feat, edge_index=edges, y=supervision_signal, y_valid=supervision_signal_valid)
                filename = os.path.join(mission, "features", name, "graph", key + ".pt")
                torch.save(data, filename)
