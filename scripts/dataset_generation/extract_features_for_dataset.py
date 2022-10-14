from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.visu import LearningVisualizer
import os
from pathlib import Path
import torch
from torch_geometric.data import Data
from wild_visual_navigation.utils import KLTTrackerOpenCV

if __name__ == "__main__":
    visu = False  # currently not used
    store = True  # storing
    extract_corrospondences = True  # optical flow
    debug = False  # debug mode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mission_names = [name for name in os.listdir("/media/Data/Datasets/2022_Perugia/wvn_output/day3")]

    mission_folders = [os.path.join("/media/Data/Datasets/2022_Perugia/wvn_output/day3/", m) for m in mission_names]

    fes = {}
    # fes["none_dino"] = FeatureExtractor(device, "none", "dino")
    # fes["none_sift"] = FeatureExtractor(device, "none", "sift")
    # fes["none_histogram"] = FeatureExtractor(device, "none", "histogram")
    # fes["slic200_resnet50"] = FeatureExtractor(device, "slic", "torchvision", model_type="resnet50", slic_num_components=200, input_size=448)
    # fes["slic200_sift"] = FeatureExtractor(device, "slic", "sift", slic_num_components=200)
    # fes["slic200_histogram"] = FeatureExtractor(device, "slic", "histogram", slic_num_components=200)

    fes["slic100_dino448_8"] = FeatureExtractor(
        device, "slic", "dino", 448, model_type="vit_small", patch_size=8, slic_num_components=100
    )
    fes["slic100_dino448_16"] = FeatureExtractor(
        device, "slic", "dino", 488, model_type="vit_small", patch_size=16, slic_num_components=100
    )
    fes["slic100_dino224_8"] = FeatureExtractor(
        device, "slic", "dino", 224, model_type="vit_small", patch_size=8, slic_num_components=100
    )
    fes["slic100_dino224_16"] = FeatureExtractor(
        device, "slic", "dino", 224, model_type="vit_small", patch_size=16, slic_num_components=100
    )
    fes["slic100_dino112_8"] = FeatureExtractor(
        device, "slic", "dino", 112, model_type="vit_small", patch_size=8, slic_num_components=100
    )
    fes["slic100_dino112_16"] = FeatureExtractor(
        device, "slic", "dino", 112, model_type="vit_small", patch_size=16, slic_num_components=100
    )
    fes["slic100_sift"] = FeatureExtractor(device, "slic", "sift", slic_num_components=100)

    # For Torchvision 0.12 this is important !
    fes["slic100_efficientnet_b0"] = FeatureExtractor(
        device, "slic", "torchvision", (256, 224), model_type="efficientnet_b0", slic_num_components=100
    )
    fes["slic100_efficientnet_b4"] = FeatureExtractor(
        device, "slic", "torchvision", (384, 380), model_type="efficientnet_b4", slic_num_components=100
    )
    fes["slic100_efficientnet_b7"] = FeatureExtractor(
        device, "slic", "torchvision", (633, 600), model_type="efficientnet_b7", slic_num_components=100
    )
    fes["slic100_resnet50"] = FeatureExtractor(
        device, "slic", "torchvision", 448, model_type="resnet50", slic_num_components=100
    )
    fes["slic100_resnet18"] = FeatureExtractor(
        device, "slic", "torchvision", 448, model_type="resnet18", slic_num_components=100
    )

    # fes["grid32_dino"] = FeatureExtractor(device, "grid", "dino", cell_size=32, input_size=448)
    # fes["grid32_sift"] = FeatureExtractor(device, "grid", "sift", cell_size=32)
    # fes["grid32_histogram"] = FeatureExtractor(device, "grid", "histogram", cell_size=32)

    # fes["stego_dino"] = FeatureExtractor(device, "stego", "stego")
    # fes["stego_sift"] = FeatureExtractor(device, "stego", "sift")
    # fes["stego_histogram"] = FeatureExtractor(device, "slic", "histogram")

    for m_nr, mission in enumerate(mission_folders):
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

        if extract_corrospondences:
            segment_buffer = {}
            feature_position_buffer = {}
            feature_edges_buffer = {}
            feature_buffer = {}
            image_buffer = None

            optical_flow_estimator = KLTTrackerOpenCV(device=device)
            optical_flow_estimator.to(device)

        for j, image in enumerate(images):
            print(f"Processing {m_nr}/{len(mission_folders)} , {j}/{len(images)}")
            key = image.split("/")[-1][:-3]  # remove .pt
            img = torch.load(image)
            for name, feature_extractor in fes.items():

                edges, feat, seg, center = feature_extractor.extract(img.clone()[None], return_centers=True)

                filename = os.path.join(mission, "features", name, "seg", key + ".pt")
                if store and ((extract_corrospondences and j != 0) or (not extract_corrospondences)):
                    torch.save(seg, filename)

                filename = os.path.join(mission, "features", name, "center", key + ".pt")
                if store and ((extract_corrospondences and j != 0) or (not extract_corrospondences)):
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

                if extract_corrospondences and j != 0:
                    # Extract the corrospondences
                    pre_pos = feature_position_buffer[name]
                    cur_pos = optical_flow_estimator(
                        pre_pos[:, 0].clone(),
                        pre_pos[:, 1].clone(),
                        image_buffer,
                        img,
                    )
                    cur_pos = torch.stack(cur_pos, dim=1)
                    m = (
                        (cur_pos[:, 0] >= 0)
                        * (cur_pos[:, 1] >= 0)
                        * (cur_pos[:, 0] < img.shape[1])
                        * (cur_pos[:, 1] < img.shape[2])
                    )
                    cor_pre = torch.arange(pre_pos.shape[0], device=device)[m]

                    cur_pos = cur_pos[m].type(torch.long)
                    cor_cur = seg[cur_pos[:, 1], cur_pos[:, 0]]
                    correspondence = torch.stack([cor_pre, cor_cur], dim=1)

                    if debug:
                        from wild_visual_navigation import WVN_ROOT_DIR
                        from wild_visual_navigation.visu import LearningVisualizer

                        visu = LearningVisualizer(
                            p_visu=os.path.join(WVN_ROOT_DIR, "results/extract_features"), store=True
                        )
                        visu.plot_sparse_optical_flow(
                            pre_pos,
                            cur_pos,
                            img1=image_buffer,
                            img2=img,
                            tag="flow_img",
                        )
                        visu.plot_correspondence_segment(
                            seg_prev=segment_buffer[name],
                            seg_current=seg,
                            img_prev=image_buffer,
                            img_current=img,
                            center_prev=feature_position_buffer[name],
                            center_current=center,
                            correspondence=correspondence,
                            tag="centers_img",
                        )

                        from sklearn.decomposition import PCA

                        X = feat.cpu().numpy()
                        pca = PCA(n_components=1)
                        pca.fit(X)
                        res = pca.transform(X)
                        res -= res.min()
                        res /= res.max()
                        pca_seg = torch.zeros_like(seg).type(torch.float32)
                        for i in range(seg.max()):
                            pca_seg[seg == i] = float(res[i])
                        visu.plot_detectron_cont(img, pca_seg, tag="input_img", alpha=0)
                        visu.plot_detectron_cont(img, pca_seg, tag="pca_of_features", alpha=1.0)

                    data = Data(
                        x=feat,
                        edge_index=edges,
                        y=supervision_signal,
                        y_valid=supervision_signal_valid,
                        x_previous=feature_buffer[name],
                        edge_index_previous=feature_edges_buffer[name],
                        correspondence=correspondence,
                    )

                if extract_corrospondences:
                    # Update the buffers
                    feature_position_buffer[name] = center
                    segment_buffer[name] = seg
                    image_buffer = img.clone()
                    feature_edges_buffer[name] = edges
                    feature_buffer[name] = feat
                else:
                    data = Data(x=feat, edge_index=edges, y=supervision_signal, y_valid=supervision_signal_valid)

                filename = os.path.join(mission, "features", name, "graph", key + ".pt")
                if store and ((extract_corrospondences and j != 0) or (not extract_corrospondences)):
                    torch.save(data, filename)
