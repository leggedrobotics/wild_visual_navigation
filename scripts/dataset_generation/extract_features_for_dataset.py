from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.visu import LearningVisualizer
import os
from pathlib import Path
import torch

# TODO
# from torch_geometric.data import Data
from wild_visual_navigation.utils import KLTTrackerOpenCV
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    visu = False  # currently not used
    store = True  # storing
    extract_corrospondences = True  # optical flow
    debug = False  # debug mode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mission_names = [name for name in os.listdir("/media/Data/Datasets/2022_Perugia/wvn_output/day3")]

    mission_folders = [os.path.join("/media/Data/Datasets/2022_Perugia/wvn_output/day3/", m) for m in mission_names]

    fes = {}

    # FOR FEATURE ABBLATION
    # fes["slic100_dino448_8"] = FeatureExtractor(
    #     device, "slic", "dino", 448, model_type="vit_small", patch_size=8, slic_num_components=100
    # )
    # fes["slic100_dino448_16"] = FeatureExtractor(
    #     device, "slic", "dino", 488, model_type="vit_small", patch_size=16, slic_num_components=100
    # )
    # fes["slic100_dino224_8"] = FeatureExtractor(
    #     device, "slic", "dino", 224, model_type="vit_small", patch_size=8, slic_num_components=100
    # )
    # fes["slic100_dino224_16"] = FeatureExtractor(
    #     device, "slic", "dino", 224, model_type="vit_small", patch_size=16, slic_num_components=100
    # )
    # fes["slic100_dino112_8"] = FeatureExtractor(
    #     device, "slic", "dino", 112, model_type="vit_small", patch_size=8, slic_num_components=100
    # )
    # fes["slic100_dino112_16"] = FeatureExtractor(
    #     device, "slic", "dino", 112, model_type="vit_small", patch_size=16, slic_num_components=100
    # )
    # fes["slic100_sift"] = FeatureExtractor(device, "slic", "sift", slic_num_components=100)

    # fes["slic100_efficientnet_b0"] = FeatureExtractor(
    #     device, "slic", "torchvision", (256, 224), model_type="efficientnet_b0", slic_num_components=100
    # )
    # fes["slic100_efficientnet_b4"] = FeatureExtractor(
    #     device, "slic", "torchvision", (384, 380), model_type="efficientnet_b4", slic_num_components=100
    # )
    # fes["slic100_efficientnet_b7"] = FeatureExtractor(
    #     device, "slic", "torchvision", (633, 600), model_type="efficientnet_b7", slic_num_components=100
    # )
    # fes["slic100_resnet50"] = FeatureExtractor(
    #     device, "slic", "torchvision", 448, model_type="resnet50", slic_num_components=100
    # )
    # fes["slic100_resnet18"] = FeatureExtractor(
    #     device, "slic", "torchvision", 448, model_type="resnet18", slic_num_components=100
    # )
    # fes["slic100_resnet50_dino"] = FeatureExtractor(
    #     device, "slic", "torchvision", 448, model_type="resnet50_dino", slic_num_components=100
    # )

    fes["slic200_dino112_8"] = FeatureExtractor(
        device,
        "slic",
        "dino",
        112,
        model_type="vit_small",
        patch_size=8,
        slic_num_components=200,
    )
    fes["grid32_dino112_8"] = FeatureExtractor(
        device, "slic", "dino", 112, model_type="vit_small", patch_size=8, cell_size=32
    )
    fes["grid16_dino112_8"] = FeatureExtractor(
        device, "slic", "dino", 112, model_type="vit_small", patch_size=8, cell_size=16
    )
    fes["stego_dino112_8"] = FeatureExtractor(device, "stego", "dino", 112, model_type="vit_small", patch_size=8)

    # read all needed images for training according to the defined split
    perugia_root = "/media/Data/Datasets/2022_Perugia"
    split_files = [str(s) for s in Path("/media/Data/Datasets/2022_Perugia/wvn_output/split").rglob("*.txt")]
    needed_images = []
    for f in split_files:
        with open(f, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                p = line.strip()
                needed_images.append(os.path.join(perugia_root, p))

    form = "{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}"

    with tqdm(
        total=len(needed_images),
        desc="Total",
        colour="green",
        position=1,
        bar_format=form,
    ) as pbar:
        for m_nr, mission in enumerate(mission_folders):
            assert os.path.isdir(
                os.path.join(mission, "image")
            ), f"{mission} is not a valid mission folder misses image"
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

            # read all available images in folder
            images = [str(s) for s in Path(mission, "image").rglob("*.pt")]
            images.sort()

            # note from which images you need to extract only features
            # and note which image you want to store with the optical flow
            extract_with_flow = [False] * len(images)
            extract_for_flow = [False] * len(images)
            for j, img in enumerate(images):
                if img in needed_images:
                    # we cannot extract the first image in a sequence
                    if j != 0 and extract_corrospondences:
                        extract_with_flow[j] = True
                        extract_for_flow[j - 1] = True

                    if not extract_corrospondences:
                        extract_with_flow[j] = True

            images_to_process = np.logical_or(np.array(extract_with_flow), np.array(extract_for_flow))
            images_to_store = np.array(extract_with_flow)

            idx_to_process = np.where(images_to_process)[0]
            idx_to_store = np.where(images_to_store)[0]

            if extract_corrospondences:
                segment_buffer = {}
                feature_position_buffer = {}
                feature_edges_buffer = {}
                feature_buffer = {}
                image_buffer = None

                optical_flow_estimator = KLTTrackerOpenCV(device=device)
                optical_flow_estimator.to(device)

            # iterate over all images (only features and images which have to be stored with the flow)
            for j, idx in enumerate(idx_to_process):
                pbar.update(1)
                store_idx = idx in idx_to_store
                image = images[idx]

                # print(f"Processing {m_nr}/{len(mission_folders)-1} , {j}/{idx_to_process.shape[0]-1}")
                key = image.split("/")[-1][:-3]  # remove .pt
                img = torch.load(image)
                for name, feature_extractor in fes.items():
                    edges, feat, seg, center = feature_extractor.extract(img.clone()[None], return_centers=True)

                    filename = os.path.join(mission, "features", name, "seg", key + ".pt")
                    if store and store_idx:
                        torch.save(seg, filename)

                    filename = os.path.join(mission, "features", name, "center", key + ".pt")
                    if store and store_idx:
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

                    if extract_corrospondences and store_idx:
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
                                p_visu=os.path.join(WVN_ROOT_DIR, "results/extract_features"),
                                store=True,
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
                        data = Data(
                            x=feat,
                            edge_index=edges,
                            y=supervision_signal,
                            y_valid=supervision_signal_valid,
                        )

                    filename = os.path.join(mission, "features", name, "graph", key + ".pt")
                    if store and store_idx:
                        torch.save(data, filename)
