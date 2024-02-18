# TODO: Jonas fix stroing to avoid leading empty [1,1] in data
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation import WVN_ROOT_DIR
from pathlib import Path
import os
import torch
import argparse
import cv2

# TODO
# from torch_geometric.data import Data

if __name__ == "__main__":
    """Converts a folder with images to a torch_geometric dataformat.
    Current implemntation does the following:
        1. Extract SLIC Superpixels
        2. Extract DINO features
        3. Extract STEGO linear_probe lables
        4. Convert SLIC Superpixels into graph based on adjacency
        5. Feature of Node is given by mean DINO feature of SLIC segment
        6. Label of Node is given by most often predicted semantic class label by STEGO.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_folder",
        type=str,
        default="/media/Data/Datasets/Perugia/preprocessing_test/2022-05-12T11:44:56_mission_0_day_3/alphasense/cam4_undist",
        help="Folder containing images.",
    )
    parser.add_argument("--dataset", type=str, default="perugia_forest", help="Dataset name.")
    parser.add_argument("--store", type=bool, default=False, help="Store data")
    parser.add_argument("--store-graph", type=bool, default=True, help="Store data")

    args = parser.parse_args()
    image_paths = [str(s) for s in Path(args.img_folder).rglob("*.png")]
    image_paths.sort()

    base_dir = os.path.join(WVN_ROOT_DIR, "results", args.dataset)

    if args.store:
        keys = ["adj", "feat", "seg", "img", "center", ""]
        for s in keys:
            os.makedirs(os.path.join(base_dir, s), exist_ok=True)

    if args.store_graph:
        os.makedirs(os.path.join(base_dir, "graph"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = FeatureExtractor(device, extractor_type="dino_slic")
    stego = FeatureExtractor(device, extractor_type="stego")

    for j, p in enumerate(image_paths[::20]):
        np_img = cv2.imread(p)
        img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
        img = img.permute(2, 0, 1)
        img = (img.type(torch.float32) / 255)[None]
        adj, feat, seg, center = dino.dino_slic(img.clone(), return_centers=True)

        linear_probs, cluster_probs = stego.stego(img)
        stego_label = linear_probs.argmax(dim=1)[0]

        ys = []
        for s in range(seg.max() + 1):
            m = seg == s
            idx, counts = torch.unique(stego_label[m], return_counts=True)
            ys.append(idx[torch.argmax(counts)])

        y = torch.stack(ys)
        graph_data = Data(x=feat, edge_index=adj, y=y)

        if args.store:
            for data, key in zip([adj, feat, seg, img, center], keys):
                path = os.path.join(base_dir, key, f"{key}_{j:06d}.pt")
                torch.save(data.cpu(), path)

        if args.store_graph:
            path = os.path.join(base_dir, "graph", f"graph_{j:06d}.pt")
            torch.save(graph_data, path)

    print(f"Created GNN dataset! Store Individual {args.store}, Store Graph: {args.store_graph}")
    print(f"Output can be found in: {base_dir}!")
