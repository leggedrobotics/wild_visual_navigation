from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation import WVN_ROOT_DIR
from pathlib import Path
import os
import kornia as K
import torch
import argparse


from torch_geometric.data import Data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_folder",
        type=str,
        default="/media/Data/Datasets/Perugia/preprocessing_test/2022-05-12T11:44:56_mission_0_day_3/alphasense/cam4_undist",
        help="Folder containing images.",
    )
    parser.add_argument("--dataset", type=str, default="perugia_forest", help="Dataset name.")
    parser.add_argument("--store", type=bool, default=True, help="Store data")
    
    args = parser.parse_args()
    image_paths = [str(s) for s in Path(args.img_folder).rglob("*.png")]
    image_paths.sort()
    
    if args.store:
        base_dir = os.path.join(WVN_ROOT_DIR, "results", args.dataset)
        os.makedirs(base_dir, exist_ok=True)
        keys = ["adj", "feat", "seg", "img", "center", ""]
        for s in keys:
            os.makedirs(os.path.join(base_dir, s), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = FeatureExtractor(device)
    
    
    


    datas = []
    
    for j, p in enumerate(image_paths[::88]):
        img = K.io.load_image(p, desired_type=K.io.ImageLoadType.RGB8, device=device)
        img = (img.type(torch.float32) / 255)[None]
        adj, feat, seg, center = fe.dino_slic(img.clone(), return_centers=True)
        
        linear_probs, cluster_probs = fe.si.inference(img)
        
        for s in torch.unique(seg):
            print( seg == s )
        
        
        data = Data(x=x, edge_index=edge_index)
        
        edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    
        if args.store:
            for data, key in zip([adj, feat, seg, img, center], keys):
                path = os.path.join(base_dir, key, f"{key}_{j:06d}.pt")
                torch.save(data.cpu(), path)
