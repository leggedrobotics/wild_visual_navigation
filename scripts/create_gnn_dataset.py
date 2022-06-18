from wild_visual_navigation.feature_extractor import FeatureExtractor

from pathlib import Path
import kornia as K
import torch

image_paths = [
    str(s)
    for s in Path(
        "/media/Data/Datasets/Perugia/preprocessing_test/2022-05-12T11:44:56_mission_0_day_3/alphasense/cam4_undist"
    ).rglob("*.png")
]


device = "cuda" if torch.cuda.is_available() else "cpu"
fe = FeatureExtractor(device)

for p in image_paths:

    img = K.io.load_image(p, desired_type=K.io.ImageLoadType.RGB8, device=device)
    img = (img.type(torch.float32) / 255)[None]

    adjacency_list, features, seg = fe.dino_slic(img)
