import os
import json

from segments import SegmentsClient
from segments.utils import load_label_bitmap_from_url, load_image_from_url
import torch

key = os.getenv("SEGMENTS_AI_API_KEY")
client = SegmentsClient(key)

env = "forest"
dataset_identifier = f"jonasfrey96/perugia_{env}"
samples = client.get_samples(dataset_identifier)
for sample in samples:
    label = client.get_label(sample.uuid)
    label.attributes.segmentation_bitmap
    res = load_label_bitmap_from_url(label.attributes.segmentation_bitmap.url)
    n = sample.name.replace(".png", ".pt")
    res = torch.from_numpy((res - 1).astype(bool))
    torch.save(res, f"/media/Data/Datasets/2022_Perugia/wvn_output/labeling/{env}/labels/{n}")
