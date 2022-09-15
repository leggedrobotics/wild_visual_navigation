from wild_visual_navigation.utils import perguia_dataset, ROOT_DIR
import os

for n in range(len(perguia_dataset)):
    os.system(
        f"/home/jonfrey/miniconda3/envs/wvn/bin/python3 /home/jonfrey/git/wild_visual_navigation/scripts/dataset_generation/extract_images_and_labels.py --n={n}"
    )
