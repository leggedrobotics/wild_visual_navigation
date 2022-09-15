<img align="left" width="80" height="80" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

# Self-Supervised Visual Navigation in the Wild

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#citation">Citation</a> •
  <a href="#setup">Setup</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#credits">Credits</a>
</p>
 
![Formatting](https://github.com/leggedrobotics/wild_visual_navigation/actions/workflows/formatting.yml/badge.svg)
---

## Useful links (internal)

- [Main tasks (Kanban)](https://github.com/leggedrobotics/wild_visual_navigation/projects/1)
- [Literature review](https://docs.google.com/spreadsheets/d/1rJPC4jVz_Hw7U6YQauh1B3Xpart7-9tC884P5ONtkaU/edit?usp=sharing)

## Overview

| &nbsp; &nbsp; &nbsp; &nbsp; Scripts &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| - | - |
| [Baseline Method](./scripts/baselines/bayesian_clustering.py ) | Lee, H. et al. Bayesian Clustering [[Paper](http://nmail.kaist.ac.kr/paper/auro2016.pdf)] |
|... | ... |

| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Modules &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Input  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; Function &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| - | - |- |
| [Feature Extraction](./wild_visual_navigation/feature_extractor) | Image | SLIC, STEGO |
| [Image Projector](./wild_visual_navigation/image_projector) | Image, Pose, 3D Object | Projection onto Image |
| [Label Generator](./wild_visual_navigation/label_generator) | Robot State | Generates Traversability Label |
| [Traversability Estimator](./wild_visual_navigation/traversability_estimator) | TBD |  Traversability Estimator |
| [Utils](./wild_visual_navigation/utils) | - | General Utilities |


![Overview](./assets/drawings/overview.svg)

## Citation

## Setup

1. Install mamba to quickly install the conda environment.
```shell
conda activate base
conda install -c conda-forge mamba

# Set correct conda settings (should be correct by default)
conda config --set safety_checks enabled
conda config --set channel_priority false
```

2. Clone the repository.
```shell
mkdir ~/git && cd ~/git 
git clone git@github.com:leggedrobotics/wild_visual_navigation.git
```

3. Install the conda environment using mamba.
```shell
# Make sure to be in the base conda environment
cd ~/git/wild_visual_navigation
mamba env create -f environment.yaml 
```

4. Install the wild_visual_navigation package.
```shell
conda activate wvn
cd ~/git
pip3 install -e ./wild_visual_navigation
```

5. (optional) Activate your W&B account for logging.
```shell
conda activate wvn
# Input your login token from W&B
wandb login 
``` 
### Requirements

## Experiments
### Robot Usage / Rosbag play [Online]
Mode to run the pipeline either fully online on the robot or to simply replay rosbags on your system.

- Launch ANYmal Converter:
`rosrun wild_visual_navigation_anymal anymal_msg_converter_node.py`

- Run WVN Node:
`python wild_visual_navigation_ros/scripts/wild_visual_navigation_node.py _mode:=default _not_time:=True`
There exist multiple configurations you can change via the ros-parameters.
Optionally the node offers services to store the created graph/trained network to the disk.

- (optionally) RVIZ:
`roslaunch wild_visual_navigation_ros view.launch`

- (replay only) Run Debayer:
```roslaunch image_proc_cuda_ros image_proc_cuda_node.launch cam0:=false cam1:=false cam2:=false cam3:=false cam4:=true cam5:=false cam6:=false run_gamma_correction:=false run_white_balance:=true run_vignetting_correction:=false run_color_enhancer:=false run_color_calibration:=false run_undistortion:=true run_clahe:=false dump_images:=false needs_rotation_cam4:=true debayer_option:=bayer_gbrg8```

- (replay only) Replay Rosbag:
```rosbag play --clock path_to_mission/*.bag```


### Learning Usage [Offline]
#### Dataset Generation

Sometimes it`s usefull to just analyze the network training therefore we provide the tools to extract a dataset usefull for learning from a given rosbag. 
In the following we explain how you can generate the dataset with the following structure: 
```
dataset_name
  split
    forest_train.txt
    forest_val.txt
    forest_test.txt
  day3
    date_time_mission_0_day_3
      features
        slic_dino
          center
            time_stamp.pt
            ...
          graph
            time_stamp.pt
            ...
          seg
            time_stamp.pt
            ...
        slic_sift
          ...
        ...
      image
        time_stamp.pt
        ...
      supervision_mask
        time_stamp.pt
        ...
```

1. Dataset is configured in **wild_visual_navigation/utils/dataset_info.py**
2. Run extract_images_and_labels.py (can be also done for multiple missions with **extract_all.py**)
   - Will at first merge all bags within the provided folders from dataset_info into bags containing the `_tf.bag` and other useful data `_wvn.bag`.
   - Steps blocking through the bag
   - Currently, this is done by storing a binary mask and images as .pt files (maybe change to png for storage)
3. Validate if images and labels are generated correctly with **validate_extract_images_and_labels.py**
   - This script will remove images if no label is available and the other way around
   - The stopping early times should be quite small within the seconds
4. Create lists with the training and train/val/test images: **create_train_val_test_lists.py**
5. Convert the correct `.pt` files to `.png` such that you can upload them for the test set to segments.ai **convert_test_images_for_labelling.py**
6. Label them online
7. Fetch the results using **download_bitmaps_from_segments_ai.py**
8. Extract the features segments and graph from the image **extract_features_for_dataset**


#### Training the Network
- Training from the final dataset.
`python3 scripts/train_gnn.py --exp=exp_forest.yaml`


## Contribution

### Code formatting
```shell
# for formatting
pip install black
black --line-length 120 .
# for checking lints
pip install flake8
flake8 .
```
Code format is checked on push.

### Testing
Introduction to [pytest](https://github.com/pluralsight/intro-to-pytest).

```shell
pytest
```
Pytest is not checked on push.

# Extract Dataset
```
cd /home/jonfrey/git/wild_visual_navigation/wild_visual_navigation_ros/scripts && python wild_visual_navigation_node.py
```

```
roslaunch image_proc_cuda_ros image_proc_cuda_node.launch cam0:=false cam1:=false cam2:=false cam3:=false cam4:=true cam5:=false cam6:=false run_gamma_correction:=false run_white_balance:=true run_vignetting_correction:=false run_color_enhancer:=false run_color_calibration:=false run_undistortion:=true run_clahe:=false dump_images:=false needs_rotation_cam4:=true debayer_option:=bayer_gbrg8
```


## Credits
