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

![Overview](./assets/drawings/overview.svg)

## Citation

## Setup

1. Clone the repository.
```shell
mkdir ~/git && cd ~/git 
git clone git@github.com:leggedrobotics/wild_visual_navigation.git
```

2. Install the conda environment.
```shell
# Make sure to be in the base conda environment
cd ~/git/wild_visual_navigation
conda env create -f environment.yaml 
```

3. Install the wild_visual_navigation package.
```shell
conda activate wvn
cd ~/git
pip3 install -e ./wild_visual_navigation
```

4. Configure global paths.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;All global paths are stored within a single yaml-file (`cfg\env\ge76.yaml`) for each machine.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The correct yaml-file for a machine is identified based on the environment variable `ENV_WORKSTATION_NAME`.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can generate a new configuration environment file in the same directory:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example: `cfg/env/your_workstation_name.yaml` 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Content: 

```yaml
# If a relative path is given it is relative the the wild_visual_navigation project directory.
base: results/learning
perugia_root: /media/Data/Datasets/2022_Perugia
```  


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We recommend to directly set the environment variable in your `~/.bashrc` by adding the following:  
  
  ```shell
  export ENV_WORKSTATION_NAME=your_workstation_name
  ```  


## Experiments
### Robot Usage / Rosbag play [Online]
Mode to run the pipeline either fully online on the robot or to simply replay rosbags on your system.

- Launch ANYmal Converter:
```
rosrun wild_visual_navigation_anymal anymal_msg_converter_node.py
```

- Run WVN Node:
```
python wild_visual_navigation_ros/scripts/wild_visual_navigation_node.py _mode:=default _not_time:=True
```
There exist multiple configurations you can change via the ros-parameters.
Optionally the node offers services to store the created graph/trained network to the disk.

- (optionally) RVIZ:
```
roslaunch wild_visual_navigation_ros view.launch
```

- (replay only) Run Debayer:
```
roslaunch image_proc_cuda_ros image_proc_cuda_node.launch cam0:=false cam1:=false cam2:=false cam3:=false cam4:=true cam5:=false cam6:=false run_gamma_correction:=false run_white_balance:=true run_vignetting_correction:=false run_color_enhancer:=false run_color_calibration:=false run_undistortion:=true run_clahe:=false dump_images:=false needs_rotation_cam4:=true debayer_option:=bayer_gbrg8
```

- (replay only) Replay Rosbag:
```
rosbag play --clock path_to_mission/*.bag
```


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
8. Extract the features segments and graph from the image **extract_features_for_dataset.py**


#### Training the Network
##### Training  
We provide scripts for training the network for a single run where a parameter configuration yaml-file can be passed to override the prameters configured within `cfg/experiments_params.py`.
Training from the final dataset.

`python3 scripts/train_gnn.py --exp=exp_forest.yaml`

##### Hyperparameter  
We also provide scripts to use optuna for hyperparameter-searching: 

`python3 scripts/train_optuna.py --exp=exp_forest.yaml`

Within the objective function you can easily adjust the trail parameter suggestions. 

##### Abblations
Finally, our abblations results reported within the paper can be reproduced by running:

`python3 scripts/run_abblation.py`

This will perform multiple training runs of the model on the provided dataset. 
In addition to interpretate the results and create the graphs shown in the paper we provide a Jupyter-Notebook, which loads the results of the runs and creates visualizations

## Contributing

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

## Credits
