<h1 style="text-align: center;">Fast Traversability Estimation for Wild Visual Navigation</h1>

<p align="center">
  <a href="#citation">Citation</a> •
  <a href="#setup">Setup</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#credits">Credits</a>
  
  ![Formatting](https://github.com/leggedrobotics/wild_visual_navigation/actions/workflows/formatting.yml/badge.svg)
</p>

![Overview](./assets/drawings/header.jpg)

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Citation

```
@INPROCEEDINGS{FreyMattamala23, 
    AUTHOR    = {Jonas Frey and Matias Mattamala and Nived Chebrolu and Cesar Cadena and Maurice Fallon and Marco Hutter}, 
    TITLE     = {{Fast Traversability Estimation for Wild Visual Navigation}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2023}, 
    ADDRESS   = {Daegu, Republic of Korea}, 
    MONTH     = {June}, 
    DOI       = {TBD} 
} 
```
Checkout out also our other works.

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

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

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Experiments
### Robot Usage [Online]
Mode to run the pipeline either fully online on the robot or to simply replay rosbags on your system.

- Launch ANYmal Converter:
```shell
rosrun wild_visual_navigation_anymal anymal_msg_converter_node.py
```

- Run WVN Node:
```shell
python wild_visual_navigation_ros/scripts/wild_visual_navigation_node.py _mode:=debug
```
There are multiple parameters you can change via the ros-parameter server.
Optionally the node offers services to store the created graph/trained network to the disk.

- (optionally) RVIZ:
```shell
roslaunch wild_visual_navigation_ros view.launch
```

- (replay only) Run Debayer:
```shell
roslaunch image_proc_cuda_ros image_proc_cuda_node.launch cam0:=false cam1:=false cam2:=false cam3:=false cam4:=true cam5:=false cam6:=false run_gamma_correction:=false run_white_balance:=true run_vignetting_correction:=false run_color_enhancer:=false run_color_calibration:=false run_undistortion:=true run_clahe:=false dump_images:=false needs_rotation_cam4:=true debayer_option:=bayer_gbrg8
```

- (replay only) Replay Rosbag:
```shell
rosbag play --clock path_to_mission/*.bag
```

### Replay Usage [Online]
We provide a launch file to start all required nodes for close-loop integration.
```shell
roslaunch wild_visual_navigation_ros replay_launch.launch
```
The launch file allows to toggle the individual modules on and off.
```xml
  <arg name="anymal_converter"  default="True"/>
  <arg name="anymal_rsl_launch" default="True"/>
  <arg name="debayer"           default="True"/>
  <arg name="rviz"              default="True"/>
  <arg name="elevation_mapping" default="True"/>
  <arg name="local_planner"     default="True"/>
```

- Run WVN Node:
```shell
python wild_visual_navigation_ros/scripts/wild_visual_navigation_node.py _mode:=default
```

- Replay Rosbag:
```shell
rosbag play --clock path_to_mission/*.bag
```

### Learning Usage [Offline]
#### Dataset Generation

Sometimes it`s useful to just analyze the network training therefore we provide the tools to extract a dataset useful for learning from a given rosbag. 
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
        slic100_dino224_16
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
We provide scripts for training the network for a single run where a parameter configuration yaml-file can be passed to override the parameters configured within `cfg/experiments_params.py`.
Training from the final dataset.

`python3 scripts/train_gnn.py --exp=exp_forest.yaml`

##### Hyperparameter  
We also provide scripts to use optuna for hyperparameter-searching: 

`python3 scripts/train_optuna.py --exp=exp_forest.yaml`

Within the objective function you can easily adjust the trail parameter suggestions. 

##### Ablations
Finally, we categorize our ablations into `loss`, `network`, `feature`, `time_adaptation` and `knn_evaluation`.

##### `loss`, `network`, and `feature`
For `loss`, `network`, and `feature` we can simply run a training script and pass the correct keyword.
We provide the configurations for those experiments within the `cfg/exp/ablation` folder.
```
python3 scripts/ablation/training_ablation.py --ablation_type=network
```
After running the training the results are stored respectively in `scripts/ablations/<ablation_type>_ablation` as a pickle file. 
For each training run the trained network is evaluate on all testing scenes and the AUROC and ROC values are stored with respect to the hand labeled gt-labels and self-supervised supervision-labels. 
We provide a jupyter notebook to interpret the training results. 
```
python3 scripts/ablation/training_ablation_visu.ipynb
```

##### `time_adaptation`
For the `time_adaptation` run simply run:
```
python3 scripts/ablation/time_adaptation.py
```
and for visualization:
```
python3 scripts/ablation/time_adaptation_visu.py
```
done.

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Contributing

The code on main should be always stable and capable to run on a robot.
The code on develop should be used for development code and then tested on the robot and merged into main. 

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

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Credits
