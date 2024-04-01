<h1 style="text-align: center;">Fast Traversability Estimation for Wild Visual Navigation</h1>

<p align="center">
  <a href="#instalation">Installation</a> •
  <a href="#overview">Overview</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#development">Development</a> •
  <a href="#citation">Citation</a>
  
  ![Formatting](https://github.com/leggedrobotics/wild_visual_navigation/actions/workflows/formatting.yml/badge.svg)
</p>

![Overview](./assets/drawings/header.jpg)


<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Installation

### Minimal
Clone the WVN and our STEGO reimplementation.
```shell
mkdir ~/git && cd ~/git 
git clone git@github.com:leggedrobotics/wild_visual_navigation.git
git clone git@github.com:leggedrobotics/self_supervised_segmentation.git
```

(Recommended) Install the virtual environment.
```shell
mkdir ~/.venv
python -m venv ~/venv/wvn
source ~/venv/wvn/bin/activate
```

Install the wild_visual_navigation package.
```shell
cd ~/git
pip3 install -e ./wild_visual_navigation
```

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Overview

### Repository Structure
```
📦wild_visual_navigation  
 ┣ 📂assets
     ┣ 📂demo_data                            # Example images
        ┣ 🖼 example_images.png
        ┗ ....
     ┗ 📂checkpoints                          # Pre-trained model checkpoints
        ┣ 📜 mountain_bike_trail_v2.pt
        ┗ ....
 ┣ 📂docker                                   # Quick start docker container
 ┣ 📂results   
 ┣ 📂test   
 ┣ 📂wild_visual_navigation                   # Core implementation of WVN
 ┣ 📂wild_visual_navigation_anymal            # ROS1 ANYmal helper package 
 ┣ 📂wild_visual_navigation_jackal            # ROS1 Jackal simulation example
 ┣ 📂wild_visual_navigation_msgs              # ROS1 message definitions
 ┣ 📂wild_visual_navigation_ros               # ROS1 nodes for running WVN 
    ┗ 📂scripts                               
       ┗ 📜 wvn_feature_extractor_node.py     
       ┗ 📜 wvn_learning_node.py    
 ┗ 📜 quick_start.py                          # Inferencing demo_data from pre-trained checkpoints
```
### Features
- Learning and Feature Extraction Nodes integrated in ROS1
- Gazebo Test Simulation Environment
- Example ROSbags
- Pre-trained models with minimalistic inference script (can be used as a easy baseline)
- Integration into elevation_mapping_cupy


<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Experiments

### Inference pre-trained model


### Online adaptation [Simulation]


### Online adaptation [Rosbag]

<p align="center">




</p>


| MPI Outdoor | MPI Indoor | Bahnhofstrasse | Bike Trail |
|----------------|------------|-------------|---------------------|
| <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/mpi_outdoor_trav.png" alt="MPI Outdoor">                |     <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/mpi_indoor_trav.png" alt="MPI Indoor">        |   <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/bahnhofstrasse_trav.png" alt="Bahnhofstrasse">           |        <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/mountain_bike_trail_trav.png" alt="Mountain Bike">              |
| <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/demp_data/mpi_outdoor_raw.png" alt="MPI Outdoor">                |     <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/demp_data/mpi_indoor_raw.png" alt="MPI Indoor">        |   <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/demp_data/bahnhofstrasse_raw.png" alt="Bahnhofstrasse">           |        <img align="center" width="120" height="120" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/demp_data/mountain_bike_trail_raw.png" alt="Mountain Bike">              |


#### Setup
Let`s set up a new catkin_ws:
```shell
# Create Workspace
source /opt/ros/noetic/setup.bash
mkdir -r ~/catkin_ws/src && cd ~/catkin_ws/src
catkin init
catkin config --extend /opt/ros/noetic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Clone Repos
git clone git@github.com:ANYbotics/anymal_d_simple_description.git
git clone git@github.com:ori-drs/procman_ros.git

# Symlink WVN
ln -s ~/git/wild_visual_navigation ~/catkin_ws/src

# Dependencies
rosdep install -ryi --from-paths . --ignore-src

# Build
cd ~/catkin_ws
catkin build anymal_d_simple_description
catkin build procman_ros
catkin build wild_visual_navigation_ros

# Source
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

After successfully building the ros workspace you can run the full pipeline by either using the launch file (this requires all packages to be installed into your system python installation), or by running the nodes from the virtual environment as plain python scripts.


- Run wild_visual_navigation
```shell
roslaunch wild_visual_navigation_ros wild_visual_navigation.launch
```

- (ANYmal replay only) Load ANYmal description for RViZ
```shell
roslaunch anymal_d_simple_description load.launch
```

- (ANYmal replay only) Replay Rosbag:
```shell
robag play --clock path_to_mission/*.bag
```

- RVIZ:
```shell
roslaunch wild_visual_navigation_ros view.launch
```


Degugging (sometimes it is desirable to run the nodes seperate):
```shell
python wild_visual_navigation_ros/scripts/wvn_feature_extractor_node.py
```
```shell
python wild_visual_navigation_ros/scripts/wvn_learning_node.py
```


- The general configuration files can be found under: `wild_visual_navigation/cfg/experiment_params.py`
- This configuration is used in the `offline-model-training` and in the `online-ros` mode.
- When running the `online-ros` mode additional configurations for the individual nodes are defined in `wild_visual_navigation/cfg/ros_params.py`.
- These configuration file is filled based on the rosparameter-server during runtime.
- The default values for this configuration can be found under `wild_visual_navigation/wild_visual_navigation_ros/config/wild_visual_navigation`.
- We set an environment variable to automatically load the correct global paths and trigger some special behavior e.g. when training on a cluster.


<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Development

### Install pre-commit
```shell
pip3 install pre-commit
cd wild_visual_navigation && python3 -m pre_commit install
cd wild_visual_navigation && python3 -m pre_commit run
```

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

### Open-Sourcing
Generating headers
```shell
pip3 install adheader

# If your are using zsh otherwise remove \
addheader wild_visual_navigation -t header.txt -p \*.py --sep-len 79 --comment='#' --sep=' '
addheader wild_visual_navigation_ros -t header.txt -p \*.py -sep-len 79 --comment='#' --sep=' '
addheader wild_visual_navigation_anymal -t header.txt -p \*.py --sep-len 79 --comment='#' --sep=' '

addheader wild_visual_navigation_ros -t header.txt -p \*CMakeLists.txt --sep-len 79 --comment='#' --sep=' '
addheader wild_visual_navigation_anymal -t header.txt -p \*.py -p \*CMakeLists.txt --sep-len 79 --comment='#' --sep=' '
```

### Releasing ANYmal data
```shell
rosrun procman_ros sheriff -l ~/git/wild_visual_navigation/wild_visual_navigation_anymal/config/procman/record_rosbags.pmd --start-roscore 
```

```shell
rosbag_play --tf --sem --flp --wvn  mission/*.bag
```

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Citation
```
@INPROCEEDINGS{frey23fast, 
  AUTHOR    = {Jonas, Frey and Matias, Mattamala and Nived, Chebrolu and Cesar, Cadena and Maurice, Fallon and Marco, Hutter}, 
  TITLE     = {\href{https://arxiv.org/abs/2305.08510}{Fast Traversability Estimation for Wild Visual Navigation}}, 
  BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
  YEAR      = {2023}, 
  ADDRESS   = {Daegu, Republic of Korea}, 
  MONTH     = {July}, 
  DOI       = {10.15607/RSS.2023.XIX.054} 
} 
```

If you are also building up on the STEGO integration or using the pre-trained models for a comparision please cite: 
```
@INPROCEEDINGS{mattamala24wild, 
  AUTHOR    = {Mattamala, Matias and Jonas, Frey and Piotr Libera and Chebrolu, Nived and Georg Martius and Cadena, Cesar and Hutter, Marco and Fallon, Maurice}, 
  TITLE     = {{Wild Visual Navigation: Fast Traversability Learning via Pre-Trained Models and Online Self-Supervision}}, 
  BOOKTITLE = {under review for Autonomous Robots}, 
  YEAR      = {2024}
} 
```

If you are using the elevation_mapping integration
```
@INPROCEEDINGS{erni23mem,
  AUTHOR={Erni, Gian and Frey, Jonas and Miki, Takahiro and Mattamala, Matias and Hutter, Marco},
  TITLE={\href{https://arxiv.org/abs/2309.16818}{MEM: Multi-Modal Elevation Mapping for Robotics and Learning}}, 
  BOOKTITLE={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  YEAR={2023},
  PAGES={11011-11018},
  DOI={10.1109/IROS55552.2023.10342108}
}
```


