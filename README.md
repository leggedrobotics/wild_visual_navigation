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

Checkout out also our other works.

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 


## Installation

1. Clone the WVN and our STEGO reimplementation.
```shell
mkdir ~/git && cd ~/git 
git clone git@github.com:leggedrobotics/wild_visual_navigation.git
git clone git@github.com:leggedrobotics/self_supervised_segmentation.git
```

2. Install the virtual environment.
```shell
cd ~/git/wild_visual_navigation
# TODO
```

3. Install the wild_visual_navigation package.
```shell
cd ~/git
pip3 install -e ./wild_visual_navigation
```

4. [Optionally] Configure custom paths 
Set your custom global paths by defining the ENV_WORKSTATION_NAME and exporting the variable in your `~/.bashrc`.
  
  ```shell
  export ENV_WORKSTATION_NAME=your_workstation_name
  ```  
The paths can be specified by modifying `wild_visual_navigation/wild_visual_navigation/cfg/gloabl_params.py`. 
Add your desired global paths. 
Per default, all results are stored in `wild_visual_navigation/results`.

<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Overview



<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Overview
![Overview](./assets/drawings/software_overview.jpg)
What we provide:
- Learning and Feature Extraction Nodes integrated in ROS1
- Gazebo Test Simulation Envrionment
- Example ROSbags
- Pre-trained models with minimalistic inference script (can be used as a easy baseline)
- Integration into elevation_mapping_cupy


<img align="right" width="40" height="40" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/main/assets/images/dino.png" alt="Dino"> 

## Experiments
### [Online] Ros-Mode
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
git clone git@github.com:ANYbotics/anymal_c_simple_description.git
git clone git@github.com:IFL-CAMP/tf_bag.git
git clone git@github.com:ori-drs/procman_ros.git

# Symlink WVN
ln -s ~/git/wild_visual_navigation ~/catkin_ws/src

# Dependencies
rosdep install -ryi --from-paths . --ignore-src

# Build
cd ~/catkin_ws
catkin build anymal_c_simple_description
catkin build tf_bag
catkin build procman_ros
catkin build wild_visual_navigation_ros

# Source
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

After successfully building the ros workspace you can run the full pipeline by either using the launch file (this requires all packages to be installed into your system python installation), or by running the nodes from the virtual environment as plain python scripts.

- Run WVN Nodes:
```shell
python wild_visual_navigation_ros/scripts/wvn_feature_extractor_node.py
```
```shell
python wild_visual_navigation_ros/scripts/wvn_learning_node.py
```

- (optionally) RVIZ:
```shell
roslaunch wild_visual_navigation_ros view.launch
```

- (replay only) Replay Rosbag:
```shell
rosrun  play --clock path_to_mission/*.bag
```



- The general configuration files can be found under: `wild_visual_navigation/cfg/experiment_params.py`
- This configuration is used in the `offline-model-training` and in the `online-ros` mode.
- When running the `online-ros` mode additional configurations for the individual nodes are defined in `wild_visual_navigation/cfg/ros_params.py`.
- These configuration file is filled based on the rosparameter-server during runtime.
- The default values for this configuration can be found under `wild_visual_navigation/wild_visual_navigation_ros/config/wild_visual_navigation`.
- We set an environment variable to automatically load the correct global paths and trigger some special behavior e.g. when training on a cluster.



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