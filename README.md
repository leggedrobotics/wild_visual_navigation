# Vision pipeline
This repo is for the online vision pipeline that uses trained physical decoders to output dense prediction of the environments in the vision channel. The pipeline is based on the previous repo from "Fast Traversability Estimation for Wild Visual Navigation".

## Installation
**Attention**: Please follow the installation order exactly as below. Otherwise, you may encounter some errors.
### Install robostack ros first:
https://robostack.github.io/GettingStarted.html

### Install pytorch next:
(Here we use mamba for virtual environment management with python 3.9)
```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
### Install other dependencies:
```bash
pip install -r requirements.txt
```
If you encounter any errors, please follow the error message to install the missing dependencies.

Set you neptune api token , username and project name in the system file `.bashrc`:
```bash
export NEPTUNE_API_TOKEN="your_neptune_api_token"
export NEPTUNE_USERNAME="your_neptune_username"
export NEPTUNE_PROJECT="your_neptune_username/your_neptune_project_name"
```

### Install this repo:
```bash
pip install .
```

### Build this repo with ROS:
```bash
catkin build
source devel/setup.bash
```

## Vision pipeline - offline training
All configs are set in `BaseWVN/config/wvn_config.py`, for all the training/testing, you should pay attention to path-related settings. 

Download segment_anything model checkpoint from [here](https://drive.google.com/file/d/1TU3asknvo1UKdhx0z50ghHDt1C_McKJu/view?usp=drive_link) and speicify the path in the config file.
### Offline Dataset
It is generated from the online rosbag playing. By setting `label_ext_mode: bool=True` you can record the dataset. The corresponding settings and paths are in config file.
```bash
roslaunch wild_visual_navigation_ros play.launch # start rosbag playing
python src/wild_visual_navigation_ros/scripts/Phy_decoder_node.py  # start phy decoders
python src/wild_visual_navigation_ros/scripts/Main_process_node.py # start main process
```
`ctrl+c` to stop/finish the recording.

The default saving path is `~/BaseWVN/results/manager` with the following files:

- `image_buffer.pt`: only store all the camera image tensors of the main nodes

- `train_data.pt`: only store the training data pairs, which are the same for an online training

- `train_nodes.pt`:store all main nodes with all information

After running offline training for the first time, you will get additional files:

- `gt_masks_SAM.pt`: all the automatically generated GT masks from SAM
- `mask_img.pt`: the corresponding color image of the GT masks above
  
You can put the above files into seperate folders, like `~/BaseWVN/results/manager/train/snow`

### Manual correction of GT masks
Beacause the automatically generated GT masks (from SAM or SEEM) are not perfect, we need to manually correct them with segments.ai . 

You can use the `BaseWVN/offline/seg_correction.py` to correct the masks. The detailed usage you can refer to the code.
### Running
For offline training/testing, you can switch the config and run the following command:
```bash
python BaseWVN/offline/offline_training_lightning.py
```

## Vision pipeline - online training

### Running
For different configs, please refer to the code and config file.
```bash
python src/wild_visual_navigation_ros/scripts/Phy_decoder_node.py  # start phy decoders
python src/wild_visual_navigation_ros/scripts/Main_process_node.py # start main process
```

It is also possible to run the commands with speicifying the python version in conda env, when you have to use the system ros:
```bash
/home/path/to/your/python src/wild_visual_navigation_ros/scripts/Phy_decoder_node.py
...
```
