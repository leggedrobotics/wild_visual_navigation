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
TODO

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
