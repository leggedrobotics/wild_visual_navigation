# Self-Supervised Visual Navigation in the Wild

## Useful links (internal)

- [Main tasks (Kanban)](https://github.com/leggedrobotics/wild_visual_navigation/projects/1)
- [Literature review](https://docs.google.com/spreadsheets/d/1rJPC4jVz_Hw7U6YQauh1B3Xpart7-9tC884P5ONtkaU/edit?usp=sharing)

## Overview

| &nbsp; &nbsp; &nbsp; &nbsp; Scripts &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Data Preprocessing](./scripts/data_preprocessing.py) | Convert ROS Bag to Images |
| [Baseline Method](./scripts/baselines/bayesian_clustering.py ) | Lee, H. et al. Bayesian Clustering: [](shorturl.at/mNUZ5) |
|... | ... |

| &nbsp; &nbsp; &nbsp; &nbsp; Modules &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Data Preprocessing](./wild_visual_navigation/data_preprocessing) | Utilities for converting ROS Bag to Images, Trajectories ...|
| [Clustering](./wild_visual_navigation/clustering) | Clustering, Feature Extraction, Descriptors, SLIC, STEGO ... |
| [Trajectory](./wild_visual_navigation/trajectory) | Define Trajectory Class: Time, Distance, Other Properties, Projection into Image |
| [Utils](./wild_visual_navigation/utils) | General Utilities |
| [Visu](./wild_visual_navigation/visu) | Visualization Tools |

## Setup
```
pip3 install -e ./wild_visual_navigation
```

### Requirements
#### data_preprocessing:
- bagpy

## Usage
TODO

## Contribution
```
# for formatting
pip install black
black --line-length 120 .
# for checking lints
pip install flake8
flake8 .
```