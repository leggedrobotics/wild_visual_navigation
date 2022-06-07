# Self-Supervised Visual Navigation in the Wild

## Useful links (internal)

- [Main tasks (Kanban)](https://github.com/leggedrobotics/wild_visual_navigation/projects/1)
- [Literature review](https://docs.google.com/spreadsheets/d/1rJPC4jVz_Hw7U6YQauh1B3Xpart7-9tC884P5ONtkaU/edit?usp=sharing)

## Overview

| &nbsp; &nbsp; &nbsp; &nbsp; Scripts &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Baseline Method](./scripts/baselines/bayesian_clustering.py ) | Lee, H. et al. Bayesian Clustering: [](shorturl.at/mNUZ5) |
|... | ... |

| &nbsp; &nbsp; &nbsp; &nbsp; Modules &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Feature Extraction](./wild_visual_navigation/feature_extractor) | Input Image; Clustering, Feature Extraction, Descriptors, SLIC, STEGO |
| [Image Projector](./wild_visual_navigation/image_projector) | Input Image, Pose, 3D object; Projection onto Image|
| [Label Generator](./wild_visual_navigation/label_generator) | Define Trajectory Class: Time, Distance, Other Properties, Projection into Image |
| [Traversability Estimator](./wild_visual_navigation/traversability_estimator) | General Utilities |
| [Utils](./wild_visual_navigation/utils) | Visualization Tools |

## Setup
```
pip3 install -e ./wild_visual_navigation
```

### Requirements

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