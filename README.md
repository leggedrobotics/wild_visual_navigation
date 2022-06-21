<img align="left" width="80" height="80" src="https://github.com/leggedrobotics/wild_visual_navigation/blob/dev/gnn_prototype/assets/images/dino.png" alt="Dino"> 

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

```shell
git clone git@github.com:leggedrobotics/wild_visual_navigation.git
pip3 install -e ./wild_visual_navigation
```

```shell
git clone git@github.com:leggedrobotics/stego.git
pip3 install -e ./stego
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

## Credits
