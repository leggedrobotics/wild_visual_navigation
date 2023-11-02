from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.general import training_routine

from omegaconf import OmegaConf

if __name__ == "__main__":
    params = OmegaConf.structured(ExperimentParams)
    training_routine(params)
