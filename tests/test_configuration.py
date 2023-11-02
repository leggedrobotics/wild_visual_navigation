from wild_visual_navigation.cfg import RosLearningNodeParams
from omegaconf import OmegaConf
from omegaconf import read_write
from omegaconf.dictconfig import DictConfig


def test_configuration():
    cfg = OmegaConf.structured(RosLearningNodeParams)
    print(cfg)
    with read_write(cfg):
        cfg.image_callback_rate = 1.0

    print(cfg.image_callback_rate)


if __name__ == "__main__":
    test_configuration()
