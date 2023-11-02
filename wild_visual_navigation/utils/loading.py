import os
import yaml
from wild_visual_navigation import WVN_ROOT_DIR

__all__ = ["file_path", "load_yaml"]


def file_path(string: str) -> str:
    """Checks if string is a file path

    Args:
        string (str): Potential file path

    Raises:
        NotADirectoryError: String is not a fail path

    Returns:
        (str): Returns the file path
    """

    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_yaml(path: str) -> dict:
    """Loads yaml file

    Args:
        path (str): File path

    Returns:
        (dict): Returns content of file
    """
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    if res is None:
        res = {}
    return res


def load_env() -> dict:
    """Uses ENV_WORKSTATION_NAME variable to load specified environment yaml file.

    Returns:
        (dict): Returns content of environment file
    """
    env_cfg_path = os.path.join(WVN_ROOT_DIR, "cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yaml")
    env = load_yaml(env_cfg_path)
    for k in env.keys():
        if k == "workstation":
            continue
        if not os.path.isabs(env[k]):
            env[k] = os.path.join(WVN_ROOT_DIR, env[k])

    return env


def save_omega_cfg(cfg, path):
    """
    Args:
        cfg (omegaconf): Cfg file
        path (str): File path
    """
    with open(path, "rb") as file:
        OmegaConf.save(config=cfg, f=file)
