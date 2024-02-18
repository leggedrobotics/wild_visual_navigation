#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR

import os
import yaml


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


def save_omega_cfg(cfg, path):
    """
    Args:
        cfg (omegaconf): Cfg file
        path (str): File path
    """
    with open(path, "rb") as file:
        OmegaConf.save(config=cfg, f=file)
