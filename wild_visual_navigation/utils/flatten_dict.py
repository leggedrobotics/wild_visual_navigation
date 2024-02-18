#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import collections

__all__ = ["flatten_dict"]


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Generates flattened dict

    Args:
        d (dict): Input dict
        parent_key (str, optional): Parent key seperater . Defaults to "".
        sep (str, optional): Seperator for output dict. Defaults to "_".

    Returns:
        (dict): Flattened dict
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                if isinstance(v[0], dict):
                    items.extend(flatten_list(v, new_key, sep=sep))
                    continue
            items.append((new_key, v))
    return dict(items)
