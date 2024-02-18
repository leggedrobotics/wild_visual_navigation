#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from dataclasses import dataclass


@dataclass
class GlobalEnvironmentParams:
    perugia_root: str
    results: str


def get_global_env_params(name):
    configs = {
        "default": GlobalEnvironmentParams(perugia_root="TBD", results="results"),
        "ge76": GlobalEnvironmentParams(perugia_root="TBD", results="results"),
        "jetson": GlobalEnvironmentParams(perugia_root="TBD", results="results"),
    }
    return configs[name]
