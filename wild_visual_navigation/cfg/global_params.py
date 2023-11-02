from dataclasses import dataclass


@dataclass
class GloabalEnvironmentParams:
    perugia_root: str
    results: str


def get_gloabl_env_params(name):
    configs = {
        "default": GloabalEnvironmentParams(perugia_root="TBD", results="results"),
    }
    return configs[name]
