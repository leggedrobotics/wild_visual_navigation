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
