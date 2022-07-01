from wild_visual_navigation.learning.network import *
import inspect


def create_registery():
    """Creates register of avialble classes to instantiate based on global scope.

    Returns:
        register (str: class): Contains all avialble classes from network module
        cfg_keys (str: str): Converts the classnames to lower_case to get correct cfg parameters.
    """

    # Finds all classes available
    register = {k: v for k, v in globals().items() if inspect.isclass(v)}

    # Changes the keys to access the configuration parameters
    # SomeNetworkNAME -> some_network_name_cfg
    cfg_keys = {}
    for key in register.keys():
        previous_large = False
        cfg_key = []
        for j, k in enumerate(key):
            if k.isupper() and not previous_large and j != 0:
                cfg_key.append("_")
            if k.isupper():
                cfg_key.append(k.lower())
                previous_large = True
            if k.islower():
                cfg_key.append(k)
                previous_large = False

        cfg_key = "".join(cfg_key)
        cfg_keys[key] = cfg_key + "_cfg"

    print("done")
    return register, cfg_keys


def get_network(model_cfg):
    """Returns the instantiated network

    Args:
        model_cfg (dict): Contains "name": (str) ClassName; "class_name_cfg": (Dict).
    Returns:
        model (nn.Module): Some torch module
    """
    name = model_cfg["name"]
    register, cfg_keys = create_registery()
    network = register[name](**model_cfg[cfg_keys[name]])
    return network


if __name__ == "__main__":
    from wild_visual_navigation import WVN_ROOT_DIR
    from wild_visual_navigation.learning.utils import load_yaml
    from os.path import join

    exp = load_yaml(join(WVN_ROOT_DIR, "cfg/exp/exp.yaml"))
    get_network(exp["model"])
