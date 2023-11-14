from MLP import SimpleMLP
import inspect
import torch


def create_registery():
    """Creates register of avialble classes to instantiate based on global scope.

    Returns:
        register (str: class): Contains all avialble classes from model module
        cfg_keys (str: str): Converts the classnames to lower_case to get correct cfg parameters.
    """

    # Finds all classes available
    register = {k: v for k, v in globals().items() if inspect.isclass(v)}

    # Changes the keys to access the configuration parameters
    # SomeModelNAME -> some_model_name_cfg
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

    return register, cfg_keys

def get_model(model_cfg) -> torch.nn.Module:
    """Returns the instantiated model

    Args:
        model_cfg (dict): Contains "name": (str) ClassName; "class_name_cfg": (Dict).
    Returns:
        model (nn.Module): Some torch module
    """
    name = model_cfg.name
    register, cfg_keys = create_registery()
    cfg_attr = getattr(model_cfg, cfg_keys[name])
    cfg_attr_dict = cfg_attr.to_dict()
    model = register[name](**cfg_attr_dict)

    return model

if __name__ == "__main__":
    from BaseWVN.config.wvn_cfg import ParamCollection
    param=ParamCollection()
    model=get_model(param.model)