from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from wild_visual_navigation.learning.utils import flatten_dict
import inspect
import os


__all__ = ["get_neptune_logger", "get_wandb_logger", "get_tensorboard_logger"]


def get_neptune_logger(exp, env, exp_p, env_p):
    """Returns NeptuneLogger

    Args:
        exp (dict): Content of environment file
        env (dict): Content of experiment file
        exp_p (str): Path to experiment file
        env_p (str): Path to environment file
    Returns:
        (logger): Logger
    """
    project_name = exp["logger"]["neptune_project_name"]  # Neptune AI project_name "username/project"

    params = flatten_dict(exp)

    name_full = exp["general"]["name"]
    name_short = "__".join(name_full.split("/")[-2:])

    if os.environ["ENV_WORKSTATION_NAME"] == "euler":
        proxies = {"http": "http://proxy.ethz.ch:3128", "https": "http://proxy.ethz.ch:3128"}
        return NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project=project_name,
            name=name_short,
            tags=[os.environ["ENV_WORKSTATION_NAME"], name_full.split("/")[-2], name_full.split("/")[-1]],
            proxies=proxies,
        )

    return NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=project_name,
        name=name_short,
        tags=[os.environ["ENV_WORKSTATION_NAME"], name_full.split("/")[-2], name_full.split("/")[-1]],
    )


def get_wandb_logger(exp, env):
    """Returns NeptuneLogger

    Args:
        exp (dict): Content of environment file

    Returns:
        (logger): Logger
    """
    project_name = exp["logger"]["wandb_project_name"]  # project_name (str): W&B project_name
    save_dir = os.path.join(env["base"], exp["general"]["name"])  # save_dir (str): File path to save directory
    params = flatten_dict(exp)
    name_full = exp["general"]["name"]
    name_short = "__".join(name_full.split("/")[-2:])
    return WandbLogger(
        name=name_short,
        project=project_name,
        entity=exp["logger"]["wandb_entity"],
        save_dir=save_dir,
    )


def get_tensorboard_logger(exp, env):
    """Returns TensorboardLoggers

    Args:
        exp (dict): Content of environment file

    Returns:
        (logger): Logger
    """
    params = flatten_dict(exp)
    return TensorBoardLogger(save_dir=exp["name"], name="tensorboard", default_hp_metric=params)


def get_logger(exp, env):
    name = exp["logger"]["name"]
    save_dir = os.path.join(env["base"], exp["general"]["name"])
    register = {k: v for k, v in globals().items() if inspect.isfunction(v)}
    return register[f"get_{name}_logger"](exp, env)
