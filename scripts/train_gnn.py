from simple_parsing import ArgumentParser
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.learning.general import training_routine
from wild_visual_navigation.learning.utils import load_yaml
from wild_visual_navigation import WVN_ROOT_DIR

import os
import dataclasses


def override(dc, exp):
    for k, v in exp.items():
        if hasattr(dc, k):
            if dataclasses.is_dataclass(getattr(dc, k)):
                setattr(dc, k, override(getattr(dc, k), v))
            else:
                setattr(dc, k, v)
    return dc


if __name__ == "__main__":
    """
    WARNING: the command line arguments are overriden based on the enviornment file! Not the inutive other way around
    """

    parser = ArgumentParser()
    parser.add_arguments(ExperimentParams, dest="experiment")

    parser.add_argument("--exp", type=str, default="nan", help="Overwrite params")
    args = parser.parse_args()

    p = os.path.join(WVN_ROOT_DIR, "cfg/exp", args.exp)
    if args.exp != "nan" and os.path.isfile(p):
        exp_override = load_yaml(p)
        args.experiment = override(args.experiment, exp_override)

    training_routine(args.experiment)
