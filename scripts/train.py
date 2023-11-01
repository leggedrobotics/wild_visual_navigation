from simple_parsing import ArgumentParser
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.general import training_routine
from wild_visual_navigation.utils import load_yaml
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.utils import override_params
import os


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
        args.experiment = override_params(args.experiment, exp_override)

    training_routine(args.experiment)
