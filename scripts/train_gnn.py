from simple_parsing import ArgumentParser
from wild_visual_navigation.learning.utils import ExperimentParams
from wild_visual_navigation.learning.general import training_routine

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentParams, dest="experiment")
    args = parser.parse_args()
    training_routine(args.experiment)
