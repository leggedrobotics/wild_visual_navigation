#!/usr/bin/python3
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from os.path import exists
import argparse
import torch

torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface to debug traversability estimator training via pickle")
    parser.add_argument("--pickle_file", type=str, help="Pickle file path")
    parser.add_argument("--device", type=str, help="device where to load the traversability estimator", default="cuda")
    args = parser.parse_args()

    # Check if file exists
    if not exists(args.pickle_file):
        raise ValueError(f"Argument pickle file [{args.pickle_file}] doesn't exist")

    te = TraversabilityEstimator.load(args.pickle_file, device=args.device)

    # Make training thread
    while True:
        print("New iteration")
        te.train()
