#!/bin/bash

# Is executed inside the singularity container.
# Start the training.

echo "Execute cluster_run_container.sh"
h=`echo $@`
echo $h
export ENV_WORKSTATION_NAME=euler

cd /home/git/wvn
exec bash -c "/root/miniconda3/envs/lightning/bin/python3 -u scripts/train.py --headless $h"