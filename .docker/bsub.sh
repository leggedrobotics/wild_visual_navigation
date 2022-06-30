#!/bin/bash

# Is executed either from your local workstation or on the cluster.
# Syncs the repostiory to the cluster (only on local workstation) and schedules the job on the cluster.
echo "Execute cluster_bsub.sh"
echo "Args: $@"
h=`echo $@`

if [[ ! $ENV_WORKSTATION_NAME = "euler" ]]
then
    rsync -r -v --exclude=.git/* --exclude=results/* --exclude=.neptune/* /home/$USERNAME/git/wild_visual_navigation/ $USERNAME@euler:/cluster/home/$USERNAME/wild_visual_navigation
    ssh $USERNAME@euler " source /cluster/home/$USERNAME/.bashrc; bsub -n 24 -R singularity -R \"rusage[mem=2596,ngpus_excl_p=1]\" -W $TIME -o $OUTFILE_NAME -R \"select[gpu_mtotal0>=10000]\" -R \"rusage[scratch=1500]\" -R \"select[gpu_driver>=470]\" /cluster/home/$USERNAME/wild_visual_navigation/.docker/cluster_run.sh $h"
else
    TIME=24:00
    bsub -n 24 -R singularity -R "rusage[mem=1596,ngpus_excl_p=1]" -W $TIME -R "select[gpu_mtotal0>=10000]" -R "rusage[scratch=1500]" -R "select[gpu_driver>=470]" /cluster/home/$USERNAME/wild_visual_navigation/.docker/cluster_run.sh $h
fi