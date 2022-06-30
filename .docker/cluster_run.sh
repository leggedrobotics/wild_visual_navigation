#!/bin/bash

# Is executed on a node on the cluster, started with the bsub command.

echo "Execute cluster_run.sh"
echo $@

module load gcc/6.3.0 cuda/11.4.2
tar -xf /cluster/work/rsl/$USERNAME/wvn/containers/wvn.tar -C $TMPDIR

singularity exec -B $TMPDIR:/home/tmpdir -B $WORK/wvn:/home/work -B $HOME/wvn:/home/git --nv --writable --containall $TMPDIR/wvn.sif /home/wild_visual_navigation/.docker/cluster_run.sh $@
echo "Execute cluster_run.sh done"
bkill $LSB_JOBID
exit 0