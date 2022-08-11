#!/bin/bash

# path to dataset
DATA_DIR=../data/NCLT
# trajectiory file name
TRAJ=2012-01-08
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=NCLT_0108_icp
# subsample rate
VOXEL=0.01
# Group size
GROUP_SIZE=8
# Mode
MODE=icp

python pre_processing.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL --group_size $GROUP_SIZE --mode $MODE
