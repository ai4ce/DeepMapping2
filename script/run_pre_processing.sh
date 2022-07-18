#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_10_03_drive_0027_sync
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0027
# subsample rate
VOXEL=1
# Group size
GROUP_SIZE=8
# Mode
MODE=icp

python pre_processing.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL --group_size $GROUP_SIZE --mode $MODE
