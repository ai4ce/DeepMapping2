#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_09_30_drive_0018_sync_raw
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0018
# subsample rate
VOXEL=1
# Group size
GROUP_SIZE=4
# Mode
MODE=local

python pre_processing.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL --group_size $GROUP_SIZE --mode $MODE
