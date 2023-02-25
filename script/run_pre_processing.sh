#!/bin/bash

# path to dataset
DATA_DIR=../data/KITTI
# trajectiory file name
TRAJ=08
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_odom08_multiway
# subsample rate
VOXEL=1
# Group size
GROUP_SIZE=8
# Mode
MODE=icp

python pre_processing_multiway.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL --group_size $GROUP_SIZE --mode $MODE