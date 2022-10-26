#!/bin/bash

# path to dataset
DATA_DIR=../data/KITTI
# trajectiory file name
TRAJ=2011_10_03_drive_0027_sync_tfvpr
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0027_multiway
# subsample rate
VOXEL=0.2
# Group size
GROUP_SIZE=8
# Mode
MODE=icp

python pre_processing_multiway.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL --group_size $GROUP_SIZE --mode $MODE
