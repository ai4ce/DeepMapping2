#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_09_30_drive_0018_sync_full
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0018
# subsample rate
VOXEL=1
# Error metrics for ICP
# point: "point2point"
# plane: "point2plane"
METRIC=point

python incremental_icp.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL -m $METRIC 
