#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_09_26_drive_0001_sync
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0001
# subsample rate
VOXEL=1.5
# Error metrics for ICP
# point: "point2point"
# plane: "point2plane"
METRIC=point

python incremental_icp.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL -m $METRIC 
