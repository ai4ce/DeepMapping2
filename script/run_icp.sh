#!/bin/bash

# path to dataset
DATA_DIR=/mnt/NAS/home/xinhao/2D_real_partial/run_0
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=2D_real
# Error metrics for ICP
# point: "point2point"
# plane: "point2plane"
METRIC=plane

python incremental_icp.py --name $NAME -d $DATA_DIR -m $METRIC 
