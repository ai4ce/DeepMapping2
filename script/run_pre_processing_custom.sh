#!/bin/bash

# path to dataset
DATA_DIR=../data/KITTI
# trajectiory file name
TRAJ=0013
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0013
# subsample rate
VOXEL=1
# Group size
GROUP_SIZE=10
# Mode
MODE=icp
# embeddings
EMBEDDINGS="../custom/tfvpr_6_embeddings.npy"

python pre_processing_custom.py --name $NAME -d $DATA_DIR -t $TRAJ -v $VOXEL --group_size $GROUP_SIZE --mode $MODE --embeddings $EMBEDDINGS