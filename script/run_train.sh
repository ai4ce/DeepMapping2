#!/bin/bash

# # dadtaset type
DATASET=KITTI
# path to dataset
DATA_DIR=../data/KITTI
# trajectiory file name
TRAJ=0018
# path to init pose
INIT=$DATA_DIR/$TRAJ/prior/init_pose.npy
# path to pairwise pose
PAIRWISE=$DATA_DIR/$TRAJ/prior/pairwise_pose.npy
# experiment name
NAME=KITTI_0018
# training epochs
EPOCH=50
# loss function
LOSS=bce_ch_eu
# number of points sampled from line-of-sight
N=10
# logging interval
LOG=1
# subsample rate
VOXEL=1
# goupr size
G_SIZE=8
# learning rate
LR=0.00005
# chamfer loss weight
ALPHA=0.1
# euclidean loss weight
BETA=0.1

CUDA_VISIBLE_DEVICES=0 python train.py --alpha $ALPHA --beta $BETA --lr $LR --name $NAME -d $DATA_DIR -t ${TRAJ} -i $INIT -p $PAIRWISE -e $EPOCH -l $LOSS -n $N -v $VOXEL --log_interval $LOG  --group_size $G_SIZE --dataset $DATASET