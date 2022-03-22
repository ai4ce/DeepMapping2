#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_09_26_drive_0001_sync
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0001
# training epochs
EPOCH=3000
# batch size
BS=8
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=35
# logging interval
LOG=10
# check point
# CHECK=../results/AVD/${NAME}/model_best.pth
# network
NETWORK=deepmapping
# INIT=../results/AVD/AVD_Home_011_1_traj5_ehcd/pose_est.npy
# subsample rate
VOXEL=1.5

### training from scratch
# CUDA_VISIBLE_DEVICES=2 python train_KITTI.py --name $NAME -d $DATA_DIR -t ${TRAJ} -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG --network $NETWORK

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
INIT_POSE=../results/KITTI/$NAME/pose_est.npy
CUDA_VISIBLE_DEVICES=2 python train_KITTI.py --name $NAME -d $DATA_DIR -t ${TRAJ} -i $INIT_POSE -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG --network $NETWORK