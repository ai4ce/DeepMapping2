#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_09_30_drive_0018_sync_full
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0018_mnet_init
# training epochs
EPOCH=3000
# batch size
BS=16
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=15
# logging interval
LOG=10
# check point
# CHECK=../results/KITTI/${NAME}/model_best.pth
# subsample rate
VOXEL=2
# group
GROUP=1
# goupr size
G_SIZE=8

### training from scratch
# CUDA_VISIBLE_DEVICES=1 python train_KITTI.py --name $NAME -d $DATA_DIR -t ${TRAJ} -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG --group $GROUP --group_size $G_SIZE

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
INIT_POSE=../results/KITTI/$NAME/pose_est_icp.npy
CUDA_VISIBLE_DEVICES=2 python train_KITTI.py --init_mnet --name $NAME -d $DATA_DIR -t ${TRAJ} -i $INIT_POSE -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG -g $GROUP --group_size $G_SIZE
