#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_09_30_drive_0018_sync_full2
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0018_clip_fix
# training epochs
EPOCH=3000
# batch size
BS=32
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=1
# logging interval
LOG=10
# check point
# CHECK=../results/KITTI/${NAME}/model_best.pth
# subsample rate
VOXEL=1
# group
GROUP=1
# goupr size
G_SIZE=4
# learning rate
LR=0.0001

### training from scratch
# CUDA_VISIBLE_DEVICES=1 python train_KITTI.py --name $NAME -d $DATA_DIR -t ${TRAJ} -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG --group $GROUP --group_size $G_SIZE

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
INIT_POSE=../results/KITTI/$NAME/pose_est_icp.npy
CUDA_VISIBLE_DEVICES=0 python train_KITTI.py --lr $LR --name $NAME -d $DATA_DIR -t ${TRAJ} -i $INIT_POSE -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG -g $GROUP --group_size $G_SIZE
