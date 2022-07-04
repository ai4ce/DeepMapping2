#!/bin/bash

# path to dataset
DATA_DIR=../data/kitti
# trajectiory file name
TRAJ=2011_09_30_drive_0018_sync_tfvpr
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=KITTI_0018_ddp
# training epochs
EPOCH=100
# batch size
BS=1
# loss function
LOSS=bce_ch_eu
# number of points sampled from line-of-sight
N=10
# logging interval
LOG=1
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
# chamfer loss weight
ALPHA=0.1
# euclidean loss weight
BETA=0.1

### training from scratch
# CUDA_VISIBLE_DEVICES=1 python train_KITTI.py --name $NAME -d $DATA_DIR -t ${TRAJ} -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG --group $GROUP --group_size $G_SIZE

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
# mkdir /mnt/NAS/home/xinhao/deepmapping/main/results/KITTI/$NAME
# cp /mnt/NAS/home/xinhao/deepmapping/main/results/KITTI/KITTI_0018_pairwise/pose_est_icp.npy /mnt/NAS/home/xinhao/deepmapping/main/results/KITTI/$NAME
INIT_POSE=../results/KITTI/$NAME/pose_est_icp.npy
python train_KITTI_ddp.py --alpha $ALPHA --beta $BETA --pairwise --lr $LR --name $NAME -d $DATA_DIR -t ${TRAJ} -i $INIT_POSE -e $EPOCH -l $LOSS -n $N -v $VOXEL --log_interval $LOG -g $GROUP --group_size $G_SIZE
