#!/bin/bash

# path to dataset
DATA_DIR=../data/NCLT
# trajectiory file name
TRAJ=2012-01-08-down2
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=NCLT_0108_down2
# training epochs
EPOCH=50
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
G_SIZE=8
# learning rate
LR=0.00005
# chamfer loss weight
ALPHA=0.1
# euclidean loss weight
BETA=0.1

### training from scratch
# CUDA_VISIBLE_DEVICES=1 python train_KITTI.py --name $NAME -d $DATA_DIR -t ${TRAJ} -e $EPOCH -b $BS -l $LOSS -n $N -v $VOXEL --log_interval $LOG --group $GROUP --group_size $G_SIZE

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
# mkdir /mnt/NAS/home/xinhao/deepmapping/main/results/KITTI/$NAME
# cp ../results/KITTI/KITTI_0027_icp/pose_est_icp.npy ../results/KITTI/$NAME
# cp ../results/KITTI/KITTI_0027_icp/pose_pairwise.npy ../results/KITTI/$NAME
INIT_POSE=../results/NCLT/$NAME/pose_est_icp.npy
CUDA_VISIBLE_DEVICES=0 python train_NCLT_amp.py --alpha $ALPHA --beta $BETA --pairwise --lr $LR --name $NAME -d $DATA_DIR -t ${TRAJ} -i $INIT_POSE -e $EPOCH -l $LOSS -n $N -v $VOXEL --log_interval $LOG -g $GROUP --group_size $G_SIZE