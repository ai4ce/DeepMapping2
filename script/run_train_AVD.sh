#!/bin/bash

# path to dataset
DATA_DIR=../data/ActiveVisionDataset/Home_011_1/
# trajectiory file name
TRAJ=traj4
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=AVD_Home_011_1_${TRAJ}_clrg
# training epochs
EPOCH=3000
# batch size
BS=16
# loss function
LOSS=bce
# number of points sampled from line-of-sight
N=35
# logging interval
LOG=5
# check point
# CHECK=../results/AVD/${NAME}/model_best.pth
# network
NETWORK=enhanced
# INIT=../results/AVD/AVD_Home_011_1_traj5_ehcd/pose_est.npy
# subsample rate
SUBSAMPLE=40

### training from scratch
CUDA_VISIBLE_DEVICES=1 python train_AVD.py --name $NAME -d $DATA_DIR -t ${TRAJ}.txt -e $EPOCH -b $BS -l $LOSS -n $N -s $SUBSAMPLE --log_interval $LOG --network $NETWORK