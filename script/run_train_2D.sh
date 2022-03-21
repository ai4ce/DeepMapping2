#!/bin/bash

# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=2D_real
# path to dataset
DATA_DIR=/mnt/NAS/home/xinhao/2D_real_partial/run_0
# training epochs
EPOCH=3000
# batch size
BS=32
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=19
# logging interval
LOG=20
# method number to run
MODE=gt

### training from scratch
#python train_2D.py --name $NAME -d $DATA_DIR -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
INIT_POSE=../results/2D/2D_real/pose_est.npy
CUDA_VISIBLE_DEVICES=2 python train_2D.py --name $NAME -d $DATA_DIR -i $INIT_POSE -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG #--mode $MODE