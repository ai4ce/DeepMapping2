#!/bin/bash

CHECKPOINT_DIR='/mnt/NAS/home/xinhao/deepmapping/main/results/KITTI/experiment'
VOXEL=0.5
CUDA_VISIBLE_DEVICES=0 python eval_vis_KITTI.py -c $CHECKPOINT_DIR -v $VOXEL
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR