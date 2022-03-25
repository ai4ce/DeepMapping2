#!/bin/bash

CHECKPOINT_DIR='../results/KITTI/KITTI_0005_group/'
VOXEL=1.2
CUDA_VISIBLE_DEVICES=3 python eval_vis_KITTI.py -c $CHECKPOINT_DIR -v $VOXEL
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR