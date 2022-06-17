#!/bin/bash

CHECKPOINT_DIR='/mnt/NAS/home/cc/Kitti_PCL/KITTI_0930_full_pose/'
VOXEL=1
CUDA_VISIBLE_DEVICES=3 python eval_vis_KITTI.py -c $CHECKPOINT_DIR -v $VOXEL
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR