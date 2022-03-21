#!/bin/bash

CHECKPOINT_DIR='../results/KITTI/KITTI_0000/'
CUDA_VISIBLE_DEVICES=3 python eval_vis_KITTI.py -c $CHECKPOINT_DIR
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR