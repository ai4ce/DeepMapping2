#!/bin/bash

CHECKPOINT_DIR='../results/KITTI/KITTI_0018'
VOXEL=1
python eval_vis_KITTI.py -c $CHECKPOINT_DIR -v $VOXEL
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR