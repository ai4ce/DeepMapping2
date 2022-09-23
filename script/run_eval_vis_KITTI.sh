#!/bin/bash

CHECKPOINT_DIR='../results/NCLT/gt_vis'
VOXEL=3
python eval_vis_KITTI.py -c $CHECKPOINT_DIR -v $VOXEL
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR