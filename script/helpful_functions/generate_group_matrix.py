import sys
sys.path.insert(0, '../../')

import os
import numpy as np
from sklearn.neighbors import KDTree
from dataset_loader import Kitti

data_dir = '../../data/kitti'
traj = "2011_09_26_drive_0005_sync"
dataset = dataset = Kitti(data_dir, traj, 1)
location = dataset.gt_pose
gt_pose = dataset.gt_pose
tree = KDTree(gt_pose)
_, group_matrix = tree.query(gt_pose, k=64)
print(group_matrix)
np.save(os.path.join(data_dir, traj, "group_matrix.npy"), group_matrix)