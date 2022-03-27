import sys
sys.path.insert(0, '../../')

import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import KITTI
from matplotlib import cm, colors

data_dir = '../../data/kitti'
traj = "2011_09_26_drive_0005_sync"
dataset = dataset = KITTI(data_dir, traj, 1)
location = dataset.gt_pose
t = np.arange(location.shape[0]) / location.shape[0]
location[:, 0] = location[:, 0] - np.mean(location[:, 0])
location[:, 1] = location[:, 1] - np.mean(location[:, 1])
# location[:, 1] = -location[:, 1]
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
ax.scatter(location[:, 0], location[:, 1], c=t, s=6, alpha=0.6, cmap='rainbow')

ax.axis('equal')
ax.tick_params(axis='both', labelsize=18)
norm = colors.Normalize(0, location.shape[0])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'))
cbar.ax.tick_params(labelsize=18)
plt.savefig("traj_vis/kitti_0005.png", dpi=600)