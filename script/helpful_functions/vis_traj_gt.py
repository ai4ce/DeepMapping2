import sys
sys.path.insert(0, '../../')

import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import KITTI
from matplotlib import cm, colors, rc

rc('image', cmap='rainbow_r')
data_dir = '../../data/kitti'
traj = "2011_09_26_drive_0036_sync"
print("loading dataset")
dataset = dataset = KITTI(data_dir, traj, 1)
print("ploting")
location = dataset.gt_pose
t = np.arange(location.shape[0]) / location.shape[0]
location[:, 0] = location[:, 0] - np.mean(location[:, 0])
location[:, 1] = location[:, 1] - np.mean(location[:, 1])
u = np.cos(location[:, 2]) * 2
v = np.sin(location[:, 2]) * 2
# location[:, 1] = -location[:, 1]
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
ax.quiver(location[:, 0], location[:, 1], u, v, t, scale=10, scale_units='inches', width=2e-3)

ax.axis('equal')
ax.tick_params(axis='both', labelsize=18)
norm = colors.Normalize(0, location.shape[0])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'))
cbar.ax.tick_params(labelsize=18)
plt.savefig("traj_vis/kitti_0036_gt.png", dpi=600)