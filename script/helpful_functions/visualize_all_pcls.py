import set_path
import os
import torch
import numpy as np 
from dataset_loader import SimulatedPointCloud
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

import utils
import scipy.io as sio
from open3d import PointCloud
from open3d import read_point_cloud

checkpoint_dir_validate = os.path.join('../../results/2D',"pcl_gts_local")

if not os.path.exists(checkpoint_dir_validate):
    os.makedirs(checkpoint_dir_validate)

data_dir = '../../data/2D/v1_pose0'

for i in range(2048):
    file_name = "00000"+f"{i:04n}"+".pcd"
    file_name = os.path.join(data_dir,file_name)
    pcd = read_point_cloud(file_name)
    pcd_np = np.asarray(pcd.points)

    #print("pcd_np.shape"+str(pcd_np.shape))
    file_name_save = "local_pcl_"+str(i)
    save_name = os.path.join(checkpoint_dir_validate, file_name_save)

    plt.plot(pcd_np[:, 0], pcd_np[:, 1], '.')
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.savefig(save_name)
    plt.close()
