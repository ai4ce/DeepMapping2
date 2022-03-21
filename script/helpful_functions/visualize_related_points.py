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

checkpoint_dir = os.path.join('../../results/2D','v1_pose0')
checkpoint_dir_validate = os.path.join('../../results/2D',"pcl_validate")

if not os.path.exists(checkpoint_dir_validate):
    os.makedirs(checkpoint_dir_validate)

save_name = os.path.join(checkpoint_dir,'best_n_points.npy')
best_n = np.load(save_name)
best_n = torch.tensor(best_n, dtype = torch.int).cpu().numpy()
#print(best_n)

data_dir = '../../data/2D/v1_pose0'

for i in range(best_n.shape[0]):
    for index in best_n[i]:
        if i!= index:
            file_name = "00000"+f"{index:04n}"+".pcd"
            file_name = os.path.join(data_dir,file_name)
            pcd = read_point_cloud(file_name)
            pcd_np = np.asarray(pcd.points)

            #print("pcd_np.shape"+str(pcd_np.shape))
            file_name_save = 'local_pcl_near_'+str(i)+"_"+str(index)
            save_name = os.path.join(checkpoint_dir_validate, file_name_save)

            plt.plot(pcd_np[:, 0], pcd_np[:, 1], '.')
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.savefig(save_name)
            plt.close()
