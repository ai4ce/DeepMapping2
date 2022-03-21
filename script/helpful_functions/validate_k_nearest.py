import set_path
import os
import torch
import numpy as np 
from dataset_loader import SimulatedPointCloud
from torch.utils.data import DataLoader

import utils
import scipy.io as sio

checkpoint_dir = os.path.join('../../results/2D',"gt_map")
checkpoint_dir_validate = os.path.join('../../results/2D',"gt_map_validate")
if not os.path.exists(checkpoint_dir_validate):
    os.makedirs(checkpoint_dir_validate)

save_name = os.path.join(checkpoint_dir,'bestn_wo_ori_gt.npy')
best_n = np.load(save_name)
best_n = torch.tensor(best_n, dtype = torch.int).cpu().numpy()
#print("best_n:"+str(best_n.size()))

data_dir = '../../data/2D/v1_pose0'
init_pose = None 
gt_file = os.path.join(data_dir,'gt_pose.mat')
gt_pose = sio.loadmat(gt_file)
gt_pose = gt_pose['pose']
gt_location = gt_pose[:,:2]
pose_est = torch.tensor(gt_pose, dtype = torch.float).cpu()
location_est = torch.tensor(gt_location, dtype = torch.float).cpu()
print("pose_est:"+str(pose_est.size()))

utils.draw_graphs(location_est, best_n, 1, checkpoint_dir_validate, downsample_size = 32)

