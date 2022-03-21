from tokenize import Double
from matplotlib.pyplot import axis
import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import open3d as o3d

from dataset_loader import AVD
import torch
from torch.utils.data import DataLoader
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint_dir',type=str,required=True,help='path to results folder')
opt = parser.parse_args()
saved_json_file = os.path.join(opt.checkpoint_dir,'opt.json')
train_opt = utils.load_opt_from_json(saved_json_file)
name = train_opt['name']
data_dir = train_opt['data_dir']
subsample_rate = train_opt['subsample_rate']
traj = train_opt['traj']

# load ground truth poses
dataset = AVD(data_dir,traj,subsample_rate)
batch = len(dataset)
gt_location = torch.tensor(dataset.gt[:, [0, 2]], dtype=torch.float)
gt_location = gt_location / dataset.depth_scale
# print(gt_location)
# assert()
gt_direction = np.expand_dims(2*np.arctan(dataset.gt[:, 3]/(dataset.gt[:, 5]+1)), 1)
gt_pose = torch.tensor(np.concatenate((gt_location, gt_direction), axis=1), dtype=torch.float)
print(gt_pose)
# assert()
gt_pcd = dataset.point_clouds.view(batch, -1, 3)
# print(gt_pcd)
obs_global_gt = utils.transform_to_global_AVD(gt_pose, gt_pcd)
# print(obs_global_gt)

n_pc = obs_global_gt.shape[0]
pcds = o3d.PointCloud()

for i in range(n_pc):
    xyz = obs_global_gt[i,:,:]
    current_pcd = utils.np_to_pcd(xyz)
    pcds += current_pcd

o3d.write_point_cloud(os.path.join(opt.checkpoint_dir, "global_gt.pcd"), pcds)
