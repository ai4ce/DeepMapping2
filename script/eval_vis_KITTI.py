import set_path
import os
import argparse
import functools
import torch
print = functools.partial(print,flush=True)

import numpy as np
import open3d as o3d

from dataset_loader import KITTI
import utils
from utils import open3d_utils

def add_y_coord_for_evaluation(pred_pos_DM):
    """
    pred_pos_DM (predicted position) estimated from DeepMapping only has x and z coordinate
    convert this to <x,y=0,z> for evaluation
    """
    n = pred_pos_DM.shape[0]
    x = pred_pos_DM[:,0]
    y = np.zeros_like(x)
    z = pred_pos_DM[:,1]
    return np.stack((x,y,z),axis=-1)

parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint_dir',type=str,required=True,help='path to results folder')
opt = parser.parse_args()
saved_json_file = os.path.join(opt.checkpoint_dir,'opt.json')
train_opt = utils.load_opt_from_json(saved_json_file)
name = train_opt['name']
data_dir = train_opt['data_dir']
voxel_size = train_opt['voxel_size']
traj = train_opt['traj']

# load ground truth poses
dataset = KITTI(data_dir,traj,voxel_size)
gt_pose = dataset.gt_pose
gt_location = gt_pose[:,:3]

# load predicted poses
pred_file = os.path.join(opt.checkpoint_dir,'pose_est.npy')
pred_pose = np.load(pred_file)
pred_location = pred_pose[:,:2]
pred_location = add_y_coord_for_evaluation(pred_location)

# print(pred_location.shape)
# print(pred_pose)
# print(gt_pose[:, 3:])
# compute absolute trajectory error (ATE)
ate,aligned_location = utils.compute_ate(pred_location,gt_location) 
print('{}, ate: {}'.format(name,ate))

# vis gt
gt_pose_torch = torch.tensor(gt_pose)
pcds = dataset.point_clouds
gt_global = utils.transform_to_global_KITTI(gt_pose_torch, pcds)
np.save(os.path.join(opt.checkpoint_dir,'obs_global_gt.npy'), gt_global)
gt_global = utils.load_obs_global_est(os.path.join(opt.checkpoint_dir,'obs_global_gt.npy'))
o3d.write_point_cloud(os.path.join(opt.checkpoint_dir, "gt_global.pcd"), gt_global)

# vis results
global_point_cloud_file = os.path.join(opt.checkpoint_dir,'obs_global_est.npy')
pcds = utils.load_obs_global_est(global_point_cloud_file)
# o3d.draw_geometries([pcds])
o3d.write_point_cloud(os.path.join(opt.checkpoint_dir, "global.pcd"), pcds)
