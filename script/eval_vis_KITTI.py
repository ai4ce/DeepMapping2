import set_path
import os
import colorsys
import argparse
import functools
import torch
print = functools.partial(print,flush=True)

import numpy as np
import open3d as o3d

from dataset_loader import KITTI
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint_dir',type=str,required=True,help='path to results folder')
parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
opt = parser.parse_args()
saved_json_file = os.path.join(opt.checkpoint_dir,'opt.json')
train_opt = utils.load_opt_from_json(saved_json_file)
name = train_opt['name']
data_dir = train_opt['data_dir']
voxel_size = opt.voxel_size
traj = train_opt['traj']

# load ground truth poses
dataset = KITTI(data_dir,traj,voxel_size)
gt_pose = dataset.gt_pose

# load predicted poses
pred_file = os.path.join(opt.checkpoint_dir,'pose_global_est.npy')
pred_pose = np.load(pred_file)

# print(pred_location.shape)
# print(pred_pose)
# print(gt_pose[:, 3:])
# compute absolute trajectory error (ATE)
trans_ate, rot_ate = utils.compute_ate(pred_pose, gt_pose) 
print('{}, translation ate: {}'.format(name,trans_ate))
print('{}, rotation ate: {}'.format(name,rot_ate))

# color in visulization
colors = []
color_hue = np.linspace(0, 0.8, dataset.n_pc)
for i in range(dataset.n_pc):
    colors.append(colorsys.hsv_to_rgb(color_hue[i], 0.8, 1))
color_palette = np.expand_dims(np.array(colors), 1)
color_palettes = np.repeat(color_palette, repeats=dataset.n_points, axis=1).reshape(-1, 3)

# vis gt
gt_pose_torch = torch.tensor(gt_pose)
pcds = dataset.point_clouds
gt_global = utils.transform_to_global_KITTI(gt_pose_torch, pcds)
np.save(os.path.join(opt.checkpoint_dir,'obs_global_gt.npy'), gt_global)
gt_global = utils.load_obs_global_est(os.path.join(opt.checkpoint_dir,'obs_global_gt.npy'))
gt_global.colors = o3d.utility.Vector3dVector(color_palettes)
o3d.io.write_point_cloud(os.path.join(opt.checkpoint_dir, "gt_global.pcd"), gt_global)

# vis results
est_pose_torch = torch.tensor(pred_pose)
est_global = utils.transform_to_global_KITTI(est_pose_torch, pcds)
global_point_cloud_file = os.path.join(opt.checkpoint_dir,'obs_global_est.npy')
np.save(global_point_cloud_file, est_global)
pcds = utils.load_obs_global_est(global_point_cloud_file)
pcds.colors = o3d.utility.Vector3dVector(color_palettes)
# o3d.draw_geometries([pcds])
o3d.io.write_point_cloud(os.path.join(opt.checkpoint_dir, "global.pcd"), pcds)
