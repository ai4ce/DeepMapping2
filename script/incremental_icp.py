import set_path
import os
import colorsys
import argparse
import functools
# print = functools.partial(print,flush=True)
import torch
import numpy as np

import utils
import open3d as o3d
from dataset_loader import KITTI
from matplotlib import rc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-m','--metric',type=str,default='point',choices=['point','plane'] ,help='minimization metric')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-t','--traj',type=str,default='traj1.txt',help='trajectory file name')
parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
opt = parser.parse_args()
rc('image', cmap='rainbow_r')

checkpoint_dir = os.path.join('../results/KITTI',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)

print('loading dataset')
dataset = KITTI(opt.data_dir, opt.traj, opt.voxel_size, group=False, group_size=2, pairwise=False)
n_pc = len(dataset)

pose_est = np.zeros((n_pc,3),dtype=np.float32)
print('running icp')
for idx in tqdm(range(n_pc-1)):
    dst = dataset[idx].numpy()
    src = dataset[idx+1].numpy()

    _,R0,t0 = utils.icp_o3d(src,dst)
    if idx == 0: 
        R_cum = R0
        t_cum = t0
    else:
        R_cum = np.matmul(R_cum , R0)
        t_cum = np.matmul(R_cum,t0) + t_cum
    
    pose_est[idx+1,:2] = t_cum[:2].T
    pose_est[idx+1,2] = np.arctan2(R_cum[1,0],R_cum[0,0]) 

save_name = os.path.join(checkpoint_dir,'pose_global_est.npy')
np.save(save_name,pose_est)

print('saving results')
pose_est = torch.from_numpy(pose_est)
# local_pc = dataset[:][0].squeeze(1)
# global_pc = utils.transform_to_global_KITTI(pose_est,local_pc)
# utils.plot_global_point_cloud(global_pc,pose_est,valid_id,checkpoint_dir)

# # visulization
colors = []
color_hue = np.linspace(0, 0.8, dataset.n_pc)
for i in range(dataset.n_pc):
    colors.append(colorsys.hsv_to_rgb(color_hue[i], 0.8, 1))
color_palette = np.expand_dims(np.array(colors), 1)
color_palettes = np.repeat(color_palette, repeats=dataset.n_points, axis=1).reshape(-1, 3)

# icp_global = utils.transform_to_global_KITTI(pose_est, dataset.point_clouds)
# np.save(os.path.join(checkpoint_dir,'pose_global_est.npy'), icp_global)
# icp_global = utils.load_obs_global_est(os.path.join(checkpoint_dir,'global_prior.npy'))
# icp_global.colors = o3d.utility.Vector3dVector(color_palettes)
# o3d.io.write_point_cloud(os.path.join(checkpoint_dir, "icp_global.pcd"), icp_global)
utils.plot_global_pose(checkpoint_dir, "prior")

# calculate ate
gt_pose_w_z = utils.add_z_coord_for_evaluation(dataset.gt_pose)
pred_pose_w_z = utils.add_z_coord_for_evaluation(pose_est)
trans_ate, rot_ate = utils.compute_ate(pred_pose_w_z,gt_pose_w_z) 
print('{}, translation ate: {}'.format(opt.name,trans_ate))
print('{}, rotation ate: {}'.format(opt.name,rot_ate))