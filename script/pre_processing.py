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
from dataset_loader import Kitti
from matplotlib import rc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-m','--metric',type=str,default='point',choices=['point','plane'] ,help='minimization metric')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-t','--traj',type=str,default='traj1.txt',help='trajectory file name')
parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
parser.add_argument('--group_size',type=int,default=4,help='size of group')
parser.add_argument('--mode',type=str,default="icp",help='local or global frame registraion')
opt = parser.parse_args()
rc('image', cmap='rainbow_r')

radius = 6378137 # earth radius

checkpoint_dir = os.path.join('../results/KITTI',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)

print('loading dataset')
# dataset = Kitti(opt.data_dir, opt.traj, opt.voxel_size, group=True, group_size=opt.group_size, pairwise=False)
# n_pc = len(dataset)
dataset_dir = os.path.join(opt.data_dir, opt.traj)
pcd_files = sorted(os.listdir(dataset_dir))
while pcd_files[-1][-3:] != "pcd":
    pcd_files.pop()
n_pc = len(pcd_files)
group_matrix = np.load(os.path.join(dataset_dir, "group_matrix.npy"))[:, :opt.group_size]
pcd_files = np.asarray(pcd_files)

pose_est = np.zeros((n_pc, 6),dtype=np.float32)
print('running icp')

# dataset.group_flag = False
for idx in tqdm(range(n_pc-1)):
    # dst = dataset[idx].numpy()
    # src = dataset[idx+1].numpy()
    dst = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[idx])).voxel_down_sample(opt.voxel_size)
    src = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[idx+1])).voxel_down_sample(opt.voxel_size)

    _, R0, t0 = utils.icp_o3d(src,dst)
    if idx == 0: 
        R_cum = R0
        t_cum = t0
    else:
        R_cum = np.matmul(R_cum, R0)
        t_cum = np.matmul(R_cum, t0) + t_cum
    
    pose_est[idx+1, :3] = t_cum[:3].T
    pose_est[idx+1, 3:] = utils.mat2ang_np(R_cum)

save_name = os.path.join(checkpoint_dir,'pose_est_icp.npy')
np.save(save_name,pose_est)

print('saving results')
utils.plot_global_pose(checkpoint_dir, mode="prior")
# calculate ate
gt_pose = np.load(os.path.join(dataset_dir, "gt_pose.npy"))
gt_pose[:, :2] *= np.pi / 180
lat_0 = gt_pose[0, 0]
gt_pose[:, 1] *= radius * np.cos(lat_0)
gt_pose[:, 0] *= radius
gt_pose[:, 1] -= gt_pose[0, 1]
gt_pose[:, 0] -= gt_pose[0, 0]
# gt_pose = gt_pose[:, [1, 0, 2, 5]]
gt_pose[:, [0, 1]] = gt_pose[:, [1, 0]]
trans_ate, rot_ate = utils.compute_ate(pose_est, gt_pose) 
print('{}, translation ate: {}'.format(opt.name,trans_ate))
print('{}, rotation ate: {}'.format(opt.name,rot_ate))

print("Running pairwise registraion")
pose_est = np.zeros((n_pc, opt.group_size-1, 6),dtype=np.float32)
for idx in tqdm(range(n_pc)):
    for group_idx in range(1, opt.group_size):
        if opt.mode == "icp":
            indices = group_matrix[idx]
            pcd_file_batch = pcd_files[indices]
            dst = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_file_batch[0])).voxel_down_sample(opt.voxel_size)
            src = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_file_batch[group_idx])).voxel_down_sample(opt.voxel_size)

            _, R, t = utils.icp_o3d(src,dst)
            # if idx == 0: 
            #     R_cum = R0
            #     t_cum = t0
            # else:
            #     R_cum = np.matmul(R_cum, R0)
            #     t_cum = np.matmul(R_cum, t0) + t_cum
            # # print(R_cum.shape)
            # # print(t_cum.shape)
            pose_est[idx, group_idx-1, :3] = t[:3].T
            pose_est[idx, group_idx-1, 3:] = utils.mat2ang_np(R)
        elif opt.mode == "gt":
            pose_est[idx, group_idx-1] = gt_pose[group_matrix[idx, 0]] - gt_pose[group_matrix[idx, group_idx]]

save_name = os.path.join(checkpoint_dir,'pose_pairwise.npy')
np.save(save_name,pose_est)