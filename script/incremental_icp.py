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
opt = parser.parse_args()
rc('image', cmap='rainbow_r')

checkpoint_dir = os.path.join('../results/KITTI',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)

print('loading dataset')
dataset = Kitti(opt.data_dir, opt.traj, opt.voxel_size, group=False, group_size=2, pairwise=False)
n_pc = len(dataset)

pose_est = np.zeros((n_pc, 4),dtype=np.float32)
print('running icp')
for idx in tqdm(range(n_pc-1)):
    dst = dataset[idx].numpy()
    src = dataset[idx+1].numpy()

    _, R0, t0 = utils.icp_o3d(src,dst)
    if idx == 0: 
        R_cum = R0
        t_cum = t0
    else:
        R_cum = np.matmul(R_cum, R0)
        t_cum = np.matmul(R_cum, t0) + t_cum
    
    pose_est[idx+1, :3] = t_cum[:3].T
    pose_est[idx+1, 3] = np.arctan2(R_cum[1,0],R_cum[0,0]) 

save_name = os.path.join(checkpoint_dir,'pose_est_icp.npy')
np.save(save_name,pose_est)

print('saving results')
utils.plot_global_pose(checkpoint_dir, mode="prior")
# calculate ate
print(pose_est.shape)
print(dataset.gt_pose.shape)
trans_ate, rot_ate = utils.compute_ate(pose_est, dataset.gt_pose) 
print('{}, translation ate: {}'.format(opt.name,trans_ate))
print('{}, rotation ate: {}'.format(opt.name,rot_ate))