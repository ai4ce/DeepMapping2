import set_path
import os
import colorsys
import argparse
import functools
import torch

import numpy as np
import open3d as o3d

from dataset_loader import Kitti
from tqdm import tqdm
import utils
import pickle


data_dir = "../data/kitti"
traj = "2011_10_03_drive_0027_sync"
dataset_dir = os.path.join(data_dir, traj)
checkpoint_dir = '../results/KITTI/gt_vis'
voxel_size = 1
pcd_files = sorted(os.listdir(os.path.join(data_dir, traj)))
while pcd_files[-1][-3:] != "pcd":
    pcd_files.pop()


# load ground truth poses
# dataset = Kitti(data_dir, traj, voxel_size)
gt_pose = np.load(os.path.join(data_dir, traj, "gt_pose.npy"))
radius = 6378137 # earth radius
gt_pose[:, :2] *= np.pi / 180
lat_0 = gt_pose[0, 0]
gt_pose[:, 1] *= radius * np.cos(lat_0)
gt_pose[:, 0] *= radius
# gt_pose[:, 1] -= gt_pose[0, 1]
# gt_pose[:, 0] -= gt_pose[0, 0]
gt_pose[:, [0, 1]] = gt_pose[:, [1, 0]]

# color in visulization
colors = []
color_hue = np.linspace(0, 0.8, gt_pose.shape[0])
for i in range(gt_pose.shape[0]):
    colors.append(colorsys.hsv_to_rgb(color_hue[i], 0.8, 1))
color_palette = np.expand_dims(np.array(colors), 1)

gt_global_list = [None] * gt_pose.shape[0]
# gt_global = o3d.geometry.PointCloud()
for i in tqdm(range(gt_pose.shape[0])):
    pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[i]))
    pcd = pcd.voxel_down_sample(voxel_size)
    # rotation = o3d.geometry.get_rotation_matrix_from_xyz(gt_pose[i:i+1, 3:].T)
    # pcd.rotate(rotation)
    # pcd.translate(gt_pose[i, :3], relative=False)
    T = np.eye(4)
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(gt_pose[i, 3:])
    T[:3, 3] = gt_pose[i, :3]
    pcd.transform(T)
    pcd.paint_uniform_color(color_palette[i].T)
    # gt_global  = gt_global + pcd
    gt_global_list[i] = np.asarray(pcd.points)

# vis gt
gt_global_np = np.concatenate(gt_global_list)
gt_global_color = np.zeros_like(gt_global_np)
count = 0
for i in range(len(gt_global_list)):
    gt_global_color[count:count+gt_global_list[i].shape[0], :] = color_palette[i]
    count += gt_global_list[i].shape[0]
gt_global = o3d.geometry.PointCloud()
gt_global.points = o3d.utility.Vector3dVector(gt_global_np)
gt_global.colors = o3d.utility.Vector3dVector(gt_global_color)
o3d.io.write_point_cloud(os.path.join(checkpoint_dir, traj+".pcd"), gt_global)
# with open(os.path.join(checkpoint_dir, traj+".pcd"), "wb")  as f:   
#     pickle.dump(gt_global, f)
