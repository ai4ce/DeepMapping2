import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import open3d as o3d
from tqdm import tqdm

import utils

def find_valid_points(local_point_cloud):
    """
    find valid points in local point cloud
        invalid points have all zeros local coordinates
    local_point_cloud: <BxNxk> 
    valid_points: <BxN> indices  of valid point (0/1)
    """
    eps = 1e-6
    non_zero_coord = torch.abs(local_point_cloud) > eps
    valid_points = torch.sum(non_zero_coord, dim=-1)
    valid_points = valid_points > 0

    # valid_points2 = (local_point_cloud[:, :, 2] > -1.5)
    return valid_points #& valid_points2

class KITTI(Dataset):
    def __init__(self,root,traj,voxel_size=1,trans_by_pose=None,loop_group=False,**kwargs):
        self.radius = 6378137 # earth radius
        self.root = root
        self.traj = traj
        data_folder = os.path.join(root, traj)
        self._trans_by_pose=trans_by_pose
        self._loop_group = loop_group

        files = os.listdir(data_folder)
        files.remove('gt_pose.npy')
        try:
            files.remove('group_matrix.npy')
        except:
            pass
        point_clouds = []
        min_points = 0
        for file in tqdm(files):
            xyz = np.load(os.path.join(data_folder, file))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd = pcd.voxel_down_sample(voxel_size)
            pcd = np.asarray(pcd.points)
            point_clouds.append(pcd)
            if min_points == 0  or min_points > pcd.shape[0]:
                # print(pcd.shape)
                min_points = pcd.shape[0]
        # print(min_points)
        for i in range(len(point_clouds)):
            point_clouds[i] = point_clouds[i][:min_points, :]

        # point_clouds = np.load(os.path.join(data_folder, 'point_cloud.npy')).astype('float32')
        gt_pose = np.load(os.path.join(data_folder, 'gt_pose.npy')).astype('float32')
        gt_pose[:, :2] *= np.pi / 180
        lat_0 = gt_pose[0, 0]
        gt_pose[:, 1] *= self.radius * np.cos(lat_0)
        gt_pose[:, 0] *= self.radius
        gt_pose[:, 1] -= gt_pose[0, 1]
        gt_pose[:, 0] -= gt_pose[0, 0]
        self.point_clouds = torch.from_numpy(np.stack(point_clouds)).float() # <BxNx3>
        self.gt_pose = gt_pose[:, [1, 0, 5]] # <Nx3>
        self.n_pc = self.point_clouds.shape[0]
        self.n_points = self.point_clouds.shape[1]
        self.valid_points = find_valid_points(self.point_clouds)
        if self._loop_group:
            self.group_matrix = np.load(os.path.join(data_folder, 'group_matrix.npy')).astype('int')
            if self.group_matrix.shape[1] < kwargs['group_size']:
                print("Warning: matrix size {} is smaller than group size {}, using {}".format(self.group_matrix.shape[1], kwargs['group_size'], self.group_matrix.shape[1]))
            else:
                self.group_matrix = self.group_matrix[:, :kwargs['group_size']]
        
    def __getitem__(self,index):
        pcd = self.point_clouds[index,:,:]  # <Nx3>
        valid_points = self.valid_points[index,:]  # <N>
        if self._trans_by_pose is not None:
            # pcd = pcd.unsqueeze(0)  # <1XNx3>
            pose = self._trans_by_pose[index, :]
            # pcd = utils.transform_to_global_KITTI(pose, pcd).squeeze(0)
        else:
            pose = torch.zeros(1,3)
        return pcd, valid_points, pose

    def __len__(self):
        return self.n_pc

class GroupSampler(Sampler):
    def __init__(self, group_matrix):
        self.group_matrix = group_matrix.reshape(-1)

    def __iter__(self):
        yield from self.group_matrix

    def __len__(self):
        return self.group_matrix.size
