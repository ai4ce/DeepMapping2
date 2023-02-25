import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
from tqdm import tqdm

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

    return valid_points

class Nclt(Dataset):
    def __init__(self, root, traj, voxel_size=1, init_pose=None,
                group_size=8, use_tqdm=True, **kwargs):
        self.root = root
        self.traj = traj
        data_folder = os.path.join(root, traj)
        pcd_folder = os.path.join(data_folder, "pcd")
        self.init_pose=init_pose
        self.pairwise_pose = kwargs["pairwise_pose"][:, :group_size-1]

        files = sorted(os.listdir(pcd_folder))
        files.remove('gt_pose.npy')
        point_clouds = []
        max_points = 0
        for file in tqdm(files, disable=not use_tqdm):
            pcd = o3d.io.read_point_cloud(os.path.join(pcd_folder, file))
            # Remove Points that are beyond range
            points = np.asarray(pcd.points)
            pcd = pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
            pcd = pcd.voxel_down_sample(voxel_size)
            pcd = np.asarray(pcd.points)
            point_clouds.append(pcd)
            if max_points < pcd.shape[0] and pcd.shape[0] <= 10000:
                max_points = pcd.shape[0]
        for i in range(len(point_clouds)):
            if point_clouds[i].shape[0] < max_points:
                point_clouds[i] = np.pad(point_clouds[i], ((0, max_points-point_clouds[i].shape[0]), (0, 0)))
            else:
                point_clouds[i] = point_clouds[i][:max_points]

        self.gt_pose = np.load(os.path.join(pcd_folder, 'gt_pose.npy')).astype('float32')
        self.point_clouds = torch.from_numpy(np.stack(point_clouds)).float() # <BxNx3>
        self.n_pc = self.point_clouds.shape[0]
        self.n_points = self.point_clouds.shape[1]
        self.valid_points = find_valid_points(self.point_clouds)
        self.group_matrix = np.load(os.path.join(data_folder, "prior", 'group_matrix.npy')).astype('int')
        if self.group_matrix.shape[1] < group_size:
            print("Warning: matrix size {} is smaller than group size {}, using {}".format(self.group_matrix.shape[1], kwargs['group_size'], self.group_matrix.shape[1]))
        else:
            self.group_matrix = self.group_matrix[:, :group_size]

    def __getitem__(self,index):
        indices = self.group_matrix[index]
        pcd = self.point_clouds[indices, :, :]  # <GxNx3>
        valid_points = self.valid_points[indices,:]  # <GxN>
        init_global_pose = self.init_pose[indices, :] # <Gx4>
        init_global_pose = torch.zeros(self.group_matrix.shape[1], 6)
        pairwise_pose = self.pairwise_pose[index]
        pairwise_pose = torch.tensor(pairwise_pose)
        return pcd, valid_points, init_global_pose, pairwise_pose

    def __len__(self):
        return self.n_pc


class NcltEval(Dataset):
    def __init__(self, train_dataset):
        super().__init__()
        self.point_clouds = train_dataset.point_clouds
        self.valid_points = train_dataset.valid_points
        self.init_pose = train_dataset.init_pose
        self.n_pc = train_dataset.n_pc
        self.n_points = train_dataset.n_points
        self.gt_pose = train_dataset.gt_pose

    def __getitem__(self, index):
        pcd = self.point_clouds[index, :, :] # <Nx3>
        init_pose = self.init_pose[index, :] # <4>
        return pcd, init_pose

    def __len__(self):
        return self.n_pc