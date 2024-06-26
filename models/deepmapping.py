# Copyright (C) 2022-2023, NYU AI4CE Lab. All rights reserved.


import torch
import torch.nn as nn
from .networks import LocNetRegKITTI, MLP
from utils import transform_to_global_KITTI, compose_pose_diff, euler_pose_to_quaternion,quaternion_to_euler_pose, qmul_torch, matrix_to_rotation_6d, rotation_6d_to_matrix, euler_pose_to_6d_pose

def get_M_net_inputs_labels(occupied_points, unoccupited_points):
    """
    get global coord (occupied and unoccupied) and corresponding labels
    """
    n_pos = occupied_points.shape[1]
    inputs = torch.cat((occupied_points, unoccupited_points), 1)
    bs, N, _ = inputs.shape

    gt = torch.zeros([bs, N, 1], device=occupied_points.device)
    gt.requires_grad_(False)
    gt[:, :n_pos, :] = 1
    return inputs, gt


def sample_unoccupied_point(local_point_cloud, n_samples):
    """
    sample unoccupied points along rays in local point cloud
    local_point_cloud: <BxLxk>
    n_samples: number of samples on each ray
    center: location of sensor <Bx1xk>
    """
    bs, L, k = local_point_cloud.shape
    # print(center.shape)
    # center = center.expand(-1,L,-1) # <BxLxk>
    # print(center.shape)
    unoccupied = torch.zeros(bs, L * n_samples, k,
                             device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        # print(center.shape)
        # print(local_point_cloud.shape)
        # unoccupied[:, (idx - 1) * L:idx * L, :] = center + (local_point_cloud-center) * fac
        unoccupied[:, (idx - 1) * L:idx * L, :] = local_point_cloud * fac
    return unoccupied


class DeepMapping2(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, n_points, loss_fn, rotation_representation='quaternion', n_samples=35, dim=[3, 64, 256, 1024, 1024, 256, 64, 1], alpha=0.1, beta=0.1):
        super(DeepMapping2, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.n_points = n_points
        self.rotation = rotation_representation
        if self.rotation == 'quaternion':
            self.loc_net = LocNetRegKITTI(n_points=n_points, out_dims=7) # <x,y,z,theta>
        elif self.rotation == '6d':
            self.loc_net = LocNetRegKITTI(n_points=n_points, out_dims=9)
        else:
            self.loc_net = LocNetRegKITTI(n_points=n_points, out_dims=6) # <x,y,z,theta>
        self.occup_net = MLP(dim)
        self.alpha = alpha
        self.beta = beta
       

    def forward(self, obs_local, sensor_pose, valid_points=None, pairwise_pose=None):
        # obs_local: <GxNx3>
        # sensor_pose: <Gx4>
        G = obs_local.shape[0]
        self.obs_local = obs_local
        if self.rotation == 'quaternion':
            sensor_pose = euler_pose_to_quaternion(sensor_pose)
            # sensor_pose: <Gx7>
        elif self.rotation == '6d':
            sensor_pose = euler_pose_to_6d_pose(sensor_pose)
            # sensor_pose: <Gx9>

        self.obs_initial = transform_to_global_KITTI(sensor_pose, self.obs_local, rotation_representation=self.rotation)
        # obs_initial: <GxNx3>
        self.l_net_out = self.loc_net(self.obs_initial)
        # l_net_out: <Gx9>
        print(self.l_net_out.shape)
        if self.rotation == 'quaternion':
            original_shape = list(sensor_pose.shape)
            xyz = self.l_net_out[:,:3]+ sensor_pose[:,:3]
            wxyz = qmul_torch(self.l_net_out[:,3:], sensor_pose[:,3:])
            self.pose_est =  torch.cat((xyz, wxyz), dim=1).view(original_shape)
        elif self.rotation == 'euler_angle':
            self.pose_est = self.l_net_out + sensor_pose
        elif self.rotation == '6d':
            original_shape = list(sensor_pose.shape)
            xyz = self.l_net_out[:, :3] + sensor_pose[:, :3]
            l_net_6d = rotation_6d_to_matrix(self.l_net_out[:, 3:])
            sensor_6d = rotation_6d_to_matrix(sensor_pose[:, 3:])
            rotation_6d = torch.matmul(l_net_6d, sensor_6d)
            rotation_6d = matrix_to_rotation_6d(rotation_6d)
            self.pose_est = torch.cat((xyz, rotation_6d), dim=1).view(original_shape)
        # l_net_out[:, -1] = 0
        # self.pose_est = cat_pose_KITTI(sensor_pose, self.loc_net(self.obs_initial))
        # self.bs = obs_local.shape[0]
        # self.obs_local = self.obs_local.reshape(self.bs,-1,3)
        self.obs_global_est = transform_to_global_KITTI(self.pose_est, self.obs_local, rotation_representation=self.rotation)
 
        if self.training:
            self.valid_points = valid_points
            if self.rotation == 'quaternion':
                pairwise_pose = euler_pose_to_quaternion(pairwise_pose)
            elif self.rotation == '6d':
                pairwise_pose = euler_pose_to_6d_pose(pairwise_pose)

            if self.loss_fn.__name__ == "pose":
                self.t_src, self.t_dst, self.r_src, self.r_dst = compose_pose_diff(self.pose_est, pairwise_pose, rotation_representation=self.rotation)
            else:
                self.centorid = self.obs_global_est[:1, :, :].expand(G-1, -1, -1)
                relative_centroid_local = self.obs_local[:1, :, :].expand(G-1, -1, -1)
                self.relative_centroid = transform_to_global_KITTI(
                    self.pose_est[1:, :], 
                    transform_to_global_KITTI(pairwise_pose, relative_centroid_local, rotation_representation=self.rotation),
                rotation_representation=self.rotation)
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples)
            self.unoccupied_global = transform_to_global_KITTI(
                self.pose_est, self.unoccupied_local, rotation_representation=self.rotation)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            self.occp_prob = self.occup_net(inputs)
            loss = self.compute_loss()
            return loss

    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=2, gamma=1-self.alpha)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        elif self.loss_fn.__name__ == 'bce_ch_eu':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est, self.relative_centroid, self.centorid,
                                self.valid_points, bce_weight, seq=2, alpha=self.alpha, beta=self.beta)
        elif self.loss_fn.__name__ == 'pose':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est, self.t_src, self.t_dst, self.r_src, self.r_dst,
                                self.valid_points, bce_weight, seq=2, alpha=self.alpha, beta=self.beta)
        return loss
