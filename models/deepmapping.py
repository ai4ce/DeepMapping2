from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import LocNetReg2D, LocNetRegAVD, LocNetRegKITTI, MLP
from utils import transform_to_global_2D, transform_to_global_AVD, transform_to_global_KITTI

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


def sample_unoccupied_point(local_point_cloud, n_samples, center):
    """
    sample unoccupied points along rays in local point cloud
    local_point_cloud: <BxLxk>
    n_samples: number of samples on each ray
    center: location of sensor <Bx1xk>
    """
    bs, L, k = local_point_cloud.shape
    # print(center.shape)
    center = center.expand(-1,L,-1) # <BxLxk>
    # print(center.shape)
    unoccupied = torch.zeros(bs, L * n_samples, k,
                             device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        # print(center.shape)
        # print(local_point_cloud.shape)
        unoccupied[:, (idx - 1) * L:idx * L, :] = center + (local_point_cloud-center) * fac
    return unoccupied

class DeepMapping2D(nn.Module):
    def __init__(self, loss_fn, n_obs=256, n_samples=19, dim=[2, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping2D, self).__init__()
        self.n_obs = n_obs
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetReg2D(n_points=n_obs, out_dims=3)
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points,sensor_pose):
        # obs_local: <BxLx2>
        # sensor_pose: init pose <Bx1x3>
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points

        self.pose_est = self.loc_net(self.obs_local)

        self.obs_global_est = transform_to_global_2D(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose[:,:,:2]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_2D(
                self.pose_est, self.unoccupied_local)

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
                                self.valid_points, bce_weight, seq=4, gamma=0.1)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss



class DeepMapping_AVD(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, loss_fn, n_samples=35, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping_AVD, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetRegAVD(out_dims=3) # <x,z,theta> y=0
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points,sensor_pose):
        # obs_local: <BxHxWx3> 
        # valid_points: <BxHxW>
        
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points
        self.pose_est = self.loc_net(self.obs_local)

        bs = obs_local.shape[0]
        self.obs_local = self.obs_local.view(bs,-1,3)
        self.valid_points = self.valid_points.view(bs,-1)
        
        self.obs_global_est = transform_to_global_AVD(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose[:,:,:3]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_AVD(
                self.pose_est, self.unoccupied_local)

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
                                self.valid_points, bce_weight, seq=2, gamma=0.9)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss

class DeepMapping_AVD_ehcd(nn.Module):
    '''
    Enhanced deepmapping where one L-net is for rotation and one
    L-net is for translation
    '''
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, loss_fn, n_samples=35, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping_AVD_ehcd, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.rot_net = LocNetRegAVD(out_dims=1) # theta
        self.trans_net = LocNetRegAVD(out_dims=2) # <x, z> y=0
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points,sensor_pose):
        # obs_local: <BxHxWx3> 
        # valid_points: <BxHxW>
        
        bs = obs_local.shape[0]
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points
        self.rot_est = self.rot_net(self.obs_local)
        self.rotated_local = transform_to_global_AVD(
            torch.cat((torch.zeros(bs, 2).cuda(), self.rot_est), dim=1), 
            self.obs_local
            )
        self.trans_est = self.trans_net(self.rotated_local)
        self.pose_est = torch.cat((self.trans_est, self.rot_est), dim=1)

        self.obs_local = self.obs_local.view(bs,-1,3)
        self.valid_points = self.valid_points.view(bs,-1)
        
        self.obs_global_est = transform_to_global_AVD(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose[:,:,:3]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples, sensor_center)
            self.unoccupied_global = transform_to_global_AVD(
                self.pose_est, self.unoccupied_local)

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
                                self.valid_points, bce_weight, seq=2, gamma=0.9)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss

class DeepMapping_AVD_clrg(nn.Module):
    '''
    Enhanced deepmapping where one L-net is for rotation and one
    L-net is for translation
    '''
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, loss_fn, n_samples=35, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping_AVD_ehcd, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.rot_net = LocNetRegAVD(out_dims=359) # probability of theta lies in each degree
        self.trans_net = LocNetRegAVD(out_dims=3) # <x, z, theta> y=0
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points,sensor_pose):
        # obs_local: <BxHxWx3> 
        # valid_points: <BxHxW>
        
        bs = obs_local.shape[0]
        angles = torch.linspace(0, 359/180*np.pi, 359, device=torch.device('cuda'), requires_grad=True)
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points
        rot_prob = F.softmax(self.rot_net(self.obs_local), dim=1)
        self.rot_est = (rot_prob * angles).sum(dim=1)
        pose_est1 = torch.cat((torch.zeros(bs, 2).cuda(), self.rot_est), dim=1)
        self.rotated_local = transform_to_global_AVD(pose_est1, self.obs_local)
        pose_est2 = self.trans_net(self.rotated_local)
        self.pose_est = pose_est1+pose_est2

        self.obs_local = self.obs_local.view(bs,-1,3)
        self.valid_points = self.valid_points.view(bs,-1)
        
        self.obs_global_est = transform_to_global_AVD(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose[:,:,:3]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_AVD(
                self.pose_est, self.unoccupied_local)

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
                                self.valid_points, bce_weight, seq=2, gamma=0.9)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss


class DeepMapping_KITTI(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, n_points, loss_fn, n_samples=35, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping_KITTI, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.n_points = n_points
        self.loc_net = LocNetRegKITTI(n_points=n_points, out_dims=3) # <x,z,theta> y=0
        self.occup_net = MLP(dim)

    def forward(self, obs_local, valid_points, sensor_pose):
        # obs_local: <BxHxWx3> 
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points
        self.pose_est = self.loc_net(self.obs_local)
        self.bs = obs_local.shape[0]
        self.obs_local = self.obs_local.view(self.bs,-1,3)
        
        self.obs_global_est = transform_to_global_KITTI(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose[:,:,:3]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_KITTI(
                self.pose_est, self.unoccupied_local)

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
                                self.valid_points, bce_weight, seq=2, gamma=0.9)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss
