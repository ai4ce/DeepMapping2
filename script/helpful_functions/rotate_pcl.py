import set_path
import os
import torch
import numpy as np

import scipy.io as sio
from dataset_loader import SimulatedPointCloud
from torch.utils.data import DataLoader
from open3d import PointCloud,Vector3dVector,write_point_cloud

def np_to_pcd(xyz):
    xyz = xyz.reshape(-1,3)
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    return pcd

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ 
    Cited from hxdengBerkeley, Thank you!
    Rotate the point cloud along up direction with certain angle.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, rotated batch of point clouds
    """
    #print("batch_data:"+str(batch_data.size()))
    #print("rotation_angle:"+str(rotation_angle.size()))

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle[k])
        sinval = np.sin(rotation_angle[k])
        rotation_matrix = np.array([[cosval, sinval],
                                    [-sinval, cosval]]) 
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.cpu().numpy(), rotation_matrix)
    rotated_data = torch.tensor(rotated_data, dtype=torch.float32, device=batch_data.device)
    #print("rotated_data::::"+str(rotated_data))
    return rotated_data

data_dir = '../../data/2D/v1_pose0_bk'
rot_data_dir = os.path.join('../../data/2D',"v1_pose0")
if not os.path.exists(rot_data_dir):
    os.makedirs(rot_data_dir)
checkpoint_dir = os.path.join('../../results/2D',"pcl_rotate")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

gt_file = os.path.join(data_dir,'gt_pose.mat')
gt_pose = sio.loadmat(gt_file)
gt_pose = gt_pose['pose']
gt_location = gt_pose[:,:2]
pose_est = torch.tensor(gt_pose, dtype = torch.float).cpu()
ori_est = torch.tensor(gt_pose[:,2], dtype = torch.float).cpu()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print("ori_est:"+str(ori_est))

init_pose = None
dataset = SimulatedPointCloud(data_dir,init_pose)
full_loader = DataLoader(dataset,batch_size=32, shuffle=False)

rot_pcls = []

for index,(obs_batch,valid_pt,pose_batch) in enumerate(full_loader):
    obs_batch = obs_batch.to(device)
    #print("obs_batch:"+str(obs_batch.size()))
    obs_batch_shape = obs_batch.shape[0]
    rot_pcl = rotate_point_cloud_by_angle(obs_batch, ori_est[index*obs_batch_shape:(index+1)*obs_batch_shape])
    rot_pcls.append(rot_pcl.cpu().numpy()) 

rot_pcls = np.concatenate(rot_pcls)
print("rot_pcls:"+str(rot_pcls.shape))

for i in range(rot_pcls.shape[0]):
    xy = rot_pcls[i]
    zero = np.zeros_like(xy)[:,0:1]
    xyz = np.concatenate((xy,zero),axis=-1)
    pcd = np_to_pcd(xyz)
    save_name = '{:09d}.pcd'.format(i)
    save_name = os.path.join(rot_data_dir,save_name)
    write_point_cloud(save_name,pcd)
