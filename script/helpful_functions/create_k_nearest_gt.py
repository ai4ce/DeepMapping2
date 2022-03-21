import os
import torch
import numpy as np

import scipy.io as sio

def wrap_angle(angles):
    for i, angle in enumerate(angles):
        while angle > np.pi:
            angle -= 2. * np.pi
        while angle < -np.pi:
            angle += 2. * np.pi
        angles[i] = angle
    return angles

data_dir = '../../data/2D/v1_pose0'
checkpoint_dir = os.path.join('../../results/2D',"gt_map")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

gt_file = os.path.join(data_dir,'gt_pose.mat')
gt_pose = sio.loadmat(gt_file)
gt_pose = gt_pose['pose']
gt_location = gt_pose[:,:2]
pose_est = torch.tensor(gt_pose, dtype = torch.float).cpu()
ori_est = torch.tensor(wrap_angle(gt_pose[:,2]), dtype = torch.float).cpu()

# Constraints
K_nearest = 40
contrain_angle = np.pi/6

result = torch.zeros(pose_est.shape[0], K_nearest, dtype = torch.float).cpu()

angle_contraint = False

for i in range(pose_est.shape[0]):
    pose_est_diff = pose_est - pose_est[i]
    #print(pose_est_diff)

    if angle_contraint:
       v_norm = torch.tensor(np.linalg.norm(pose_est_diff[:,:2], axis=1), dtype = torch.float).cpu() #[1024]
       ori_est_diff = ori_est - ori_est[i]
       for j in range(ori_est_diff.shape[0]):
           if abs(ori_est_diff[j]) > contrain_angle:
               v_norm[j] = 999999
    else:
       v_norm = torch.tensor(np.linalg.norm(pose_est_diff[:,:2], axis=1), dtype = torch.float).cpu()
    (value,index) = torch.topk(v_norm, dim=0, k=K_nearest, largest=False)
    result[i] = index[1:]

if angle_contraint:
    print(result)
    save_name = os.path.join(checkpoint_dir,'bestn_w_ori_gt.npy')
    np.save(save_name,result)
else:
    save_name = os.path.join(checkpoint_dir,'bestn_wo_ori_gt.npy')
    np.save(save_name,result)

