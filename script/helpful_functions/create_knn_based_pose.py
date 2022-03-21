import os
import torch
import numpy as np

import scipy.io as sio

data_dir = '../../data/2D/v1_pose0'
checkpoint_dir = os.path.join('../../results/2D',"gt_map")

pose_file = os.path.join('../../results/2D/icp_v1_pose0','pose_est.npy')
pose_est = np.load(pose_file)
print("pose_est:::"+str(pose_est.shape))

pose_est = torch.tensor(pose_est, dtype = torch.float).cpu()

# Constraints
K_nearest = 5
contrain_angle = np.pi/6

result = torch.zeros(pose_est.shape[0], K_nearest, dtype = torch.float).cpu()

for i in range(pose_est.shape[0]):
    pose_est_diff = pose_est - pose_est[i]

    v_norm = torch.tensor(np.linalg.norm(pose_est_diff[:,:2], axis=1), dtype = torch.float).cpu()
    (value,index) = torch.topk(v_norm, dim=0, k=K_nearest+1, largest=False)
    result[i] = index[1:]

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
save_name = os.path.join(checkpoint_dir,'bestn_wo_ori_gt.npy')
np.save(save_name,result)

