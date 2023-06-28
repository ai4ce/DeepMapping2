import set_path
import os
import colorsys
import argparse
import functools
# print = functools.partial(print,flush=True)
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

import utils
import open3d as o3d
from matplotlib import rc
from tqdm import tqdm


def plot_traj_with_neighbors(poses, gm):
    fig = plt.figure()
        
    plt.ion()
    for i, neigh in enumerate(gm):
        plt.plot(poses[:, 0], poses[:, 1], c = 'g')
        ln = plt.scatter(poses[neigh, 0], poses[neigh, 1], c = 'r', s = 10, linewidth = 5)

        plt.pause(0.05)

        ln.remove()
        plt.draw()


parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-m','--metric',type=str,default='point',choices=['point','plane'] ,help='minimization metric')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-t','--traj',type=str,default='traj1.txt',help='trajectory file name')
parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
parser.add_argument('--group_size',type=int,default=4,help='size of group')
parser.add_argument('--mode',type=str,default="icp",help='local or global frame registraion')
parser.add_argument('--embeddings',type=str,default="../custom/tfvpr_6_embeddings.npy",help='path to embeddings')
parser.add_argument('-r', '--rotation', type=str, default="euler_angle", help="The rotation representation of pose estimation")
opt = parser.parse_args()
rc('image', cmap='rainbow_r')

dataset = opt.data_dir.split("/")[-1]

checkpoint_dir = os.path.join('../results', dataset, opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir, opt)

if opt.rotation not in ['quaternion','euler_angle']:
    print("Unsupported rotation representation")
    assert()

print('Loading dataset')

prior_dir = os.path.join(opt.data_dir, opt.traj, "prior")
dataset_dir = os.path.join(opt.data_dir, opt.traj, "pcd")
pcd_files = sorted(os.listdir(dataset_dir))

print(f'Found {len(pcd_files)} point clouds')

while pcd_files[-1][-3:] != "pcd":
    pcd_files.pop()
n_pc = len(pcd_files)
pcd_files = np.asarray(pcd_files)

point_clouds = []
for idx in tqdm(range(n_pc)):
    point_clouds.append(o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[idx])))

pose_est = np.zeros((n_pc, 6),dtype=np.float32)
print('Running ICP')

for idx in tqdm(range(n_pc-1)):
    if idx == 0:
        #dst_pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[0]))
        dst_pcd = point_clouds[0]
        
        points = np.asarray(dst_pcd.points)
        dst_pcd = dst_pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
        dst_pcd = dst_pcd.voxel_down_sample(opt.voxel_size)
        dst_pcd.estimate_normals()
        
        #src_pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[1]))
        src_pcd = point_clouds[1]
        
        points = np.asarray(src_pcd.points)
        src_pcd = src_pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
        src_pcd = src_pcd.voxel_down_sample(opt.voxel_size)
        src_pcd.estimate_normals()
    else:
        dst_pcd = src_pcd
        #src_pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[idx+1]))
        src_pcd = point_clouds[idx+1]
        
        points = np.asarray(src_pcd.points)
        src_pcd = src_pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
        src_pcd = src_pcd.voxel_down_sample(opt.voxel_size)
        src_pcd.estimate_normals()
    R0, t0 = utils.icp_o3d(src_pcd, dst_pcd, opt.voxel_size)
    if idx == 0: 
        R_cum = R0
        t_cum = t0
    else:
        R_cum = np.matmul(R_cum, R0)
        t_cum = np.matmul(R_cum, t0) + t_cum
    
    pose_est[idx+1, :3] = t_cum[:3].T
    pose_est[idx+1, 3:] = utils.mat2ang_np(R_cum)

np.save(os.path.join(checkpoint_dir, 'pose_est_icp.npy'), pose_est)
np.save(os.path.join(prior_dir, 'init_pose.npy'), pose_est)

print('Saving results')
utils.plot_global_pose(checkpoint_dir, dataset, mode="prior", rotation_representation = opt.rotation)

## Calculate ATE ##
#pose_est = np.load(os.path.join(checkpoint_dir, "pose_est_icp.npy"))
gt_pose = np.load(os.path.join(dataset_dir, "gt_pose.npy"))

trans_ate, rot_ate = utils.compute_ate(pose_est, gt_pose, rotation_representation = opt.rotation)
print('{}, translation ate: {}'.format(opt.name,trans_ate))
print('{}, rotation ate: {}'.format(opt.name,rot_ate))


## Get 30 Neighbors using KNN ##
print('Generating group matrix')
database = np.load(opt.embeddings)

print(database.shape)

nbrs = NearestNeighbors(n_neighbors=30, algorithm='kd_tree').fit(database)
distance, indice = nbrs.kneighbors(database)

distance = np.array(distance)
indice = np.array(indice)

print(distance.shape)
print(indice.shape)


## Filter above neighbors based on actual distance ##
neighbors = []

for i in range(n_pc):
    factor = 1
    neighbor = [i]
    p1 = pose_est[i][:3]  # get x, y, z of the anchor
    for ind in indice[i]:
        if ind == i:
            continue
        p2 = pose_est[ind][:3]  # get x, y, z of the neighbors predicted using KNN

        dist = math.dist(p1, p2)

        if dist < 10.0:
            neighbor.append(ind)
        
        #print(f'Anchor is {p1}')
        #print(f'Neighbor is {p2}')
        #print(f'Distance is {dist}')
        #print("-----------------------------------------------")

    # append temporal neighbors if length of neighbors is less than group size
    if len(neighbor) < opt.group_size:
        while (len(neighbor) < opt.group_size):
            if (i - factor) not in neighbor:
                if (i - factor) < 0:
                    pass
                else:
                    neighbor.append(i - factor)
            
            if len(neighbor) == opt.group_size:
                break
            
            if (i + factor) not in neighbor:
                if (i + factor) > (n_pc - 1):
                    pass
                else:
                    neighbor.append(i + factor)
            
            factor = factor + 1
    # else restrict to group size
    else:
        neighbor = neighbor[:opt.group_size]
        

    neighbors.append(np.asarray(neighbor))

neighbors = np.asarray(neighbors)
print(neighbors.shape)

np.save(os.path.join(prior_dir, "group_matrix.npy"), neighbors)

vis = True
if vis:
    plot_traj_with_neighbors(gt_pose, neighbors)

print("Running pairwise registration")
#group_matrix = np.load(os.path.join(dataset_dir, "group_matrix.npy"))[:, :opt.group_size]
#group_matrix = np.load("group_matrix.npy")[:, :opt.group_size].astype(int)

group_matrix = neighbors

print(group_matrix)

pose_est = np.zeros((n_pc, opt.group_size-1, 6),dtype=np.float32)
for idx in tqdm(range(n_pc)):
    #src_pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[group_matrix[idx, 0]]))
    src_pcd = point_clouds[group_matrix[idx, 0]]
    
    points = np.asarray(src_pcd.points)
    src_pcd = src_pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
    src_pcd = src_pcd.voxel_down_sample(opt.voxel_size)
    src_pcd.estimate_normals()
    for group_idx in range(1, opt.group_size):
        if opt.mode == "icp":
            #dst_pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[group_matrix[idx, group_idx]]))
            dst_pcd = point_clouds[group_matrix[idx, group_idx]]
            
            points = np.asarray(dst_pcd.points)
            dst_pcd = dst_pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
            dst_pcd = dst_pcd.voxel_down_sample(opt.voxel_size)
            dst_pcd.estimate_normals()
            R, t = utils.icp_o3d(src_pcd, dst_pcd, opt.voxel_size)
            pose_est[idx, group_idx-1, :3] = t[:3].T
            pose_est[idx, group_idx-1, 3:] = utils.mat2ang_np(R)
        elif opt.mode == "gt":
            pose_est[idx, group_idx-1] = gt_pose[group_matrix[idx, 0]] - gt_pose[group_matrix[idx, group_idx]]

np.save(os.path.join(prior_dir, 'pairwise_pose.npy'), pose_est)




