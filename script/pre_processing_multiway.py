import set_path
import os
import pickle
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


def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-m','--metric',type=str,default='point',choices=['point','plane'] ,help='minimization metric')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-t','--traj',type=str,default='traj1.txt',help='trajectory file name')
parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
parser.add_argument('--group_size',type=int,default=4,help='size of group')
parser.add_argument('--mode',type=str,default="icp",help='local or global frame registraion')
opt = parser.parse_args()
rc('image', cmap='rainbow_r')
print("group_size=", opt.group_size)

dataset = opt.data_dir.split("/")[-1]
if opt.data_dir == "/":
    dataset = 'NCLT'
checkpoint_dir = os.path.join('../results', dataset,opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)
# dataset = Kitti(opt.data_dir, opt.traj, opt.voxel_size, group=True, group_size=opt.group_size, pairwise=False)
# n_pc = len(dataset)
dataset_dir = os.path.join(opt.data_dir, opt.traj)
pcd_files = sorted(os.listdir(dataset_dir))
while pcd_files[-1][-3:] != "pcd":
    pcd_files.pop()
n_pc = len(pcd_files)
group_matrix = np.load(os.path.join(dataset_dir, "group_matrix.npy"))[:, :opt.group_size]
pcd_files = np.asarray(pcd_files)
pcds = []
if dataset == "KITTI":
    for i in tqdm(range(len(pcd_files))):
        pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[i])).voxel_down_sample(opt.voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
elif dataset == "NCLT":
    for i in tqdm(range(0, len(pcd_files), 2)):
    # for i in tqdm(range(10000)):
        pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[i]))
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
        pcd = pcd.voxel_down_sample(opt.voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
pose_est = np.zeros((n_pc, 6),dtype=np.float32)
print('running icp')

# # dataset.group_flag = False
pose_graph = o3d.pipelines.registration.PoseGraph()
odometry = np.identity(4)
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
# n_pc=10000
for source_id in tqdm(range(0, n_pc-1, 2)):
    transformation_icp, information_icp = pairwise_registration(
            pcds[source_id // 2], pcds[source_id//2+1], 1, 0.5)
    # transformation_icp, information_icp = np.identity(4), np.identity(6)
    odometry = np.dot(transformation_icp, odometry)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(
            np.linalg.inv(odometry)))
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(source_id // 2,
                                                    source_id // 2+1,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=False))
    for target_id in group_matrix[source_id]:
        if target_id == source_id + 2 or target_id == source_id:
            continue
        if target_id % 2 != 0:
            continue
        transformation_icp, information_icp = pairwise_registration(
            pcds[source_id // 2], pcds[target_id // 2], 1, 0.5)
        # transformation_icp, information_icp = np.identity(4), np.identity(6)
        pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(source_id // 2,
                                                    target_id // 2,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=True))
del pcds
# icp_pose = np.load("/mnt/NAS/home/xinhao/deepmapping/main/results/NCLT/NCLT_0108_icp/pose_est_icp.npy")
# pairwise_pose = np.load("/mnt/NAS/home/xinhao/deepmapping/main/results/NCLT/NCLT_0108_icp/pose_pairwise.npy")
# for source_id in tqdm(range(n_pc-1)):
#     pose_graph.edges.append(
#         o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                     source_id+1,
#                                                     transformation_icp,
#                                                     information_icp,
#                                                     uncertain=False))

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=0.5,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

print('saving results')
for i in tqdm(range(n_pc // 2)):
    transformation = pose_graph.nodes[i].pose.copy()
    R, t = transformation[:3, :3], transformation[:3, 3:]
    pose_est[i, :3] = t[:3].T
    pose_est[i, 3:] = utils.mat2ang_np(R)
save_name = os.path.join(checkpoint_dir,'pose_est_icp.npy')
np.save(save_name,pose_est)

utils.plot_global_pose(checkpoint_dir, dataset, mode="prior")
# calculate ate
gt_pose = np.load(os.path.join(dataset_dir, "gt_pose.npy"))
if dataset == "KITTI": 
    gt_pose[:, :2] *= np.pi / 180
    lat_0 = gt_pose[0, 0]
    radius = 6378137 # earth radius
    gt_pose[:, 1] *= radius * np.cos(lat_0)
    gt_pose[:, 0] *= radius
    gt_pose[:, 1] -= gt_pose[0, 1]
    gt_pose[:, 0] -= gt_pose[0, 0]
    # gt_pose = gt_pose[:, [1, 0, 2, 5]]
    gt_pose[:, [0, 1]] = gt_pose[:, [1, 0]]
trans_ate, rot_ate = utils.compute_ate(pose_est, gt_pose) 
print('{}, translation ate: {}'.format(opt.name,trans_ate))
print('{}, rotation ate: {}'.format(opt.name,rot_ate))

# print("Running pairwise registraion")
# pose_est = np.zeros((n_pc, opt.group_size-1, 6),dtype=np.float32)
# for idx in tqdm(range(n_pc)):
#     src_pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[group_matrix[idx, 0]]))
#     points = np.asarray(src_pcd.points)
#     src_pcd = src_pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
#     src_pcd = src_pcd.voxel_down_sample(opt.voxel_size)
#     src_pcd.estimate_normals()
#     for group_idx in range(1, opt.group_size):
#         if opt.mode == "icp":
#             # src = pcds[group_matrix[idx, 0]]
#             # dst = pcds[group_matrix[idx, group_idx]]
#             dst_pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[group_matrix[idx, group_idx]]))
#             points = np.asarray(dst_pcd.points)
#             dst_pcd = dst_pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
#             dst_pcd = dst_pcd.voxel_down_sample(opt.voxel_size)
#             dst_pcd.estimate_normals()
#             R, t = utils.icp_o3d(src_pcd, dst_pcd, 0.5)
#             pose_est[idx, group_idx-1, :3] = t[:3].T
#             pose_est[idx, group_idx-1, 3:] = utils.mat2ang_np(R)
#         elif opt.mode == "gt":
#             pose_est[idx, group_idx-1] = gt_pose[group_matrix[idx, 0]] - gt_pose[group_matrix[idx, group_idx]]

# save_name = os.path.join(checkpoint_dir,'pose_pairwise.npy')
# np.save(save_name,pose_est)