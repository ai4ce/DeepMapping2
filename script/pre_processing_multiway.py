import set_path
import os
import argparse
import time
import numpy as np

import utils
import open3d as o3d
from matplotlib import rc
from tqdm import tqdm


def pairwise_registration(src, dst, max_correspondence_distance_coarse, max_correspondence_distance_fine, init_pose):
    # icp_coarse = o3d.pipelines.registration.registration_icp(
    #     src, dst, max_correspondence_distance_coarse, np.identity(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        src, dst, max_correspondence_distance_fine,
        init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, dst, max_correspondence_distance_fine,
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
parser.add_argument('-r', '--rotation', type=str, default="euler_angle", help="The rotation representation of pose estimation")
opt = parser.parse_args()
rc('image', cmap='rainbow_r')

if opt.rotation not in ['quaternion','euler_angle']:
    print("Unsupported rotation representation")
    assert()

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
# pcd_files = np.asarray(pcd_files)
pcds = []
if dataset == "KITTI" or "Nebula":
    for i in tqdm(range(n_pc)):
        pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[i])).voxel_down_sample(opt.voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
elif dataset == "NCLT":
    for i in tqdm(range(n_pc)):
    # for i in tqdm(range(10)):
        pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, pcd_files[i]))
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(np.linalg.norm(points, axis=1) < 100)[0])
        pcd = pcd.voxel_down_sample(opt.voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
num_seg = 250 # Number of segments in multiway regiatration
K = n_pc // num_seg # Number of frames in a segment

print('running initial icp')
trans_cums = [np.eye(4)]
segments = []
for idx in tqdm(range(n_pc-1)):
    src, dst = pcds[idx], pcds[idx+1]
    trans_cur = utils.icp_o3d(src, dst, 0.5, "matrix")
    if idx == 0: 
        trans_cum = trans_cur
    else:
        trans_cum = trans_cur @ trans_cum
    trans_cums.append(np.linalg.inv(trans_cum))
        
    if idx % K == K - 2:
        cur_segment = o3d.geometry.PointCloud()
        for sub_idx in range(idx-K+2, idx+1):
            cur_segment += pcds[sub_idx].transform(trans_cums[sub_idx])
            cur_segment = cur_segment.voxel_down_sample(opt.voxel_size)
            cur_segment = cur_segment.voxel_down_sample(1)
            cur_segment.estimate_normals()
        cur_segment.transform(np.linalg.inv(trans_cums[idx-K+2]))
        segments.append(cur_segment)
    # if len(segments) == 10:
    #     o3d.io.write_point_cloud("segment1.pcd", segments[8])
    #     o3d.io.write_point_cloud("segment2.pcd", segments[9])
    #     np.save("transform1.npy", trans_cums[8*K])
    #     np.save("transform2.npy", trans_cums[9*K])
    #     print(K, len(trans_cums))
    #     assert()
print("Number of segments:", len(segments))
print("running pairwise registration on segements")
start_time = time.time()
pose_graph = o3d.pipelines.registration.PoseGraph()
odometry = np.identity(4)
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
for source_id in tqdm(range(len(segments))):
    for target_id in range(source_id + 1, len(segments)):
        relative_pose = np.linalg.inv(trans_cums[source_id*K]) @ trans_cums[target_id*K]
        transformation_icp, information_icp = pairwise_registration(
                segments[source_id], segments[target_id], 80, opt.voxel_size, np.linalg.inv(relative_pose))
        if target_id == source_id + 1:  # odometry case
            odometry = transformation_icp @ odometry
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
            )
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=False))
        else:  # loop closure case
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=True))

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=opt.voxel_size,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )
end_time = time.time()
print("Running time:", end_time-start_time)

print('saving results')
pose_est = np.zeros((len(segments * K), 6))
for i in tqdm(range(len(segments * K))):
    trans_cur = pose_graph.nodes[i // K].pose.copy()
    trans_cum =  np.linalg.inv(trans_cums[i//K*K]) @ trans_cums[i]
    transformation = trans_cur @ trans_cum
    R, t = transformation[:3, :3], transformation[:3, 3:]
    pose_est[i, :3] = t[:3].T
    pose_est[i, 3:] = utils.mat2ang_np(R)
save_name = os.path.join(checkpoint_dir,'pose_est_icp.npy')
np.save(save_name,pose_est)

utils.plot_global_pose(checkpoint_dir, dataset, mode="prior", rotation_representation = opt.rotation)
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
gt_pose = gt_pose[:pose_est.shape[0]]
trans_ate, rot_ate = utils.compute_ate(pose_est, gt_pose, rotation_representation = opt.rotation) 
print('{}, translation ate: {}'.format(opt.name,trans_ate))
print('{}, rotation ate: {}'.format(opt.name,rot_ate))