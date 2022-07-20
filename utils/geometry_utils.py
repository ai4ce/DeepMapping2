import torch
import numpy as np
import open3d as o3d
from open3d import pipelines
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from .pytorch3d_utils import *
import sys


# def transform_to_global_KITTI(pose, obs_local):
#     """
#     transform obs local coordinate to global corrdinate frame
#     :param pose: <Nx4> <x,y,z,theta>
#     :param obs_local: <BxNx3> 
#     :return obs_global: <BxNx3>
#     """
#     # translation
#     assert obs_local.shape[0] == pose.shape[0]
#     theta = pose[:, 3:]
#     cos_theta = torch.cos(theta)
#     sin_theta = torch.sin(theta)
#     rotation_matrix = torch.cat((cos_theta, sin_theta, -sin_theta, cos_theta), dim=1).reshape(-1, 2, 2)
#     xy = obs_local[:, :, :2]
#     xy_rotated = torch.bmm(xy, rotation_matrix)
#     obs_global = torch.cat((xy_rotated, obs_local[:, :, [2]]), dim=2)
#     # obs_global[:, :, 0] = obs_global[:, :, 0] + pose[:, [0]]
#     # obs_global[:, :, 1] = obs_global[:, :, 1] + pose[:, [1]]
#     obs_global = obs_global + pose[:, :3].unsqueeze(1)
#     return obs_global


def transform_to_global_KITTI(pose, obs_local):
    """
    transform obs local coordinate to global corrdinate frame
    :param pose: <Bx6> <x, y, z, row, pitch, yaw>
    :param obs_local: <BxNx3> 
    :return obs_global: <BxNx3>
    """
    # translation
    assert obs_local.shape[0] == pose.shape[0]
    rpy = pose[:, 3:]
    rotation_matrix = euler_angles_to_matrix(rpy, convention="XYZ")
    obs_global = torch.bmm(obs_local, rotation_matrix.transpose(1, 2))
    # obs_global[:, :, 0] = obs_global[:, :, 0] + pose[:, [0]]
    # obs_global[:, :, 1] = obs_global[:, :, 1] + pose[:, [1]]
    obs_global = obs_global + pose[:, :3].unsqueeze(1)
    return obs_global


def rigid_transform_kD(A, B):
    """
    Find optimal transformation between two sets of corresponding points
    Adapted from: http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    Args:
        A.B: <Nxk> each row represent a k-D points
    Returns:
        R: kxk
        t: kx1
        B = R*A+t
    """
    assert len(A) == len(B)
    N,k = A.shape
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA) , BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T , U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[k-1,:] *= -1
        R = np.matmul(Vt.T , U.T)

    t = np.matmul(-R,centroid_A.T) + centroid_B.T
    t = np.expand_dims(t,-1)
    return R, t


def icp_o3d(src,dst,voxel_size=1):
    '''
    Don't support init_pose and only supports 3dof now.
    Args:
        src: <Nx3> 3-dim moving points
        dst: <Nx3> 3-dim fixed points
        n_iter: a positive integer to specify the maxium nuber of iterations
        init_pose: [tx,ty,theta] initial transformation
        torlerance: the tolerance of registration error
        metrics: 'point' or 'plane'
        
    Return:
        src: transformed src points
        R: rotation matrix
        t: translation vector
        R*src + t
    '''
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    treg = o3d.t.pipelines.registration
    src_pcd = o3d.t.geometry.PointCloud(device)
    src_pcd.point["positions"] = o3d.core.Tensor(np.asarray(src.points), dtype, device)
    src_pcd.estimate_normals()
    dst_pcd = o3d.t.geometry.PointCloud(device)
    dst_pcd.point["positions"] = o3d.core.Tensor(np.asarray(dst.points), dtype, device)
    dst_pcd.estimate_normals()

    voxel_sizes = o3d.utility.DoubleVector([voxel_size])

    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=1e-5,
                                    relative_rmse=1e-5,
                                    max_iteration=30),
        # treg.ICPConvergenceCriteria(1e-5, 1e-5, 30),
        # treg.ICPConvergenceCriteria(1e-6, 1e-6, 50)
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.utility.DoubleVector([3 * voxel_size])

    # Initial alignment or source to target transform.
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64)

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPlane()

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    save_loss_log = True

    registration_ms_icp = treg.multi_scale_icp(src_pcd, dst_pcd, voxel_sizes,
                                           criteria_list,
                                           max_correspondence_distances,
                                           init_source_to_target, estimation,
                                           save_loss_log)

    transformation = registration_ms_icp.transformation
    R = transformation[:3, :3]
    t = transformation[:3, 3:]
    return None, R.numpy(), t.numpy()


def compute_ate(output,target):
    """
    compute absolute trajectory error for avd dataset
    Args:
        output: <Nx6> predicted trajectory positions, where N is #scans
        target: <Nx6> ground truth trajectory positions
    Returns:
        trans_ate: <N> translation absolute trajectory error for each pose
        rot_ate: <N> rotation absolute trajectory error for each pose
    """
    output_location = output[:, :3]
    target_location = target[:, :3]
    R, t = rigid_transform_kD(output_location,target_location)
    location_aligned = np.matmul(R , output_location.T) + t
    location_aligned = location_aligned.T
    yaw_aligned = np.arctan2(R[1,0],R[0,0]) + output[:, -1]
    while np.any(yaw_aligned > np.pi):
        yaw_aligned[yaw_aligned > np.pi] = yaw_aligned[yaw_aligned > np.pi] - 2 * np.pi
    while np.any(yaw_aligned < -np.pi):
        yaw_aligned[yaw_aligned < np.pi] = yaw_aligned[yaw_aligned < np.pi] + 2 * np.pi

    trans_error = np.linalg.norm(location_aligned - target_location, axis=1)
    rot_error = np.linalg.norm(yaw_aligned - target[:, -1])
    
    trans_ate = np.sqrt(np.mean(trans_error))
    rot_ate = np.sqrt(np.mean(rot_error))

    return trans_ate, rot_ate

def remove_invalid_pcd(pcd):
    """
    remove invalid in valid points that have all-zero coordinates
    pcd: open3d pcd objective
    """
    pcd_np = np.asarray(pcd.points) # <Nx3>
    non_zero_coord = np.abs(pcd_np) > 1e-6 # <Nx3>
    valid_ind = np.sum(non_zero_coord,axis=-1)>0 #<N>
    valid_ind = list(np.nonzero(valid_ind)[0])
    valid_pcd = o3d.geometry.select_down_sample(pcd,valid_ind)
    return valid_pcd


def mat2ang_np(mat):
    r = R.from_matrix(mat)
    return r.as_euler("XYZ", degrees=False)


def ang2mat_np(ang):
    r = R.from_euler("XYZ", ang)
    return r.as_matrix()
