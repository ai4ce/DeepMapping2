import torch
import numpy as np
import open3d as o3d
from open3d import pipelines
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as Rot
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


def compose_pose_diff(pose_est, pairwise):
    """
    compose global estimation and local pairwise pose for comparison
    :param pose_est: global estimation of shape <Bx6> <x,y,z,row,pitch,yaw>
    :param pairwise: pairwise pose of shape <B-1x6> <x,y,z,row,pitch,yaw>
    :return t_src: global estimation of shape <B-1x6>
    :return dst: pairwise + adjacent pose of shape <B-1x6>
    """
    G = pose_est.shape[0]
    # src = pose_est[:1, :].expand(G-1, -1)
    t_src = pose_est[:1, :3].expand(G-1, -1)
    r_src = pose_est[:1, 3:].expand(G-1, -1)
    r_src = euler_angles_to_matrix(r_src, convention="XYZ")
    r_src = matrix_to_quaternion(r_src)

    xyz = pose_est[1:, :3] + pairwise[:, :3]
    t_dst = xyz
    rpy_est = pose_est[1:, 3:]
    rpy_pairwise = pairwise[:, 3:]
    rotation_est = euler_angles_to_matrix(rpy_est, convention="XYZ")
    rotation_est = matrix_to_quaternion(rotation_est)

    rotation_pairwise = euler_angles_to_matrix(rpy_pairwise, convention="XYZ")
    rotation_pairwise = matrix_to_quaternion(rotation_pairwise)

    r_dst = torch.bmm(rotation_est, rotation_pairwise)
    # rpy = matrix_to_euler_angles(rotation, convention="XYZ")
    # dst = torch.concat((xyz, rpy), dim=1)
    return t_src, t_dst, r_src, r_dst


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


def icp_o3d(src, dst, voxel_size=0.5, return_type="Rt"):
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
    # List of Convergence-Criteria for Multi-Scale ICP:
    threshold = 1
    trans_init = np.eye(4)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, 1e-6, max_iteration=80)

    registration = o3d.pipelines.registration.registration_icp(
        src, dst, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )

    transformation = registration.transformation
    if return_type == "Rt":
        R = transformation[:3, :3]
        t = transformation[:3, 3:]
        return R.copy(), t.copy()
    else:
        return transformation.copy()


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
    rotation = Rot.from_matrix(R).as_euler("XYZ")
    yaw_aligned = output[:, -1] + rotation[-1]
    yaw_gt = target[:, -1]
    while np.any(yaw_aligned > np.pi):
        yaw_aligned[yaw_aligned > np.pi] = yaw_aligned[yaw_aligned > np.pi] - 2 * np.pi
    while np.any(yaw_aligned < -np.pi):
        yaw_aligned[yaw_aligned < -np.pi] = yaw_aligned[yaw_aligned < -np.pi] + 2 * np.pi
    while np.any(yaw_gt > np.pi):
        yaw_gt[yaw_gt > np.pi] = yaw_gt[yaw_gt > np.pi] - 2 * np.pi
    while np.any(yaw_gt < -np.pi):
        yaw_gt[yaw_gt < -np.pi] = yaw_gt[yaw_gt < -np.pi] + 2 * np.pi

    trans_error = np.linalg.norm(location_aligned - target_location, axis=1)
    rot_error = np.abs(yaw_aligned - yaw_gt)
    rot_error[rot_error > np.pi] = 2 * np.pi - rot_error[rot_error > np.pi]
    
    trans_ate = np.sqrt(np.mean(trans_error))
    rot_ate = np.mean(rot_error)

    return trans_ate, rot_ate*180/np.pi


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
    r = Rot.from_matrix(mat)
    return r.as_euler("XYZ", degrees=False)


def ang2mat_np(ang):
    r = Rot.from_euler("XYZ", ang)
    return r.as_matrix()
