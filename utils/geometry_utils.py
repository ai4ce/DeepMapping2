import torch
import numpy as np
import open3d as o3d
from open3d import pipelines
from sklearn.neighbors import NearestNeighbors
import sys

def add_z_coord_for_evaluation(pred_pos_DM):
    """
    pred_pos_DM (predicted position) estimated from DeepMapping only has x and z coordinate
    convert this to <x,y,z=0,theta> for evaluation
    """
    n = pred_pos_DM.shape[0]
    x = pred_pos_DM[:,0]
    y = pred_pos_DM[:,1]
    z = np.zeros_like(x)
    theta = pred_pos_DM[:,2]
    return np.stack((x,y,z,theta),axis=-1)

def transform_to_global_2D(pose, obs_local):
    """ 
    transform local point cloud to global frame
    row-based matrix product
    pose: <Bx3> each row represents <x,y,theta>
    obs_local: <BxLx2> 
    """
    L = obs_local.shape[1]
    # c0 is the loc of sensor in global coord. frame c0: <Bx2>
    c0, theta0 = pose[:, 0:2], pose[:, 2]
    c0 = c0.unsqueeze(1).expand(-1, L, -1)  # <BxLx2>

    cos = torch.cos(theta0).unsqueeze(-1).unsqueeze(-1)
    sin = torch.sin(theta0).unsqueeze(-1).unsqueeze(-1)
    R_transpose = torch.cat((cos, sin, -sin, cos), dim=1).reshape(-1, 2, 2)

    obs_global = torch.bmm(obs_local, R_transpose) + c0
    return obs_global

def transform_to_global_AVD(pose, obs_local):
    """
    transform obs local coordinate to global corrdinate frame
    :param pose: <Bx3> <x,z,theta> y = 0
    :param obs_local: <BxLx3> (unorganized) or <BxHxWx3> (organized)
    :return obs_global: <BxLx3> (unorganized) or <BxHxWx3> (organized)
    """
    is_organized = 1 if len(obs_local.shape) == 4 else 0
    b = obs_local.shape[0]
    if is_organized:
        H,W = obs_local.shape[1:3]
        obs_local = obs_local.view(b,-1,3) # <BxLx3>
    
    L = obs_local.shape[1]

    c0, theta0 = pose[:,0:2],pose[:,2] # c0 is the loc of sensor in global coord frame c0 <Bx2> <x,z>

    zero = torch.zeros_like(c0[:,:1])
    c0 = torch.cat((c0,zero),-1) # <Bx3> <x,z,y=0>
    c0 = c0[:,[0,2,1]] # <Bx3> <x,y=0,z>
    c0 = c0.unsqueeze(1).expand(-1,L,-1) # <BxLx3>
    
    cos = torch.cos(theta0).unsqueeze(-1).unsqueeze(-1)
    sin = torch.sin(theta0).unsqueeze(-1).unsqueeze(-1)
    zero = torch.zeros_like(sin)
    one = torch.ones_like(sin)
    
    R_y_transpose = torch.cat((cos,zero,-sin,zero,one,zero,sin,zero,cos),dim=1).reshape(-1,3,3)
    obs_global = torch.bmm(obs_local,R_y_transpose) + c0
    if is_organized:
        obs_global = obs_global.view(b,H,W,3)
    return obs_global


def transform_to_global_KITTI(pose, obs_local):
    """
    transform obs local coordinate to global corrdinate frame
    :param pose: <Nx3> <x,y,theta> z = 0
    :param obs_local: <BxNx3> 
    :return obs_global: <BxNx3>
    """
    # translation
    assert obs_local.shape[0] == pose.shape[0]
    theta = -pose[:, [2]]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation_matrix = torch.cat((cos_theta, -sin_theta, sin_theta, cos_theta), dim=1).reshape(-1, 2, 2)
    xy = obs_local[:, :, :2]
    xy_rotated = torch.bmm(xy, rotation_matrix)
    obs_global = torch.cat((xy_rotated, obs_local[:, :, [2]]), dim=2)
    obs_global[:, :, 0] = obs_global[:, :, 0] + pose[:, [0]]
    obs_global[:, :, 1] = obs_global[:, :, 1] + pose[:, [1]]
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

def estimate_normal_eig(data):
    """
    Computes the vector normal to the k-dimensional sample points
    """
    data -= np.mean(data,axis=0)
    data = data.T
    A = np.cov(data)
    w,v = np.linalg.eig(A)
    idx = np.argmin(w)
    v = v[:,idx]
    v /= np.linalg.norm(v,2)
    return v
    
def surface_normal(pc,n_neighbors=6):
    """
    Estimate point cloud surface normal
    Args:
        pc: Nxk matrix representing k-dimensional point cloud
    """
    
    n_points,k = pc.shape
    v = np.zeros_like(pc)
    
    # nn search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(pc)
    _, indices = nbrs.kneighbors(pc)
    neighbor_points = pc[indices]
    for i in range(n_points):
        # estimate surface normal
        v_tmp = estimate_normal_eig(neighbor_points[i,])
        v_tmp[abs(v_tmp)<1e-5] = 0
        if v_tmp[0] < 0:
            v_tmp *= -1
        v[i,:] = v_tmp
    return v


def point2plane_metrics_2D(p,q,v):
    """
    Point-to-plane minimization
    Chen, Y. and G. Medioni. “Object Modelling by Registration of Multiple Range Images.” 
    Image Vision Computing. Butterworth-Heinemann . Vol. 10, Issue 3, April 1992, pp. 145-155.
    
    Args:
        p: Nx2 matrix, moving point locations
        q: Nx2 matrix, fixed point locations
        v:Nx2 matrix, fixed point normal
    Returns:
        R: 2x2 matrix
        t: 2x1 matrix
    """
    assert q.shape[1] == p.shape[1] == v.shape[1] == 2, 'points must be 2D'
    
    p,q,v = np.array(p),np.array(q),np.array(v)
    c = np.expand_dims(np.cross(p,v),-1)
    cn = np.concatenate((c,v),axis=1)  # [ci,nix,niy]
    C = np.matmul(cn.T,cn)
    if np.linalg.cond(C)>=1/sys.float_info.epsilon:
        # handle singular matrix
        raise ArithmeticError('Singular matrix')
    
#     print(C.shape)
    qp = q-p
    b = np.array([
        [(qp*cn[:,0:1]*v).sum()],
        [(qp*cn[:,1:2]*v).sum()],
        [(qp*cn[:,2:]*v).sum()],
    ])

    X = np.linalg.solve(C, b)
    cos_ = np.cos(X[0])[0]
    sin_ = np.sin(X[0])[0]
    R = np.array([
        [cos_,-sin_],
        [sin_,cos_]
    ])
    t = np.array(X[1:])
    return R,t

def icp(src,dst,nv=None,n_iter=100,init_pose=[0,0,0],torlerance=1e-6,metrics='point',verbose=False):
    '''
    Currently only works for 2D case
    Args:
        src: <Nx2> 2-dim moving points
        dst: <Nx2> 2-dim fixed points
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
    n_src = src.shape[0]
    if metrics == 'plane' and nv is None:
        nv = surface_normal(dst)

    #src = np.matrix(src)
    #dst = np.matrix(dst)
    #Initialise with the initial pose estimation
    R_init = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]), 0],
                    [np.sin(init_pose[2]), np.cos(init_pose[2]), 0],
                    [0, 0, 1]  ])
    t_init = np.array([[init_pose[0]],
                    [init_pose[1]],
                    [0]  ])  
    
    #src =  R_init*src.T + t_init
    src = np.matmul(R_init,src.T) + t_init
    src = src.T
    
    R,t = R_init,t_init

    prev_err = np.inf
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
    for i in range(n_iter):
        # Find the nearest neighbours
        # print(src.shape)
        _, indices = nbrs.kneighbors(src)

        # Compute the transformation
        if metrics == 'point':
            R0,t0 = rigid_transform_kD(src,dst[indices[:,0]])
        elif metrics=='plane':
            try:
                R0,t0 = point2plane_metrics_2D(src,dst[indices[:,0]], nv[indices[:,0]]) 
            except ArithmeticError:
                print('Singular matrix')
                return src,R,t
        else:
            raise ValueError('metrics: {} not recognized.'.format(metrics))
        # Update dst and compute error
        src = np.matmul(R0,src.T) + t0
        src = src.T

        R = np.matmul(R0,R)
        t = np.matmul(R0,t) + t0
        #R = R0*R
        #t = R0*t + t0
        current_err = np.sqrt((np.array(src-dst[indices[:,0]])**2).sum()/n_src)

        if verbose:
            print('iter: {}, error: {}'.format(i,current_err))
            
        if  np.abs(current_err - prev_err) < torlerance:
            break
        else:
            prev_err = current_err
            
    return src,R,t


def icp_o3d(src,dst,nv=None,n_iter=100,init_pose=[0,0,0],torlerance=1e-6,metrics='point',verbose=False):
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
    src_pcd.point["positions"] = o3d.core.Tensor(src, dtype, device)
    src_pcd.estimate_normals()
    dst_pcd = o3d.t.geometry.PointCloud(device)
    dst_pcd.point["positions"] = o3d.core.Tensor(dst, dtype, device)
    dst_pcd.estimate_normals()

    voxel_sizes = o3d.utility.DoubleVector([1, 0.1, 0.01])

    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=1e-5,
                                    relative_rmse=1e-5,
                                    max_iteration=40),
        treg.ICPConvergenceCriteria(1e-6, 1e-6, 30),
        treg.ICPConvergenceCriteria(1e-8, 1e-8, 30)
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.utility.DoubleVector([3, 0.3, 0.03])

    # Initial alignment or source to target transform.
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64)

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPoint()

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
        output: <Nx4> predicted trajectory positions, where N is #scans
        target: <Nx4> ground truth trajectory positions
    Returns:
        trans_ate: <N> translation absolute trajectory error for each pose
        rot_ate: <N> rotation absolute trajectory error for each pose
    """
    output_location = output[:, :3]
    target_location = target[:, :3]
    R,t = rigid_transform_kD(output_location,target_location)
    location_aligned = np.matmul(R , output_location.T) + t
    location_aligned = location_aligned.T
    yaw_aligned = np.arctan2(R[1,0],R[0,0]) + output[:, 3]

    trans_error = np.linalg.norm(location_aligned - target_location, axis=1)
    rot_error = np.linalg.norm(yaw_aligned - target[:, 3])
    
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

def ang2mat(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c,-s],[s,c]])
    return R

def cat_pose_2D(pose0,pose1):
    """
    pose0, pose1: <Nx3>, numpy array
    each row: <x,y,theta>
    """
    assert(pose0.shape==pose1.shape)
    n_pose = pose0.shape[0]
    pose_out = np.zeros_like(pose0) 
    for i in range(n_pose):
        R0 = ang2mat(pose0[i,-1])
        R1 = ang2mat(pose1[i,-1])
        t0 = np.expand_dims(pose0[i,:2],-1)
        t1 = np.expand_dims(pose1[i,:2],-1)
        
        R = np.matmul(R1,R0)
        theta = np.arctan2(R[1,0],R[0,0])
        t = np.matmul(R1,t0) + t1
        pose_out[i,:2] = t.T
        pose_out[i,2] = theta
    return pose_out

def cat_pose_AVD(pose0,pose1):
    """
    pose0, pose1: <Nx3>, numpy array
    each row: <x,y,theta>
    """
    assert(pose0.shape==pose1.shape)
    n_pose = pose0.shape[0]
    pose_out = np.zeros_like(pose0) 
    for i in range(n_pose):
        R0 = ang2mat(pose0[i,-1])
        R1 = ang2mat(pose1[i,-1])
        t0 = np.expand_dims(pose0[i,:2],-1)
        t1 = np.expand_dims(pose1[i,:2],-1)
        
        R = np.matmul(R1,R0)
        theta = np.arctan2(R[1,0],R[0,0])
        t = np.matmul(R1,t0) + t1
        pose_out[i,:2] = t.T
        pose_out[i,2] = theta
    return pose_out

def convert_depth_map_to_pc(depth,fxy,cxy,max_depth=7000,depth_scale=2000):
    """
    create point cloud from depth map and camera instrinsic
    depth: <hxw> numpy array
    fxy: [fx,fy]
    cxy: [cx,cy]
    """
    fx,fy = fxy 
    cx,cy = cxy
    h,w = depth.shape
    
    c,r = np.meshgrid(range(1,w+1), range(1,h+1))
    invalid = depth >= max_depth
    depth[invalid] = 0

    z = depth / float(depth_scale)
    x = z * (c-cx) / fx
    y = z * (r-cy) / fy
    xyz = np.dstack((x,y,z)).astype(np.float32)
    return xyz
    
