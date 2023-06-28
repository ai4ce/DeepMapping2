import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import cm, colors, rc
from .pytorch3d_utils import *
from .geometry_utils import *

def plot_global_point_cloud(point_cloud, pose, valid_points, save_dir, **kwargs):
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.cpu().detach().numpy()
    if torch.is_tensor(pose):
        pose = pose.cpu().detach().numpy()
    if torch.is_tensor(valid_points):
        valid_points = valid_points.cpu().detach().numpy()

    file_name = 'global_map_pose'
    if kwargs is not None:
        for k, v in kwargs.items():
            file_name = file_name + '_' + str(k) + '_' + str(v)
    save_name = os.path.join(save_dir, file_name)

    bs = point_cloud.shape[0]
    for i in range(bs):
        current_pc = point_cloud[i, :, :]
        idx = valid_points[i, ] > 0
        current_pc = current_pc[idx]

        plt.plot(current_pc[:, 0], current_pc[:, 1], '.')
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.plot(pose[:, 0], pose[:, 1], color='black')
    plt.savefig(save_name)
    plt.close()

def save_global_point_cloud_open3d(point_cloud,pose,save_dir):
    file_name = 'global_map_pose'
    save_name = os.path.join(save_dir, file_name)

    n_pcd = len(point_cloud)
    for i in range(n_pcd):
        current_pc = np.asarray(point_cloud[i].points)
        plt.plot(current_pc[:, 0], current_pc[:, 1], '.',markersize=1)

    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.plot(pose[:, 0], pose[:, 1], color='black')
    plt.savefig(save_name)
    plt.close()

def plot_global_pose(checkpoint_dir, dataset="kitti", epoch=None, mode=None, rotation_representation="quaternion"):
    rc('image', cmap='rainbow_r')
    if mode == "prior":
        location = np.load(os.path.join(checkpoint_dir, "pose_est_icp.npy"))
    elif mode == "gt":
        location = np.load(os.path.join(checkpoint_dir, "gt_pose.npy"))
    else:
        location = np.load(os.path.join(checkpoint_dir, "pose_ests", str(epoch)+".npy"))
    if rotation_representation == "quaternion":
        q = location[:,3:]
        q = q[:, [1, 2, 3, 0]]
        rpy = Rot.from_quat(q).as_euler("XYZ")
        location = np.concatenate((location[:,:3],rpy),axis=1)
    t = np.arange(location.shape[0]) / location.shape[0]
    # location[:, 0] = location[:, 0] - np.mean(location[:, 0])
    # location[:, 1] = location[:, 1] - np.mean(location[:, 1])
    u = np.cos(location[:, -1]) * 2
    v = np.sin(location[:, -1]) * 2
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    if dataset.lower() == "kitti":
        ax.quiver(location[:, 0], location[:, 1], u, v, t, scale=10, scale_units='inches', width=2e-3)
    elif dataset.lower() == "nclt":
        ax.quiver(location[:, 0], location[:, 1], u, v, t, scale=20, scale_units='inches', width=1e-3)
    elif dataset.lower() == "tandon":
        u = np.cos(location[:, -1] + np.pi/2) * 2
        v = np.sin(location[:, -1] + np.pi/2) * 2
        ax.quiver(location[:, 0], location[:, 1], u, v, t, scale=10, scale_units='inches', width=2e-3)
    elif dataset.lower() == "nebula":
        ax.quiver(location[:, 0], location[:, 1], u, v, t, scale=20, scale_units='inches', width=1e-3)
    else:
        assert 0, "Unsupported dataset"

    ax.axis('equal')
    ax.tick_params(axis='both', labelsize=18)
    norm = colors.Normalize(0, location.shape[0])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'))
    cbar.ax.tick_params(labelsize=18)
    if mode == 'prior':
        ax.set_title("Warm Start", fontsize=32)
        plt.savefig(os.path.join(checkpoint_dir, "pose_prior.png"), dpi=600)
    elif mode == "gt":
        ax.set_title("Ground Truth", fontsize=32)
        plt.savefig(os.path.join(checkpoint_dir, "pose_gt.png"), dpi=600)
    else:
        if not os.path.exists(os.path.join(checkpoint_dir, "vis_traj")):
            os.mkdir(os.path.join(checkpoint_dir, "vis_traj"))
        ax.set_title("Epoch "+str(epoch), fontsize=32)
        plt.savefig(os.path.join(checkpoint_dir, "vis_traj", "pose_"+str(epoch)+".png"), dpi=600)
    plt.close()

def plot_curve(data, title, checkpoint_dir):
    plt.plot(np.arange(1, len(data)+1), data)
    plt.title(title)
    plt.savefig(os.path.join(checkpoint_dir, title+".png"))
    plt.close()

def plot_loss(loss, bce, ch, eu, title, checkpoint_dir):
    plt.plot(np.arange(1, len(loss)+1), loss, label="Weighted loss")
    plt.plot(np.arange(1, len(bce)+1), bce, label="BCE loss")
    plt.plot(np.arange(1, len(ch)+1), ch, label="Chamfer loss")
    plt.plot(np.arange(1, len(eu)+1), eu, label="Consistency loss")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, title+".png"))
    plt.close()
