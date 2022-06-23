import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

folder = "/mnt/NAS/home/xinhao/deepmapping/main/data/kitti/2011_09_30_drive_0018_sync_full2"
target_folder = "/mnt/NAS/home/xinhao/deepmapping/main/data/kitti/2011_09_30_drive_0018_sync_raw"
if not os.path.exists(target_folder):
    os.mkdir(target_folder)
files = os.listdir(folder)
files.remove("group_matrix.npy")
files.remove("gt_pose.npy")
for file in tqdm(files):
    xyz = np.load(os.path.join(folder, file))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # _, inliers = pcd.segment_plane(distance_threshold=0.1,
    #                                         ransac_n=10,
    #                                         num_iterations=500)
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.io.write_point_cloud(os.path.join(target_folder, file[:4]+".pcd"), outlier_cloud)
    o3d.io.write_point_cloud(os.path.join(target_folder, file[:4]+".pcd"), pcd)