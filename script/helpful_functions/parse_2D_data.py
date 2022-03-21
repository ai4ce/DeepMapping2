import os
import numpy as np
from open3d import PointCloud,Vector3dVector,write_point_cloud
from scipy.io import savemat

out_folder = "/mnt/NAS/data/cc_data/2D_real_corrected_bk/run_1"
in_file_folder = "/mnt/NAS/data/cc_data/2D_log/fr079-complete.gfs.log"

gt_poses = []

def np_to_pcd(xyz):
    xyz = xyz.reshape(-1,3)
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    return pcd

count = -1
with open(in_file_folder) as f:
    lines = f.readlines()
    for line in lines:
        lines_array = line.split(" ")
        if lines_array[0]=="FLASER":
            count = count + 1
            xyz = np.zeros((360,3), dtype=np.float32)
            for i in range(360):
                xyz[i,0] = float(lines_array[i+2])*np.cos(i*np.pi/180)
                xyz[i,1] = float(lines_array[i+2])*np.sin(i*np.pi/180)
                xyz[i,2] = 0
            pcd = np_to_pcd(xyz)
            save_name = '{:09d}.pcd'.format(count)
            save_name = os.path.join(out_folder,save_name)
            write_point_cloud(save_name,pcd)

            gt_pose = []
            gt_pose.append(float(lines_array[i+3]))
            gt_pose.append(float(lines_array[i+4]))
            gt_pose.append(float(lines_array[i+5]))
            gt_poses.append(gt_pose)
    gt_poses = np.array(gt_poses,dtype=np.float32)
    print("gt_poses:"+str(gt_poses.shape))
    print("count:"+str(count))
    savemat(os.path.join(out_folder,"gt_pose.mat"),{"pose":gt_poses})
