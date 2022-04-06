import numpy as np
import struct
import os

from multiprocessing import Pool


def convert_kitti_bin_to_np(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    return np_pcd


def convert_kitti_txt_to_pose(path):
    with open(path, 'r') as f:
        data = f.read().strip().split(' ')
        pose = data[:6]
        for i in range(len(pose)):
            pose[i] = eval(pose[i])
        return np.asarray(pose)


def convert_kitti_sync_to_deepmapping(root):
    save_path = os.path.join('/mnt/NAS/home/xinhao/deepmapping/DeepMappingPP/data/kitti', root.split('/')[-1])
    os.mkdir(save_path)
    pcd_folder = os.path.join(root, 'velodyne_points/data')
    files = sorted(os.listdir(pcd_folder))
    for i in range(len(files)):
        files[i] = os.path.join(pcd_folder, files[i])
    # print(files)
    # assert()
    with Pool(8) as p:
        pcd_data = p.map(convert_kitti_bin_to_np, files)
    for i in range(len(pcd_data)):
        np.save(os.path.join(save_path, f'{i:04}.npy'), pcd_data[i])

    pose_file = os.path.join(root, 'oxts/data')
    files = sorted(os.listdir(pose_file))
    for i in range(len(files)):
        files[i] = os.path.join(pose_file, files[i])

    with Pool(8) as p:
        np_pose_data = p.map(convert_kitti_txt_to_pose, files)
    np_pose = np.stack(np_pose_data)
    np.save(os.path.join(save_path, 'gt_pose.npy'), np_pose)
    print(np_pose.shape)

root = '/mnt/NAS/home/xinhao/deepmapping/temp'
files = os.listdir(root)
for file in files:
    convert_kitti_sync_to_deepmapping(os.path.join(root, file))
    print(file, 'finished')
print('DONE')