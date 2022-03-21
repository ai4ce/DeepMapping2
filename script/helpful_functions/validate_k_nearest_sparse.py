import set_path
import os
import torch
import numpy as np 
from dataset_loader import SimulatedPointCloud
from torch.utils.data import DataLoader

import utils
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from open3d import read_point_cloud

checkpoint_dir = os.path.join('../../','data')
checkpoint_dir_validate = os.path.join('../../results/2D',"gt_map_validate")
if not os.path.exists(checkpoint_dir_validate):
    os.makedirs(checkpoint_dir_validate)

def validate(temp_index):
    #print("epoch num:"+str(temp_index))
    #mode = "0"
    #mode = "1"
    #mode = "2_aug"
    #mode = "2_aug_8"
    #mode = "2_aug_devel_8"
    #mode = "2_aug_devel_2"
    #mode = "2_aug_no_feature"
    mode = "4"
    #mode = "6"

    if mode == "0":
        new_dir = "/home/cc/with_no_time_info/0_Unsupervised-PointNetVlad_time_seq2/results"
    elif mode == "1":
        new_dir = "/home/cc/with_no_time_info/1_Unsupervised-PointNetVlad_SOTA_FS/results"
    elif mode == "2_aug":
        new_dir = "/home/cc/with_no_time_info/2_Unsupervised-PointNetVlad_aug/results"
    elif mode == "2_aug_8":
        new_dir = "/home/cc/with_no_time_info/2_Unsupervised-PointNetVlad_aug_8/results"
    elif mode == "2_aug_devel_8":
        new_dir = "/home/cc/with_no_time_info/2_Unsupervised-PointNetVlad_aug_devel_2_8/results"
    elif mode == "2_aug_devel_2":
        new_dir = "/home/cc/with_no_time_info/2_Unsupervised-PointNetVlad_aug_devel_2/results"
    elif mode == "2_aug_no_feature":
        new_dir = "/home/cc/with_no_time_info/2_Unsupervised-PointNetVlad_aug_devel_no_feature_space_neighborhood/results" 
    elif mode == "4":
        new_dir = "/home/cc/with_no_time_info/4_Unsupervised-PointNetVlad_SOTA_AUG_2/results"
    elif mode == "6":
        new_dir = "/home/cc/with_no_time_info/6_Supervised-PointNetVlad/results"
    
    save_name = os.path.join(new_dir,'database'+str(temp_index)+'.npy')
    best_matrix = np.load(save_name)
    best_matrix = torch.tensor(best_matrix, dtype = torch.float64)
    best_matrix = np.array(best_matrix)

    data_dir = '/home/cc/dm_data_sampled/'
    all_folders = sorted(os.listdir(data_dir))

    folders = []
    # All runs are used for training (both full and partial)
    index_list = [5,6,7,9,11,14,18,19,29,31,32,34,36,40,44,46,49,51,53,54,57,59,61,63,65,68,69,72,75,77,79,83,85,88,91,93,94]
    if mode == "supervised":
        index_list = [5,6,7,9]
    for index in index_list:
        folders.append(all_folders[index])

    all_folders = folders

    indices = []
    best_matrix_list = []
    
    check_total = best_matrix.shape[0] * best_matrix.shape[1]
    best_matrix = best_matrix.reshape((int(best_matrix.shape[0])*best_matrix.shape[1]),best_matrix.shape[2])
    
    #for index,folder in enumerate(all_folders):
    '''
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(best_matrix)
    distance, indice = nbrs.kneighbors(best_matrix)
    indices.append(indice)
    
    indices = np.array(indices, dtype=np.int64)
    '''

    nbrs_10 = NearestNeighbors(n_neighbors=int(check_total//10), algorithm='ball_tree').fit(best_matrix)
    nbrs_25 = NearestNeighbors(n_neighbors=int(check_total//25), algorithm='ball_tree').fit(best_matrix)
    nbrs_100 = NearestNeighbors(n_neighbors=int(check_total//100), algorithm='ball_tree').fit(best_matrix)
    distance_10, indice_10 = nbrs_10.kneighbors(best_matrix)
    distance_25, indice_25 = nbrs_25.kneighbors(best_matrix)
    distance_100, indice_100 = nbrs_100.kneighbors(best_matrix)
    indice_10 = np.array(indice_10, dtype=np.int64)
    indice_25 = np.array(indice_25, dtype=np.int64)
    indice_100 = np.array(indice_100, dtype=np.int64)
    
    '''
    best_matrix_list = np.array(best_matrix_list, dtype=np.float64)

    F, N, _ = best_matrix_list.shape
    colors = cm.rainbow(np.linspace(0, 1, N))

    if not os.path.exists(os.path.join(checkpoint_dir_validate, "pcl_map_TSNE")):
        os.makedirs(os.path.join(checkpoint_dir_validate, "pcl_map_TSNE"))

    for f in range(F):
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.scatter(best_matrix_list[f, :, 0], best_matrix_list[f, :, 1], zorder=3, c=colors)
        print("saving figure to "+str(os.path.join(checkpoint_dir_validate, "pcl_map_TSNE", 'unsupervised_tsne_'+str(f)+'.png')))
        plt.savefig(os.path.join(checkpoint_dir_validate, "pcl_map_TSNE", 'unsupervised_tsne'+str(f)+'.png'), bbox_inches='tight')
        plt.close()
    '''

    init_pose = None

    indices_gt = []
    indices_gt_true = []

    total_location = torch.zeros((check_total,2), dtype=torch.float).cpu()

    for index,folder in enumerate(all_folders):
        data_dir_f = os.path.join(data_dir, folder) 
        if not os.path.exists(data_dir_f):
            os.makedirs(data_dir_f)
        checkpoint_dir_validate_f = os.path.join(checkpoint_dir_validate, folder)
        if not os.path.exists(checkpoint_dir_validate_f):
            os.makedirs(checkpoint_dir_validate_f)
        gt_file = os.path.join(data_dir_f,'gt_pose.mat')
        gt_pose = sio.loadmat(gt_file)
        gt_pose = gt_pose['pose']
        gt_location = gt_pose[:,:2]
        pose_est = torch.tensor(gt_pose, dtype = torch.float).cpu()
        location_est = torch.tensor(gt_location, dtype = torch.float).cpu()
        #print("pose_est:"+str(pose_est.size()))
        total_location[index*32:(index+1)*32,:] = location_est
    
    #assert(0)
    '''
    nbrs_10 = NearestNeighbors(n_neighbors=int(check_total//10), algorithm='ball_tree').fit(total_location)
    nbrs_20 = NearestNeighbors(n_neighbors=int(check_total//20), algorithm='ball_tree').fit(total_location)
    nbrs_100 = NearestNeighbors(n_neighbors=int(check_total//100), algorithm='ball_tree').fit(total_location)

    distance_10, indice_10 = nbrs_10.kneighbors(total_location)
    distance_20, indice_20 = nbrs_20.kneighbors(total_location)
    distance_100, indice_100 = nbrs_100.kneighbors(total_location)
    
    indice_10 = np.array(indice_10, dtype=np.int64)
    indice_20 = np.array(indice_25, dtype=np.int64)
    indice_100 = np.array(indice_10, dtype=np.int64)
    '''
    nbrs_25_gt = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(total_location)
    distance_25_gt, indice_25_gt = nbrs_25_gt.kneighbors(total_location)
    indice_25_gt = np.array(indice_25_gt, dtype=np.int64)
    '''
    indices_gt.append(indice_25)
    indices_gt = np.array(indices_gt, dtype=np.int64) # Loose Constraint
    '''
    #indices_gt_true = np.array(indices_gt_true, dtype=np.int64)


    #### Accuracy ######
    print("indice_10:"+str(indice_10.shape))
    P_10, N_10 = indice_10.shape
    P_25, N_25 = indice_25.shape
    P_100, N_100 = indice_100.shape
    P_25_gt, N_25_gt = indice_25_gt.shape
    
    #total_grid = float(P*N)
    acc_10_count = 0.0
    acc_25_count = 0.0
    acc_100_count = 0.0
    fp_count = 0.0
    afp_count = 0.0
    fn_count_10 = 0.0
    fn_count_25 = 0.0
    fn_count_100 = 0.0
    afn_count = 0.0
    false_positive = []
    absolute_false_positive = []
    false_negative = []
    absolute_false_negative = []

    for p in range(P_10):
        for n in range(N_10):
            if indice_10[p,n] in indice_25_gt[p]:
                acc_10_count = acc_10_count + 1
    for p in range(P_25):
        for n in range(N_25):
            if indice_25[p,n] in indice_25_gt[p]:
                acc_25_count = acc_25_count + 1
    for p in range(P_100):
        for n in range(N_100):
            if indice_100[p,n] in indice_25_gt[p]:
                acc_100_count = acc_100_count + 1

            '''
            if (indices[p,n] not in indices_gt_true[p]):
                afp_count = afp_count + 1
            if (indices[p,n] not in indices_gt[p]):
                fp_count = fp_count + 1
            if (indices_gt_true[p,n] not in indices[p]):
                afn_count = afn_count + 1
            '''
    print(acc_10_count)
    print(acc_25_count)
    print(acc_100_count)
    for p in range(P_25_gt):
        for n in range(N_25_gt):
            if (indice_25_gt[p,n] not in indice_10[p]):
                fn_count_10 = fn_count_10 + 1
            if (indice_25_gt[p,n] not in indice_25[p]):
                fn_count_25 = fn_count_25 + 1
            if (indice_25_gt[p,n] not in indice_100[p]):
                fn_count_100 = fn_count_100 + 1
            '''
            if (indices[b,p,n] not in indices_gt_true[b,p]):
                absolute_false_positive.append(indices[b,p,n])
            if (indices[b,p,n] not in indices_gt[b,p]):
                false_positive.append(indices[b,p,n])
            if (indices_gt_true[b,p,n] not in indices[b,p]):
                absolute_false_negative.append(indices_gt_true[b,p,n])
            if (indices_gt[b,p,n] not in indices[b,p]):
                false_negative.append(indices_gt[b,p,n])
            '''
    '''
    print("trained nearest 16 neigbours for packet 0:"+str(indices[0,0]))
    print("top 16 nearest neighbours: :"+str(indices_gt_true[0,0]))
    print("top nearest 48 neigbours:"+str(indices_gt[0,0]))
    print("false_positives for packet 0:"+str(false_positive))
    print("false_positives with loose constraint for packet 0:"+str(absolute_false_positive))
    print("false_negatives for packet 0:"+str(false_negative))
    print("false_negatives with loose constraint for packet 0:"+str(absolute_false_negative))
    '''
    '''
    if not os.path.exists(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train")):
        os.makedirs(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train"))

    if not os.path.exists(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt")):
        os.makedirs(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt"))

    if not os.path.exists(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true")):
        os.makedirs(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true"))

    for index_ in indices_gt_true[0,0]:
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        #print("index_:"+str('{0:04}'.format(index_)))
        pcl = read_point_cloud("/home/cc/dm_data/v0_pose05/00000"+str('{0:04}'.format(index_))+".pcd")
        pcl = np.asarray(pcl.points, dtype=np.float32)
        ax.scatter(pcl[:, 0], pcl[:, 1], zorder=2)
        #print("saving figure to "+str(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true", 'unsupervised_'+str(index_)+'.png')))
        plt.savefig(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true", 'unsupervised_'+str(index_)+'.png'), bbox_inches='tight')
        plt.close()

    for index_ in indices_gt[0,0]:
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        #print("index_:"+str('{0:04}'.format(index_)))
        pcl = read_point_cloud("/home/cc/dm_data/v0_pose05/00000"+str('{0:04}'.format(index_))+".pcd")
        pcl = np.asarray(pcl.points, dtype=np.float32)
        ax.scatter(pcl[:, 0], pcl[:, 1], zorder=2)
        #print("saving figure to "+str(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt", 'unsupervised_'+str(index_)+'.png')))
        plt.savefig(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt", 'unsupervised_'+str(index_)+'.png'), bbox_inches='tight')
        plt.close()

    for index_ in indices[0,0]:
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        #print("index_:"+str('{0:04}'.format(index_)))
        pcl = read_point_cloud("/home/cc/dm_data/v0_pose05/00000"+str('{0:04}'.format(index_))+".pcd")
        pcl = np.asarray(pcl.points, dtype=np.float32)
        ax.scatter(pcl[:, 0], pcl[:, 1], zorder=2)
        #print("saving figure to "+str(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train", 'unsupervised_'+str(index_)+'.png')))
        plt.savefig(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train", 'unsupervised_'+str(index_)+'.png'), bbox_inches='tight')
        plt.close()
    
    '''
    print("################################################")
            #print("indices[b,p,n]:"+str(indices[b,p,n]))
            #print("indices_gt[b,p]:"+str(indices_gt[b,p]))
    
    #print("acc_count:"+str(acc_count))
    #print("fp_count:"+str(fp_count))
    #print("The Precision: "+str(acc_count/(fp_count+acc_count)))
    print("The Recall_10: "+str(acc_10_count/(fn_count_10+acc_10_count)))
    print("The Recall_25: "+str(acc_25_count/(fn_count_25+acc_25_count)))
    print("The Recall_100: "+str(acc_100_count/(fn_count_100+acc_100_count)))
    save_name = os.path.join(checkpoint_dir,'best_n_trained.npy')

    #np.save(save_name,indices[0])
    print("Done")


if __name__ == "__main__":
    for i in range(100):
        validate(i)
