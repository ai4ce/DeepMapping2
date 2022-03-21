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
    #mode = "baseline"
    #mode = "Supervised_NetVlad"
    #mode = "Our_method_RGB"
    mode = "Our_method_RGB_w_temporal"
    #mode = "Method_2_RGB_no_feat_w_temporal"
   # mode = "Method_2_RI_RGB_no_feat__w_temporal"

    if mode == "baseline":
        new_dir = "/home/cc/baseline_RGB/results/Nimmons"
    elif mode == "Supervised_NetVlad":
        new_dir = "/home/cc/Supervised_NetVlad/results/Nimmons"
    elif mode == "Our_method_RGB":
        new_dir = "/home/cc/Our_method_RGB/results/Micanopy"
    elif mode == "Our_method_RGB_w_temporal":
        new_dir = "/home/cc/Our_method_RGB_w_temporal/results/Micanopy"
    elif mode == "Method_2_RGB_no_feat_w_temporal":
        new_dir = "/home/cc/Method_2_RGB_no_feat_w_temporal/results/Micanopy"
    elif mode == "Method_2_RI_RGB_no_feat__w_temporal":
        new_dir = "/home/cc/Method_2_RI_RGB_no_feat__w_temporal/results/Micanopy"
    save_name = os.path.join(new_dir,'database'+str(temp_index)+'.npy')
    best_matrix = np.load(save_name)
    best_matrix = torch.tensor(best_matrix, dtype = torch.float64)
    best_matrix = np.array(best_matrix)
    data_dir = '/mnt/NAS/home/yiming/habitat_2/Micanopy'
    all_folders = sorted(os.listdir(data_dir))

    folders = []
    # All runs are used for training (both full and partial)
    index_list = [0,1,3]
    print("Number of runs: "+str(len(index_list)))
    for index in index_list:
        print("all_folders[index]:"+str(all_folders[index]))
        folders.append(all_folders[index])
    print(folders)
    folder_sizes = []
    #all_folder_sizes = []

    for folder_ in folders:
        all_files = list(sorted(os.listdir(os.path.join(data_dir,folder_))))
        all_files.remove("gt_pose.mat")
        folder_sizes.append(len(all_files))
    
    '''
    for folder_ in all_folders:
        all_files = list(sorted(os.listdir(os.path.join(data_dir,folder_,"jpg_rgb"))))
        all_folder_sizes.append(len(all_files))
    '''
    all_folders = folders
    indices_10 = []
    indices_25 = []
    indices_100 = []
    best_matrix_list = []
    
    #print("all_folder_sizes:"+str(all_folder_sizes))
    print("len(index_list):"+str(len(index_list)))
    #############
    
    for j, index in enumerate(index_list):
        folder = all_folders[j]
        if j == 0:
            overhead = 0
        else:
            overhead = 0
            for i in range(j):
                overhead = overhead + folder_sizes[i]
        #print("best_matrix:"+str(best_matrix.shape))
        #best_matrix_sec = best_matrix[overhead:overhead+folder_sizes[j]]
        best_matrix_sec = best_matrix[j]
        nbrs_10 = NearestNeighbors(n_neighbors=int(best_matrix_sec.shape[0]/10), algorithm='ball_tree').fit(best_matrix_sec)
        nbrs_25 = NearestNeighbors(n_neighbors=int(best_matrix_sec.shape[0]/25), algorithm='ball_tree').fit(best_matrix_sec)
        nbrs_100 = NearestNeighbors(n_neighbors=int(best_matrix_sec.shape[0]/100), algorithm='ball_tree').fit(best_matrix_sec)
        
        distance_10, indice_10 = nbrs_10.kneighbors(best_matrix_sec)
        distance_25, indice_25 = nbrs_25.kneighbors(best_matrix_sec)
        distance_100, indice_100 = nbrs_100.kneighbors(best_matrix_sec)
        indices_10.append(indice_10)
        indices_25.append(indice_25)
        indices_100.append(indice_100)
    indices_10 = np.array(indices_10, dtype=np.int64)
    indices_25 = np.array(indices_25, dtype=np.int64)
    indices_100 = np.array(indices_100, dtype=np.int64)
    
    #############
    init_pose = None

    indices_gt = []
    #indices_gt_true = []

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
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(location_est)
        #nbrs_16 = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(location_est)

        distance_, indice_ = nbrs.kneighbors(location_est)
        #distance_16, indice_16 = nbrs_16.kneighbors(location_est)

        indice_ = np.array(indice_, dtype=np.int64)
        #indice_16 = np.array(indice_16, dtype=np.int64)

        indices_gt.append(indice_)
        #indices_gt_true.extend(indice_16)

        #utils.draw_graphs(location_est, indices[index], 1, checkpoint_dir_validate_f, downsample_size = 32)

    indices_gt = np.array(indices_gt, dtype=np.int64) # Loose Constraint
    #indices_gt_true = np.array(indices_gt_true, dtype=np.int64)
    #print("indices_gt:"+str(indices_gt.shape))
    #print("indices_gt_true:"+str(indices_gt_true.shape))
    #assert(0)
    
    #### Accuracy ######
    B_10, N_10, P_10 = indices_10.shape
    B_25, N_25, P_25 = indices_25.shape
    B_100, N_100, P_100 = indices_100.shape
    B_gt, N_gt, P_gt = indices_gt.shape

    acc_count_10 = 0.0
    acc_count_25 = 0.0
    acc_count_100 = 0.0
    
    '''
    fp_count = 0.0
    afp_count = 0.0
    '''
    fn_count_10 = 0.0
    fn_count_25 = 0.0
    fn_count_100 = 0.0
    '''
    afn_count = 0.0
    false_positive = []
    absolute_false_positive = []
    false_negative = []
    absolute_false_negative = []
    '''
    print("B_10:"+str(B_10))
    print("N_10:"+str(N_10))
    print("indice_10:"+str(indice_10.shape))
    print("indices_gt:"+str(indices_gt.shape))
    for b in range(B_10):
        for n in range(N_10):
            for p in range(P_10):
                if indices_10[b,n,p] in indices_gt[b,n]:
                    acc_count_10 = acc_count_10 + 1
    for b in range(B_25):
        for n in range(N_25):
            for p in range(P_25):
                if indices_25[b,n,p] in indices_gt[b,n]:
                    acc_count_25 = acc_count_25 + 1
    for b in range(B_100):
        for n in range(N_100):
            for p in range(P_100):
                if indices_100[b,n,p] in indices_gt[b,n]:
                    acc_count_100 = acc_count_100 + 1
    
            '''
            if (indices[b,n] not in indices_gt_true[b]):
                afp_count = afp_count + 1
            if (indices[b,n] not in indices_gt[b]):
                fp_count = fp_count + 1
            if (indices_gt_true[b,n] not in indices[b]):
                afn_count = afn_count + 1
            '''
    for b in range(B_gt):
        for n in range(N_gt):
            for p in range(P_gt):
                if (indices_gt[b,n,p] not in indices_10[b,n]):
                    fn_count_10 = fn_count_10 + 1
                if (indices_gt[b,n,p] not in indices_25[b,n]):
                    fn_count_25 = fn_count_25 + 1
                if (indices_gt[b,n,p] not in indices_100[b,n]):
                    fn_count_100 = fn_count_100 + 1
            

            '''
            if (b==0) and (indices[b,n] not in indices_gt_true[b]):
                absolute_false_positive.append(indices[b,n])
            if (b==0) and (indices[b,n] not in indices_gt[b]):
                false_positive.append(indices[b,n])
            if (b==0) and (indices_gt_true[b,n] not in indices[b]):
                absolute_false_negative.append(indices_gt_true[b,n])
            if (b==0) and (indices_gt[b,n] not in indices[b]):
                false_negative.append(indices_gt[b,n])
            
    print("trained nearest 16 neigbours for packet 0:"+str(indices[0]))
    print("top 16 nearest neighbours: :"+str(indices_gt_true[0]))
    print("top nearest 48 neigbours:"+str(indices_gt[0]))
    print("false_positives for packet 0:"+str(false_positive))
    print("false_positives with loose constraint for packet 0:"+str(absolute_false_positive))
    print("false_negatives for packet 0:"+str(false_negative))
    print("false_negatives with loose constraint for packet 0:"+str(absolute_false_negative))

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
    '''
    print("acc_count:"+str(acc_count))
    print("fp_count:"+str(fp_count))
    ''' 
    print("The Recall_10: "+str(acc_count_10/(fn_count_10+acc_count_10)))
    print("The Recall_25: "+str(acc_count_25/(fn_count_25+acc_count_25)))
    print("The Recall_100: "+str(acc_count_100/(fn_count_100+acc_count_100)))
    save_name = os.path.join(checkpoint_dir,'best_n_trained.npy')

    np.save(save_name,indices_100[0])
    print("Done")


if __name__ == "__main__":
    for i in range(10):
        validate(i)
