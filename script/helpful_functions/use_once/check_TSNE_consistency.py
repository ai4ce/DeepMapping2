import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

check1 =  "/home/cc/0_Unsupervised-PointNetVlad_time_seq2/results/database99.npy"
check2 =  "/home/cc/6_Supervised-PointNetVlad/results/database33.npy"

data_1 = np.load(check1)
data_2 = np.load(check2)

data_1 = data_1[0]
data_2 = data_2[:2048]

nbrs_1 = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(data_1)
distance_1, indice_1 = nbrs_1.kneighbors(data_1)

nbrs_2 = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(data_2)
distance_2, indice_2 = nbrs_2.kneighbors(data_2)

ran = [-2,-1,0,1,2]

count_1 = 0
count_2 = 0
for i in range(indice_1.shape[0]):
    print("i:"+str(i))
    for j in ran:
        if (i+j>=0) and (i+j<indice_1.shape[0]):
            if (i+j) in indice_1[i]:
                count_1 = count_1 + 1

for i in range(indice_2.shape[0]):
    print("i:"+str(i))
    for j in ran:
        if (i+j>=0) and (i+j<indice_2.shape[0]):
            if (i+j) in indice_2[i]:
                count_2 = count_2 + 1

print("count_1:"+str(count_1))
print("count_2:"+str(count_2))
print("Done!!!!!")
'''
print("distance_1:"+str(distance_1[0]))
print("indice_1:"+str(indice_1[0]))
print("distance_1:"+str(distance_1[1]))
print("indice_1:"+str(indice_1[1]))
print("distance_1:"+str(distance_1[2]))
print("indice_1:"+str(indice_1[2]))
print("distance_1:"+str(distance_1[3]))
print("indice_1:"+str(indice_1[3]))
print("distance_1:"+str(distance_1[4]))
print("indice_1:"+str(indice_1[4]))

print("distance_2:"+str(distance_2.shape))
'''
