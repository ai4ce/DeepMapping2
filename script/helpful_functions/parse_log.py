import os
import numpy as np
from scipy.io import savemat

path = "/home/cc/DeepMapping-plus/script/helpful_functions"
file_name_pre = "log_method_4"
file_name = file_name_pre+".txt"

myfile = open(os.path.join(path,file_name), "r")
recall_10 = []
recall_25 = []
recall_100 = []
while myfile:
    line  = myfile.readline()
    if line.split("Recall_10:")[0] == "The ":
        recall_10.append(float(line.split("Recall_10:")[1]))
    elif line.split("Recall_25:")[0] == "The ":
        recall_25.append(float(line.split("Recall_25:")[1]))
    
    if line == "":
        break

myfile.close()

index = list(np.arange(len(recall_10)))
print("len of index:"+str(len(index)))

mdict = {"recall_10":recall_10, "recall_25":recall_25, "index":index}
savemat(file_name_pre+".mat", mdict)
