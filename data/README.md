## Download
Please download the data from [Google Drive](https://drive.google.com/drive/folders/1jWfpmCAXaO9122mdkAuSV-yq_2DanPRh?usp=share_link) and put it in the `data` folder.

## Orgnize
The data should be organized as follows:

```
data
|-- KITTI # Type of dataset
|   |-- 0018 # Trajectory number
|   |   |-- pcd # Point cloud data
|   |   |   |-- 000000.pcd
|   |   |   |-- 000001.pcd
|   |   |   |-- ...
|   |   |-- prior # Prior data
|   |   |   |-- group_matrix.npy # Group matrix
|   |   |   |-- init_pose.npy # Initial pose
|   |   |   |-- pairwise.npy # Pairwise registration
|   |-- 0027 
|   |-- ...
|-- NCLT
|-- NeBuLa
|-- ...
```
## Customize
You can also use a different initial pose and pairwise registration by replacing `init_pose.npy` and `pairwise.npy` with your own files.

`init_pose.npy` should be an Nx6 numpy array, where N is the number of frames. Each row is the initial pose of a frame represented by `x, y, z, row, pitch, yaw`.

`pairwise.npy` should be an NxMx6 numpy array, where N is the number of frames, M is the number of neighbors of each frame. This should correspond to the shape of `group_matrix.npy`. The last dimension is the pairwise transformation from the neighbor to the center represented by `x, y, z, row, pitch, yaw`.
