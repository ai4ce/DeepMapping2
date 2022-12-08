# DeepMapping2

On going project of DeepMapping++

## Environment requirements:
* Python=3.9
* PyTorch=1.11.0
* Open3d=0.15.2

To install the packages, you can run
`pip install -r requirements.txt`

## Run the code
### Pre-processing
Run the pre-processing by excecuting the script

`./script/run_pre_processing.sh`

When it finishes, it will generate `pose_est_icp.npy` and `pose_pairwise.npy` in the corresponding
results folder.

### DeepMapping++
Run deepmapping++ training by excecuting the script

`./script/run_train_KITTI.sh`

During training, curves for ates and losses, and trajectory visualization will be generated to the corresponding result folder.
