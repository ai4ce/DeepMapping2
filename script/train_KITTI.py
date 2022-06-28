import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import loss
from models import DeepMapping_KITTI
from dataset_loader import KITTI

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-e','--n_epochs',type=int,default=1000,help='number of epochs')
parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')
parser.add_argument('-n','--n_samples',type=int,default=35,help='number of sampled unoccupied points along rays')
parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
parser.add_argument('-d','--data_dir',type=str,default='../data/ActiveVisionDataset/',help='dataset path')
parser.add_argument('-t','--traj',type=str,default='2011_09_30_drive_0018_sync_full',help='trajectory file folder')
parser.add_argument('-m','--model', type=str, default=None,help='pretrained model name')
parser.add_argument('-i','--init', type=str, default=None,help='init pose')
parser.add_argument('--log_interval',type=int,default=10,help='logging interval of saving results')
parser.add_argument('-g', '--group', type=int, default=0, help='whether to group frames')
parser.add_argument('--group_size',type=int,default=8,help='group size')
parser.add_argument('--pairwise', action='store_true',
                    help='If present, use global consistency loss')
parser.add_argument('--resume', action='store_true',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--alpha', type=float, default=0.1, help='weight for chamfer loss')
parser.add_argument('--beta', type=float, default=0.1, help='weight for euclidean loss')

opt = parser.parse_args()

checkpoint_dir = os.path.join('../results/KITTI',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if opt.init is not None:
    print("loading initial pose from:", opt.init)
    init_pose_np = np.load(opt.init)
    init_pose_np = init_pose_np.astype("float32")
    init_pose = torch.from_numpy(init_pose_np)
else:
    init_pose = None

print('loading dataset')
dataset = KITTI(opt.data_dir, opt.traj, opt.voxel_size, init_pose=init_pose, 
        group=opt.group, group_size=opt.group_size, pairwise=opt.pairwise)
loader = DataLoader(dataset, batch_size=None)
# if opt.group:
#     group_sampler = GroupSampler(dataset.group_matrix)
#     train_loader = DataLoader(dataset,batch_size=opt.batch_size, shuffle=False, sampler=group_sampler, num_workers=8)
# else:
#     train_loader = loader
loss_fn = eval('loss.'+opt.loss)

print('creating model')
model = DeepMapping_KITTI(n_points=dataset.n_points, loss_fn=loss_fn,
    n_samples=opt.n_samples, alpha=opt.alpha, beta=opt.beta).to(device)
    
optimizer = optim.Adam(model.parameters(),lr=opt.lr)

if opt.model is not None:
    utils.load_checkpoint(opt.model,model,optimizer)

if opt.resume:
    resume_filename = checkpoint_dir + "model_best.pth"
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    starting_epoch = checkpoint['epoch']
    
    model.load_state_dict(saved_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    starting_epoch = 0

print('start training')
best_loss = 200
for epoch in range(starting_epoch, opt.n_epochs):
    training_loss= 0.0
    model.train()

    for index,(obs, valid_pt, init_global_pose, pairwise_pose) in enumerate(loader):
        obs = obs.to(device)
        valid_pt = valid_pt.to(device)
        init_global_pose = init_global_pose.to(device)
        pairwise_pose = pairwise_pose.to(device)
        loss = model(obs, valid_pt, init_global_pose, pairwise_pose)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    
    training_loss_epoch = training_loss/len(loader)

    if (epoch+1) % opt.log_interval == 0:
        print('[{}/{}], training loss: {:.4f}'.format(
            epoch+1,opt.n_epochs,training_loss_epoch))
    if training_loss_epoch < best_loss:
        print("lowest loss:", training_loss_epoch)
        best_loss = training_loss_epoch
        obs_global_est_np = []
        pose_est_np = []
        with torch.no_grad():
            model.eval()
            for index,(obs, valid_pt, init_global_pose, pairwise_pose) in enumerate(loader):
                obs = obs.to(device)
                valid_pt = valid_pt.to(device)
                init_global_pose = init_global_pose.to(device)
                pairwise_pose = pairwise_pose.to(device)
                model(obs, valid_pt, init_global_pose, pairwise_pose)

                obs_global_est = model.obs_global_est[0]
                pose_est = model.pose_est[0]
                obs_global_est_np.append(obs_global_est.cpu().detach().numpy())
                pose_est_np.append(pose_est.cpu().detach().numpy())
            
            pose_est_np = np.stack(pose_est_np)
            # if init_pose is not None:
            #    pose_est_np = utils.cat_pose_AVD(init_pose_np,pose_est_np)
            
            save_name = os.path.join(checkpoint_dir,'model_best.pth')
            utils.save_checkpoint(save_name,model,optimizer,epoch)

            obs_global_est_np = np.stack(obs_global_est_np)
            #kwargs = {'e':epoch+1}
            #valid_pt_np = dataset.valid_points.cpu().detach().numpy()

            save_name = os.path.join(checkpoint_dir,'obs_global_est.npy')
            np.save(save_name,obs_global_est_np)

            save_name = os.path.join(checkpoint_dir,'pose_global_est.npy')
            np.save(save_name,pose_est_np)

            utils.plot_global_pose(checkpoint_dir, epoch)
