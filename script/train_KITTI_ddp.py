import set_path
import os
import time
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import loss
from models import DeepMapping2
from dataset_loader import Kitti, KittiEval
from tqdm import tqdm
from time import sleep


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def ddp_func(rank, world_size, opt):
    utils.setup(rank, world_size)
    checkpoint_dir = os.path.join('../results/KITTI',opt.name)
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(os.path.join(checkpoint_dir, "pose_ests")):
            os.makedirs(os.path.join(checkpoint_dir, "pose_ests"))
        utils.save_opt(checkpoint_dir,opt)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.init:
        if rank == 0:
            print("loading initial pose from:", opt.init)
        init_pose_np = np.load(opt.init)
        init_pose_np = init_pose_np.astype("float32")
        init_pose = torch.from_numpy(init_pose_np)
    else:
        init_pose = None

    if opt.pairwise:
        pairwise_path = os.path.join(checkpoint_dir, "pose_pairwise.npy")
        if rank == 0:
            print("loading pairwise pose from", pairwise_path)
        pairwise_pose = np.load(pairwise_path)
    else:
        pairwise_pose = None

    # sleep(int(rank) * 5) # prevent different processes reading same file
    if rank == 0:
        print('loading dataset')
        train_dataset = Kitti(opt.data_dir, opt.traj, opt.voxel_size, init_pose=init_pose, 
            group=opt.group, group_size=opt.group_size, pairwise=opt.pairwise, pairwise_pose=pairwise_pose)
    else:
        train_dataset = Kitti(opt.data_dir, opt.traj, opt.voxel_size, init_pose=init_pose, 
            group=opt.group, group_size=opt.group_size, pairwise=opt.pairwise, pairwise_pose=pairwise_pose, use_tqdm=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=None, sampler=train_sampler)
    if rank == 0:
        eval_dataset = KittiEval(train_dataset)
        eval_loader = DataLoader(eval_dataset, batch_size=128, num_workers=8)

    loss_fn = eval('loss.'+opt.loss)
    
    if opt.rotation not in ['quaternion','euler_angle']:
        print("Unsupported rotation representation")
        assert()
    
    if rank == 0:
        print('creating model')
    model = DeepMapping2(n_points=train_dataset.n_points, loss_fn=loss_fn,
        n_samples=opt.n_samples, alpha=opt.alpha, beta=opt.beta, rotation_representation=opt.rotation).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    if opt.optimizer == "Adam":
        optimizer = optim.Adam(ddp_model.parameters(),lr=opt.lr)
    elif opt.optimizer == "SGD":
        optimizer = optim.SGD(ddp_model.parameters(), lr=opt.lr, momentum=0.9)
    else:
        print("Unsupported optimizer")
        assert()

    scaler = torch.cuda.amp.GradScaler()

    if opt.resume:
        save_name = os.path.join(checkpoint_dir, "model_best.pth")
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        state = torch.load(save_name, map_location=map_location)
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])

        ddp_model.load_state_dict(state["state_dict"])
        starting_epoch = state["epoch"]
        print('model loaded from {}'.format(save_name))
    else:
        starting_epoch = 0
    
    if rank == 0:
        print('start training')
        training_losses = []
        trans_ates = []
        rot_ates = []
    best_loss = 10
        
    for epoch in range(starting_epoch, opt.n_epochs):
        training_loss= 0.0
        ddp_model.train()

        time_start = time.time()
        for index,(obs, valid_pt, init_global_pose, pairwise_pose) in enumerate(train_loader):
            obs = obs.to(rank)
            valid_pt = valid_pt.to(rank)
            init_global_pose = init_global_pose.to(rank)
            pairwise_pose = pairwise_pose.to(rank)
            with torch.autocast("cuda"):
                loss, _, _, _ = ddp_model(obs, init_global_pose, valid_pt, pairwise_pose)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # optimizer.step()

            training_loss += loss.item()
        
        time_end = time.time()
        if rank == 0:
            print("Training time: {:.2f}s".format(time_end - time_start))
            training_loss_epoch = training_loss/len(train_loader)
            training_losses.append(training_loss_epoch)

            print('[{}/{}], training loss: {:.4f}'.format(epoch+1,opt.n_epochs,training_loss_epoch))
            obs_global_est_np = []
            pose_est_np = []
            with torch.no_grad():
                ddp_model.eval()
                for index,(obs, init_global_pose) in enumerate(eval_loader):
                    obs = obs.to(rank)
                    init_global_pose = init_global_pose.to(rank)
                    ddp_model(obs, init_global_pose)

                    obs_global_est = ddp_model.module.obs_global_est
                    pose_est = ddp_model.module.pose_est
                    obs_global_est_np.append(obs_global_est.cpu().detach().numpy())
                    pose_est_np.append(pose_est.cpu().detach().numpy())
                
            pose_est_np = np.concatenate(pose_est_np)

            save_name = os.path.join(checkpoint_dir, "pose_ests", str(epoch+1))
            np.save(save_name,pose_est_np)

            utils.plot_global_pose(checkpoint_dir, "KITTI", epoch+1, rotation_representation=opt.rotation)

            trans_ate, rot_ate = utils.compute_ate(pose_est_np, train_dataset.gt_pose, rotation_representation=opt.rotation)
            trans_ates.append(trans_ate)
            rot_ates.append(rot_ate)
            # utils.plot_curve(trans_ates, "translation_ate", checkpoint_dir)
            # utils.plot_curve(rot_ates, "rotation_ate", checkpoint_dir)
            # utils.plot_curve(training_losses, "training_loss", checkpoint_dir)

            if training_loss_epoch < best_loss:
                print("lowest loss:", training_loss_epoch)
                best_loss = training_loss_epoch

                # Visulize global point clouds
                obs_global_est_np = np.concatenate(obs_global_est_np)
                save_name = os.path.join(checkpoint_dir,'obs_global_est.npy')
                np.save(save_name,obs_global_est_np)

                # Save checkpoint
                save_name = os.path.join(checkpoint_dir,'model_best.pth')
                utils.save_checkpoint(save_name, ddp_model, optimizer, epoch)
    utils.cleanup()


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
parser.add_argument('--optimizer', type=str, default="Adam", help="The optimizer to use")
parser.add_argument('-r', '--rotation', type=str, default="quaternion", help="The rotation representation to use")

opt = parser.parse_args()

if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(ddp_func,
                args=(world_size, opt),
                nprocs=world_size,
                join=True)
