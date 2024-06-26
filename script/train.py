# Copyright (C) 2022-2023, NYU AI4CE Lab. All rights reserved.

import set_path
import os
import time
import argparse
import functools

print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import loss
from models import DeepMapping2
from dataset_loader import *
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test', help='experiment name')
parser.add_argument('-e', '--n_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('-l', '--loss', type=str, default='bce_ch', help='loss function')
parser.add_argument('-n', '--n_samples', type=int, default=35, help='number of sampled unoccupied points along rays')
parser.add_argument('-v', '--voxel_size', type=float, default=1, help='size of downsampling voxel grid')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--dataset', type=str, default="KITTI", help="Type of dataset to use")
parser.add_argument('-d', '--data_dir', type=str, default='../data/ActiveVisionDataset/', help='dataset path')
parser.add_argument('-t', '--traj', type=str, default='2011_09_30_drive_0018_sync_full', help='trajectory file folder')
parser.add_argument('-m', '--model', type=str, default=None, help='pretrained model name')
parser.add_argument('-i', '--init', type=str, default=None, help='path to initial pose')
parser.add_argument('-p', '--pairwise', type=str, default=None, help='path to pairwise pose')
parser.add_argument('--log_interval', type=int, default=10, help='logging interval of saving results')
parser.add_argument('--group_size', type=int, default=8, help='group size')
parser.add_argument('--resume', action='store_true',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--alpha', type=float, default=0.1, help='weight for chamfer loss')
parser.add_argument('--beta', type=float, default=0.1, help='weight for euclidean loss')
parser.add_argument('--optimizer', type=str, default="Adam", help="The optimizer to use")
parser.add_argument('--amp', default=False, action='store_true', help="Toggle auto mixed precision")
parser.add_argument('-r', '--rotation', type=str, default="quaternion", help="The rotation representation to use")

opt = parser.parse_args()

checkpoint_dir = os.path.join('../results/' + opt.dataset, opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(os.path.join(checkpoint_dir, "pose_ests")):
    os.makedirs(os.path.join(checkpoint_dir, "pose_ests"))
utils.save_opt(checkpoint_dir, opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# opt.init: INIT=$DATA_DIR/$TRAJ/prior/init_pose.npy
# init_pose.npy should be an Nx6 numpy array, where N is the number of frames.
# Each row is the initial pose of a frame represented by x, y, z, row, pitch, yaw.
# 把initial pose转换成tensor
init_pose_np = np.load(opt.init).astype("float32")
init_pose = torch.from_numpy(init_pose_np)
pairwise_pose = np.load(opt.pairwise).astype("float32")

print('loading dataset')
if opt.dataset == "KITTI":
    train_dataset = Kitti(opt.data_dir, opt.traj, opt.voxel_size, init_pose=init_pose, group_size=opt.group_size,
                          pairwise_pose=pairwise_pose)
    eval_dataset = KittiEval(train_dataset)
    # eval_dataset大致与train_dataset相同，但是不包含gt_pose

elif opt.dataset == "NCLT" or "Nebula":
    train_dataset = Nclt(opt.data_dir, opt.traj, opt.voxel_size, init_pose=init_pose, group_size=opt.group_size,
                         pairwise_pose=pairwise_pose)
    eval_dataset = NcltEval(train_dataset)
else:
    assert 0, "Unsupported dataset"


train_loader = DataLoader(train_dataset, batch_size=None, num_workers=4, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=64, num_workers=4)
# loss function is bce_ch_eu
loss_fn = eval('loss.' + opt.loss)

if opt.rotation not in ['quaternion', 'euler_angle', '6d']:
    print("Unsupported rotation representation")
    assert ()

print('creating model')
model = DeepMapping2(n_points=train_dataset.n_points, loss_fn=loss_fn,
                     n_samples=opt.n_samples, alpha=opt.alpha, beta=opt.beta, rotation_representation=opt.rotation).to(
    device)

if opt.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
else:
    print("Unsupported optimizer")
    assert ()

scaler = torch.cuda.amp.GradScaler()

if opt.model is not None:
    utils.load_checkpoint(opt.model, model, optimizer)

if opt.resume:
    resume_filename = os.path.join(checkpoint_dir, "model_best.pth")
    print("Resuming From ", resume_filename)
    starting_epoch = utils.load_checkpoint(resume_filename, model, optimizer)
else:
    starting_epoch = 0

print('start training')
best_loss = 20
training_losses = []
bce_losses, ch_losses, eu_losses = [], [], []
trans_ates = []
rot_ates = []
for epoch in range(starting_epoch, opt.n_epochs):
    training_loss = 0
    bce_loss = 0
    ch_loss = 0
    eu_loss = 0
    model.train()
    time_start = time.time()
    for index, (obs, valid_pt, init_global_pose, pairwise_pose) in enumerate(train_loader):
        obs = obs.to(device)
        valid_pt = valid_pt.to(device)
        init_global_pose = init_global_pose.to(device)
        pairwise_pose = pairwise_pose.to(device)
        if opt.amp:
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                loss, bce, ch, eu = model(obs, init_global_pose, valid_pt, pairwise_pose)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
            # optimizer.step()
        else:
            loss, bce, ch, eu = model(obs, init_global_pose, valid_pt, pairwise_pose)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss += loss.item()
        bce_loss += bce
        ch_loss += ch
        if loss == "bce_ch_eu" or loss == "pose":
            eu_loss += eu

    time_end = time.time()
    # print(model.parameters().grad)
    print("Training time: {:.2f}s".format(time_end - time_start))
    training_loss_epoch = training_loss / len(train_loader)
    bce_epoch = bce_loss / len(train_loader)
    ch_epoch = ch_loss / len(train_loader)
    eu_epoch = eu_loss / len(train_loader)
    training_losses.append(training_loss_epoch)
    bce_losses.append(bce_epoch)
    ch_losses.append(ch_epoch)
    eu_losses.append(eu_epoch)

    print('[{}/{}], training loss: {:.4f}'.format(epoch + 1, opt.n_epochs, training_loss_epoch))
    obs_global_est_np = []
    pose_est_np = []
    with torch.no_grad():
        model.eval()
        for index, (obs, init_global_pose) in enumerate(eval_loader):
            obs = obs.to(device)
            init_global_pose = init_global_pose.to(device)
            model(obs, init_global_pose)

            obs_global_est = model.obs_global_est
            pose_est = model.pose_est
            obs_global_est_np.append(obs_global_est.cpu().detach().numpy())
            pose_est_np.append(pose_est.cpu().detach().numpy())

    pose_est_np = np.concatenate(pose_est_np)

    save_name = os.path.join(checkpoint_dir, "pose_ests", str(epoch + 1))
    np.save(save_name, pose_est_np)

    utils.plot_global_pose(checkpoint_dir, opt.dataset, epoch + 1, rotation_representation=opt.rotation)

    try:
        trans_ate, rot_ate = utils.compute_ate(pose_est_np, train_dataset.gt_pose, rotation_representation=opt.rotation)
    except np.linalg.LinAlgError:
        print("SVD did not converge, using ATE from last epoch.")
        trans_ate = trans_ates[-1]
        rot_ate = rot_ates[-1]
    print("Translation ATE:", trans_ate)
    print("Rotation ATE", rot_ate)
    trans_ates.append(trans_ate)
    rot_ates.append(rot_ate)
    utils.plot_curve(trans_ates, "translation_ate", checkpoint_dir)
    utils.plot_curve(rot_ates, "rotation_ate", checkpoint_dir)
    utils.plot_loss(training_losses, bce_losses, ch_losses, eu_losses, "training_loss", checkpoint_dir)

    if training_loss_epoch < best_loss:
        print("lowest loss:", training_loss_epoch)
        best_loss = training_loss_epoch

        # Visulize global point clouds
        obs_global_est_np = np.concatenate(obs_global_est_np)
        save_name = os.path.join(checkpoint_dir, 'obs_global_est.npy')
        np.save(save_name, obs_global_est_np)

        # Save checkpoint
        save_name = os.path.join(checkpoint_dir, 'model_best.pth')
        utils.save_checkpoint(save_name, model, optimizer, epoch)

    print()

training_losses = np.array(training_losses)
np.save(os.path.join(checkpoint_dir, "loss.npy"), training_losses)
