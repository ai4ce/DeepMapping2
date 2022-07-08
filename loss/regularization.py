import torch

def rgl_loss(l_net_out):
    xyz = l_net_out[:, :3]
    distance = torch.norm(xyz, dim=-1)
    theta = l_net_out[:, -1]
    return 0.01 * distance.mean() + 0.02 * theta.abs().mean()