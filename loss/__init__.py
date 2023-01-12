from .chamfer_dist import chamfer_loss
from .bce_loss import bce
from .eu_loss import euclidean_loss
import torch


def bce_ch(pred, targets, obs_global, valid_obs=None, bce_weight=None, seq=2, gamma=0.1):
    """
    pred: <Bx(n+1)Lx1>, occupancy probabiliry from M-Net
    targets: <Bx(n+1)Lx1>, occupancy label
    obs_global: <BxLxk> k = 2,3, global point cloud
    valid_obs: <BxL>, indices of valid point (0/1), 
               invalid points are not used for computing chamfer loss
    bce_weight: <Bx(n+1)Lx1>, weight for each point in computing bce loss
    """
    bce_loss = bce(pred, targets, bce_weight)
    ch_loss = chamfer_loss(obs_global, valid_obs, seq)
    loss = gamma * bce_loss + (1 - gamma) * ch_loss
    return loss, bce_loss.item(), ch_loss.item(), None


def bce_ch_eu(pred, targets, obs_global, src, dst, valid_obs=None, bce_weight=None, seq=2, alpha=0.1, beta=0.1):
    bce_loss = bce(pred, targets, bce_weight)
    ch_loss = chamfer_loss(obs_global, valid_obs, seq)
    eu_loss = euclidean_loss(src, dst)
    loss = (1-alpha-beta) * bce_loss + alpha * ch_loss + beta * eu_loss
    return loss, bce_loss.item(), ch_loss.item(), eu_loss.item()


def pose(pred, targets, obs_global, src_t, dst_t, src_R, dst_R, valid_obs=None, bce_weight=None, seq=2, alpha=0.1, beta=0.1):
    bce_loss = bce(pred, targets, bce_weight)
    ch_loss = chamfer_loss(obs_global, valid_obs, seq)
    t_loss = euclidean_loss(src_t, dst_t)
    id = torch.eye(3).unsqueeze(0).cuda()
    # print(src_R.shape)
    # print(dst_R.shape)
    # print(id.shape)
    # assert()
    r_loss = torch.norm((torch.bmm(src_R.transpose(2, 1).contiguous(), dst_R) - id), dim=(1, 2))
    # print(torch.bmm(src_R.transpose(2, 1).contiguous(), dst_R).shape)
    # print(r_loss.shape)
    eu_loss = t_loss + 5 * r_loss.mean()
    loss = (1-alpha-beta) * bce_loss + alpha * ch_loss + beta * eu_loss
    return loss, bce_loss.item(), ch_loss.item(), eu_loss.item()
