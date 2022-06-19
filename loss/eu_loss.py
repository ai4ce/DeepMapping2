import torch

def euclidean_loss(input, target):
    return torch.norm(target - input, dim=-1).mean()