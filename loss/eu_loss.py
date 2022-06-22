import torch

def euclidean_loss(input, target):
    return torch.nn.functional.smooth_l1_loss(input, target)