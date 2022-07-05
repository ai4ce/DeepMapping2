import torch

def rgl_loss(rotation):
    return rotation.abs().mean()