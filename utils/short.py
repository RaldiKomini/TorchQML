import torch

def tensor(x, **args):
    return torch.tensor(x, dtype= torch.cfloat, **args)