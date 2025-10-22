from qml_lib.config import torch, DTYPE, DEVICE

def density_matrix(mystate):
    p1 = torch.outer(mystate, torch.conjugate(mystate))
    return p1