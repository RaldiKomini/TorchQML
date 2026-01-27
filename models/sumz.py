import torch
import torch.nn as nn
from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.gates import Z, I
from TorchQML.encoding.amplitude import AmpEnc
import torch.nn.functional as F

def singleZ(psi, tg, nq):
    op = None
    for i in range(nq):
        m = Z.matrix if i == tg else I.matrix
        m = m.to(psi.device, psi.dtype)
        op = m if op is None else torch.kron(op, m)

    return (torch.conj(psi) @ (op @ psi)).real


def sumZ(psi, nq):
    allZ = torch.stack([singleZ(psi, i, nq) for i in range(nq)])
    return allZ.sum()

def forward_sumZ(Xb, t, circ, amp_enc = None):
    nq = circ.num_qubits
    batch_S = []
    for x in Xb:
        if amp_enc == None:
            psi = circ.apply_to(x=x, theta=t)
        else:
            psi_init = amp_enc(x)
            psi = circ.apply_to(state = psi_init, x=x, theta=t)
        psi = psi / (psi.norm() + 1e-12)

        S = sumZ(psi, nq)      
        batch_S.append(S)
    S = torch.stack(batch_S, dim=0) 
    return S


class SumZModel(nn.Module):
    def __init__(self, circ, spec, amp_enc = False, scale:float = 0.01):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.scale = scale

        theta0 = scale * torch.randn(spec.tlen, device=DEVICE, dtype=DTYPE)
        self.theta = nn.Parameter(theta0)
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None

    def forward(self, xb):
        xb = xb.to(device = self.theta.device, dtype = self.theta.dtype)

        return forward_sumZ(xb, self.theta, self.circ, self.amp_enc)
    

    

__all__ = ["SumZModel"]
