import torch
import torch.nn as nn
from TorchQML.encoding.amplitude import AmpEnc
from TorchQML.core.config import DEVICE, DTYPE




def pauli_expectations(psi, nq):
    psi = psi.view([2] * nq)
    res = []

    for q in range(nq):
        psi_q = psi.movedim(q, 0).reshape(2, -1)
        a, b = psi_q[0], psi_q[1]

        ex = 2 * (a.conj() * b).real.sum()
        ey = 2 * (a.conj() * b).imag.sum()
        ez = (a.conj() * a - b.conj() * b).real.sum()

        res.append(torch.stack([ex, ey, ez]))

    return torch.stack(res)


def weighted_pauli_sum(psi, w, nq):
    """
    w: [nq, 3]  trainable weights
    returns: scalar
    """
    P = pauli_expectations(psi, nq)
    return (w * P).sum()


def forward_weighted_pauli(Xb, theta, w, circ, amp_enc = None):
    nq = circ.num_qubits
    batch_S = []

    for x in Xb:
        if amp_enc == None:
            psi = circ.apply_to(x=x, theta=theta)
        else:
            psi_init = amp_enc(x)
            psi = circ.apply_to(state = psi_init, x=x, theta=theta)
        psi = psi / (psi.norm() + 1e-12)

        S = weighted_pauli_sum(psi, w, nq)
        batch_S.append(S)

    return torch.stack(batch_S).real


class PauliHeadModel(nn.Module):
    def __init__(self, circ, spec, scale_theta=0.01, scale_w=0.1, amp_enc = False):
        super().__init__()
        self.circ = circ
        self.spec = spec

        # circuit parameters
        theta0 = scale_theta * torch.randn(spec.tlen, device=DEVICE, dtype=torch.float32)
        self.theta = nn.Parameter(theta0)

        # observable parameters: one weight per (qubit, Pauli)
        w0 = scale_w * torch.ones(circ.num_qubits, 3, device=DEVICE, dtype=torch.float32)
        self.w = nn.Parameter(w0)
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None
            

    def forward(self, xb):
        xb = xb.to(device=self.theta.device, dtype=self.theta.dtype)
        return forward_weighted_pauli(xb, self.theta, self.w, self.circ, self.amp_enc)
    

__all__ = ["PauliHeadModel"]
