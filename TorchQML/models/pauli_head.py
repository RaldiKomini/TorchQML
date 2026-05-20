import torch
import torch.nn as nn
from TorchQML.encoding.amplitude import AmpEnc
from TorchQML.core.config import DEVICE
from TorchQML.core.runtime import compute_states




def pauli_expectations(psi, nq):
    squeeze = False
    if psi.ndim == 1:
        psi = psi.unsqueeze(0)
        squeeze = True

    psi = psi.view(psi.shape[0], *([2] * nq))
    res = []

    for q in range(nq):
        psi_q = psi.movedim(q + 1, 1).reshape(psi.shape[0], 2, -1)
        a, b = psi_q[:, 0], psi_q[:, 1]

        ex = 2 * (a.conj() * b).real.sum(dim=1)
        ey = 2 * (a.conj() * b).imag.sum(dim=1)
        ez = (a.conj() * a - b.conj() * b).real.sum(dim=1)

        res.append(torch.stack([ex, ey, ez], dim=1))

    out = torch.stack(res, dim=1)
    return out.squeeze(0) if squeeze else out


def weighted_pauli_sum(psi, w, nq):
    """
    w: [nq, 3]  trainable weights
    returns: scalar
    """
    P = pauli_expectations(psi, nq)
    if P.ndim == 2:
        return (w * P).sum()
    return (w.unsqueeze(0) * P).sum(dim=(1, 2))


def forward_weighted_pauli(Xb, theta, w, circ, amp_enc = None):
    psi = compute_states(Xb, theta, circ, amp_enc=amp_enc)
    return weighted_pauli_sum(psi, w, circ.num_qubits).real


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
