import torch
import torch.nn as nn

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.runtime import compute_states, z_expectations_from_states
from TorchQML.encoding.amplitude import AmpEnc


def singleZ(psi, target_qubit, num_qubits):
    expectations = z_expectations_from_states(psi.unsqueeze(0), num_qubits)
    return expectations[0, target_qubit]


def sumZ(psi, num_qubits):
    expectations = z_expectations_from_states(psi.unsqueeze(0), num_qubits)
    return expectations[0].sum()


def forward_sumZ(xb, theta, circ, amp_enc=None):
    states = compute_states(xb, theta, circ, amp_enc=amp_enc)
    return z_expectations_from_states(states, circ.num_qubits).sum(dim=1)


class SumZModel(nn.Module):
    """Sum of per-qubit Z expectations for a trainable circuit."""

    def __init__(self, circ, spec, amp_enc=False, scale: float = 0.01):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.scale = scale
        theta0 = scale * torch.randn(spec.tlen, device=DEVICE, dtype=DTYPE)
        self.theta = nn.Parameter(theta0)
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None

    def forward(self, xb):
        xb = xb.to(device=self.theta.device, dtype=self.theta.dtype)
        return forward_sumZ(xb, self.theta, self.circ, self.amp_enc)


__all__ = ["SumZModel", "forward_sumZ", "singleZ", "sumZ"]
