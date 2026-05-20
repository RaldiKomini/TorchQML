import torch
import torch.nn as nn

from TorchQML.core.config import DEVICE
from TorchQML.core.runtime import compute_states
from TorchQML.encoding.amplitude import AmpEnc
from TorchQML.models.pauli_head import pauli_expectations


def forward_vector_pauli(xb, theta, circ, amp_enc=None):
    psi = compute_states(xb, theta, circ, amp_enc=amp_enc)
    measurements = pauli_expectations(psi, circ.num_qubits)
    return measurements.reshape(measurements.shape[0], -1)


class PauliVectorModel(nn.Module):
    """Pauli-expectation feature vector with a trainable linear readout."""

    def __init__(self, circ, spec, scale_theta=0.01, scale_w=0.1, amp_enc=False):
        super().__init__()
        self.circ = circ
        self.spec = spec
        theta0 = scale_theta * torch.randn(spec.tlen, device=DEVICE, dtype=torch.float32)
        self.theta = nn.Parameter(theta0)
        self.w = nn.Parameter(
            scale_w * torch.randn(spec.num_qubits * 3, device=DEVICE, dtype=torch.float32)
        )
        self.b = nn.Parameter(torch.zeros((), device=DEVICE, dtype=torch.float32))
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None

    def forward(self, xb):
        xb = xb.to(device=self.theta.device, dtype=self.theta.dtype)
        pauli_features = forward_vector_pauli(xb, self.theta, self.circ, self.amp_enc)
        return (pauli_features @ self.w + self.b).real


__all__ = ["PauliVectorModel", "forward_vector_pauli"]
