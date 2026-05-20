import torch
import torch.nn as nn

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.runtime import compute_states, z_expectations_from_states
from TorchQML.encoding.amplitude import AmpEnc


def calcZvec(xb, theta, circ, spec, amp_enc):
    """Run the circuit and return one Z expectation per qubit."""
    psi = compute_states(xb, theta, circ, amp_enc=amp_enc)
    return z_expectations_from_states(psi, spec.num_qubits)


class VectorZLinear(nn.Module):
    """Z-expectation quantum model with a trainable linear readout."""

    def __init__(self, circ, spec, amp_enc=False):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.theta = nn.Parameter(
            0.1 * torch.randn(spec.tlen, device=DEVICE, dtype=DTYPE)
        )
        self.w = nn.Parameter(
            torch.randn(spec.num_qubits, device=DEVICE, dtype=torch.float32)
        )
        self.b = nn.Parameter(torch.zeros((), device=DEVICE, dtype=torch.float32))
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None

    def forward(self, xb):
        z_out = calcZvec(xb, self.theta, self.circ, self.spec, self.amp_enc)
        z_out = z_out.real.to(dtype=self.w.dtype)
        # The circuit produces features; the linear head keeps the readout easy to inspect.
        return (z_out @ self.w + self.b).real


__all__ = ["VectorZLinear", "calcZvec"]
