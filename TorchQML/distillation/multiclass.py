import torch
import torch.nn as nn

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.runtime import compute_states, z_expectations_from_states
from TorchQML.encoding.amplitude import AmpEnc
from TorchQML.models.pauli_head import pauli_expectations


class VectorZMultiClass(nn.Module):
    """Multiclass linear head over per-qubit Z expectations."""

    def __init__(self, circ, spec, num_classes: int, amp_enc: bool = False):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.num_classes = num_classes
        self.theta = nn.Parameter(
            0.1 * torch.randn(spec.tlen, device=DEVICE, dtype=DTYPE)
        )
        self.weight = nn.Parameter(
            torch.randn(spec.num_qubits, num_classes, device=DEVICE, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.zeros(num_classes, device=DEVICE, dtype=torch.float32))
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None

    def forward(self, xb):
        psi = compute_states(xb, self.theta, self.circ, amp_enc=self.amp_enc)
        zvec = z_expectations_from_states(psi, self.spec.num_qubits).real.to(self.weight.dtype)
        return zvec @ self.weight + self.bias


class PauliHeadMultiClass(nn.Module):
    """Multiclass linear head over X/Y/Z expectations for every qubit."""

    def __init__(self, circ, spec, num_classes: int, amp_enc: bool = False):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.num_classes = num_classes
        self.theta = nn.Parameter(
            0.1 * torch.randn(spec.tlen, device=DEVICE, dtype=DTYPE)
        )
        self.weight = nn.Parameter(
            torch.randn(spec.num_qubits * 3, num_classes, device=DEVICE, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.zeros(num_classes, device=DEVICE, dtype=torch.float32))
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None

    def forward(self, xb):
        psi = compute_states(xb, self.theta, self.circ, amp_enc=self.amp_enc)
        features = pauli_expectations(psi, self.spec.num_qubits).reshape(xb.shape[0], -1)
        return features.real.to(self.weight.dtype) @ self.weight + self.bias


__all__ = ["PauliHeadMultiClass", "VectorZMultiClass"]
