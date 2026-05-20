import torch
import torch.nn as nn
import torch.nn.functional as F

from TorchQML.core.config import DEVICE
from TorchQML.core.runtime import compute_states
from TorchQML.encoding.amplitude import AmpEnc


def spooky_measure(psi, num_qubits):
    """Return per-qubit linear entropy for a batch of state vectors."""
    batch_size = psi.shape[0]
    psi = psi.view(batch_size, *([2] * num_qubits))

    entropies = []
    for qubit in range(num_qubits):
        perm = [0, qubit + 1] + [
            idx for idx in range(1, num_qubits + 1) if idx != qubit + 1
        ]
        psi_q = psi.permute(perm).reshape(batch_size, 2, -1)
        rho = torch.matmul(psi_q, psi_q.conj().transpose(1, 2))
        purity = torch.einsum("bij,bji->b", rho, rho)
        entropies.append(1.0 - purity.real)

    return torch.stack(entropies, dim=1)


class SpookyModel(nn.Module):
    """Entanglement-readout model based on per-qubit linear entropy."""

    def __init__(self, circ, spec, amp_enc=False):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.tau = nn.Parameter(torch.tensor(0.25))
        self.theta = nn.Parameter(
            torch.randn(spec.tlen, device=DEVICE, dtype=torch.float32)
        )
        self.w = nn.Parameter(
            torch.randn(spec.num_qubits, device=DEVICE, dtype=torch.float32)
        )
        self.b = nn.Parameter(torch.zeros((), device=DEVICE, dtype=torch.float32))
        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None

    def forward(self, xb):
        psi = compute_states(xb, self.theta, self.circ, amp_enc=self.amp_enc)
        entanglement = spooky_measure(psi, self.spec.num_qubits)
        score = entanglement @ self.w + self.b
        return score, entanglement

    def spooky_loss3(self, output, y):
        score, _ = output
        y_pm = 1 - 2 * y.float()
        return F.softplus(y_pm * (score - self.tau)).mean()


__all__ = ["SpookyModel", "spooky_measure"]
