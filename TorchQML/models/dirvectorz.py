import torch
import torch.nn as nn
import torch.nn.functional as F

from TorchQML.core.config import DEVICE
from TorchQML.models.vectorz import calcZvec


class VectorZDirHam(nn.Module):
    """Z-vector readout with differentiable architecture weights per slot."""

    def __init__(self, circ, spec, n_slots, amp_enc=None):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.n_slots = n_slots
        self.amp_enc = amp_enc
        self.n_main = 2 * n_slots
        self.n_arch = 3 * n_slots
        self.theta_main = nn.Parameter(0.01 * torch.randn(self.n_main, device=DEVICE))
        self.arch_raw = nn.Parameter(torch.zeros(n_slots, 3, device=DEVICE))
        self.fc = nn.Linear(spec.num_qubits, 1)

    def arch_weights(self):
        beta = F.softplus(self.arch_raw) + 1e-4
        return beta / beta.sum(dim=1, keepdim=True)

    def effective_theta(self):
        return torch.cat([self.theta_main, self.arch_weights().reshape(-1)], dim=0)

    def print_arch(self):
        weights = self.arch_weights().detach().cpu()
        for idx in range(weights.shape[0]):
            vals = ", ".join(f"{value:.3f}" for value in weights[idx])
            print(f"slot {idx}: [{vals}]")

    def forward(self, xb):
        theta_eff = self.effective_theta()
        z_out = calcZvec(xb, theta_eff, self.circ, self.spec, self.amp_enc)
        return self.fc(z_out.real).squeeze(-1)


__all__ = ["VectorZDirHam"]
