import torch
from TorchQML.core.config import DEVICE, DTYPE


class AmpEnc:
    """Amplitude encoder that pads or truncates features to 2**nq."""

    def __init__(self, nq):
        """Set the target number of qubits and state dimension."""
        self.num_qubits = nq
        self.dim = 1 << self.num_qubits

    def __call__(self, x):
        """Return normalized complex amplitudes for a vector or batch."""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device=DEVICE)

        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        elif x.dim() == 2:
            squeeze = False
        else:
            raise ValueError("X has wrong dim")

        if x.shape[-1] >= self.dim:
            x = x[..., : self.dim]
        else:
            pad_shape = (*x.shape[:-1], self.dim - x.shape[-1])
            pad = torch.zeros(pad_shape, device=DEVICE, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)

        if not torch.is_complex(x):
            x = x.to(torch.complex64)

        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        psi = (x / (norm + 1e-12)).to(dtype=DTYPE)
        return psi.squeeze(0) if squeeze else psi


__all__ = ["AmpEnc"]
