import torch

from TorchQML.core.config import DEVICE, DTYPE


_Z_SIGN_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def same_tensor(x1, x2) -> bool:
    """Check whether two tensor variables share the same storage."""
    return x1 is x2 or (
        torch.is_tensor(x1)
        and torch.is_tensor(x2)
        and x1.shape == x2.shape
        and x1.device == x2.device
        and x1.dtype == x2.dtype
        and x1.data_ptr() == x2.data_ptr()
    )


def compute_states(xb, theta, circ, amp_enc=None, normalize: bool = True) -> torch.Tensor:
    """Run a circuit over a batch and return normalized state vectors."""
    xb = xb.to(device=DEVICE)
    theta = theta.to(device=DEVICE, dtype=DTYPE)
    state = amp_enc(xb) if amp_enc is not None else None
    psi = circ.apply_to(state=state, x=xb, theta=theta)
    if psi.ndim == 1:
        psi = psi.unsqueeze(0)
    if normalize:
        denom = torch.linalg.vector_norm(psi, dim=1, keepdim=True).clamp_min(1e-12)
        psi = psi / denom
    return psi


def fidelity_kernel_from_states(psi_a: torch.Tensor, psi_b: torch.Tensor | None = None) -> torch.Tensor:
    """Return the squared inner-product kernel between two state batches."""
    psi_b = psi_a if psi_b is None else psi_b
    gram = psi_a @ psi_b.conj().T
    return gram.abs().square().real


def z_expectations_from_states(psi: torch.Tensor, nq: int) -> torch.Tensor:
    """Compute per-qubit Z expectations from state amplitudes."""
    probs = (psi.conj() * psi).real
    # The sign matrix is independent of the circuit, so it is reused across batches.
    signs = _z_sign_matrix(nq, probs.device, probs.dtype)
    return probs @ signs


def _z_sign_matrix(nq: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the +/- signs that map basis probabilities to Z expectations."""
    key = (nq, device.type, device.index)
    if key not in _Z_SIGN_CACHE:
        dim = 1 << nq
        idx = torch.arange(dim, device=device)
        bitpos = torch.arange(nq - 1, -1, -1, device=device)
        bits = ((idx.unsqueeze(1) >> bitpos) & 1).to(torch.float32)
        _Z_SIGN_CACHE[key] = 1.0 - 2.0 * bits
    return _Z_SIGN_CACHE[key].to(dtype=dtype)


__all__ = [
    "compute_states",
    "fidelity_kernel_from_states",
    "same_tensor",
    "z_expectations_from_states",
]
