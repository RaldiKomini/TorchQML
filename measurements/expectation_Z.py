import torch
from functools import reduce
from TorchQML.config import DEVICE, DTYPE
from TorchQML.gates.gate import I, Z

     
import torch

def expectation_Z(state: torch.Tensor, index: int, n_qubits: int) -> torch.Tensor:
    """
    Returns a scalar tensor ⟨Z_index⟩.
    Assumes basis ordering consistent with kron_all([q0,...,q_{n-1}]),
    i.e., index=0 is the LEFTMOST (most significant) qubit.
    """
    dim = 1 << n_qubits
    if state.ndim != 1 or state.numel() != dim:
        raise ValueError(f"state must be 1D of length {dim} for n_qubits={n_qubits}")
    if not (0 <= index < n_qubits):
        raise ValueError("index out of range")

    probs = (state.conj() * state).real  # |ψ|^2, shape (2^n,)

    # bit of the target qubit in each basis index:
    bitpos = n_qubits - 1 - index        # because index 0 is LEFTMOST
    basis_ids = torch.arange(dim, device=state.device)
    bit = (basis_ids >> bitpos) & 1       # 0 or 1

    # Z contributes +1 when bit==0, -1 when bit==1  → signs = 1 - 2*bit
    signs = (1 - 2 * bit).to(probs.dtype)

    return (probs * signs).sum()          # scalar tensor (good for autograd)

__all__ = ["expectation_Z"]