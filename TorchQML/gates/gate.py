import math
from functools import lru_cache
import torch
from TorchQML.core.config import DTYPE, DEVICE


class Gate:
    """Quantum gate wrapper around a dense PyTorch matrix."""

    def __init__(self, matrix: torch.Tensor, name: str = "Not specified", params=None):
        """Store the matrix on the configured device and dtype."""
        self.matrix = matrix.to(device=DEVICE, dtype=DTYPE)
        self.name = name
        self.params = list(params) if params is not None else []

    def __matmul__(self, other: "Gate") -> "Gate":
        """Compose two gates by matrix multiplication."""
        if not isinstance(other, Gate):
            raise TypeError("Not a gate!")
        new_matrix = self.matrix @ other.matrix
        new_name = f"({self.name} @ {other.name})"
        new_params = self.params + other.params
        return Gate(new_matrix, new_name, new_params)

    def __repr__(self):
        return f"Gate name {self.name}, params = {self.params}"

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the gate to a state vector or batch of state vectors."""
        state = state.to(device=self.matrix.device, dtype=self.matrix.dtype)
        return torch.matmul(self.matrix, state.unsqueeze(-1)).squeeze(-1)

    def dagger(self) -> "Gate":
        """Return the Hermitian conjugate gate."""
        new_matrix = self.matrix.mH
        return Gate(new_matrix, f"{self.name}.dagger()", self.params)

    def __invert__(self) -> "Gate":
        return self.dagger()


# --- Fixed 1-qubit gates ---
Gate.I = Gate(torch.eye(2, dtype=DTYPE, device=DEVICE), "I")
Gate.X = Gate(torch.tensor([[0, 1], [1, 0]], dtype=DTYPE, device=DEVICE), "X")
Gate.Y = Gate(torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE, device=DEVICE), "Y")
Gate.Z = Gate(torch.tensor([[1, 0], [0, -1]], dtype=DTYPE, device=DEVICE), "Z")

_s = 1.0 / math.sqrt(2.0)
Gate.H = Gate(torch.tensor([[1, 1], [1, -1]], dtype=DTYPE, device=DEVICE) * _s, "H")

Gate.S = Gate(torch.tensor([[1, 0], [0, 1j]], dtype = DTYPE, device=DEVICE), "S")
Gate.Sdg = Gate(torch.tensor([[1, 0], [0, -1j]], dtype=DTYPE, device=DEVICE), "Sdg")

_t = complex(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
_tdg = complex(math.cos(math.pi / 4.0), -math.sin(math.pi / 4.0))

Gate.T = Gate(torch.tensor([[1, 0], [0, _t]], dtype=DTYPE, device=DEVICE), "T")
Gate.Tdg = Gate(torch.tensor([[1, 0], [0, _tdg]], dtype=DTYPE, device=DEVICE), "Tdg")

def CNOT(nq, ctr, trg):
    """Build an `nq`-qubit CNOT gate."""
    assert 0 <= ctr < nq
    assert 0 <= trg  < nq
    assert ctr != trg
    return Gate(_cnot_matrix(nq, ctr, trg), f"CNOT(c={ctr},t={trg})")


@lru_cache(maxsize=None)
def _cnot_matrix(nq: int, ctr: int, trg: int) -> torch.Tensor:
    """Return the dense CNOT matrix for the given qubits."""
    dim = 1 << nq
    basis = torch.arange(dim, device=DEVICE)
    flipped = basis.clone()
    # Keep the original TorchQML bit convention: qubit 0 is the least-significant bit.
    control_on = ((basis >> ctr) & 1).bool()
    flipped[control_on] = flipped[control_on] ^ (1 << trg)

    mat = torch.zeros((dim, dim), dtype=DTYPE, device=DEVICE)
    mat[flipped, basis] = 1.0
    return mat



I = Gate.I
X = Gate.X
Y = Gate.Y
Z = Gate.Z
H = Gate.H
S = Gate.S
Sdg = Gate.Sdg
T = Gate.T
Tdg = Gate.Tdg




__all__ = [
    "Gate",
    "I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "CNOT"
]
