import math
import torch
from TorchQML.core.config import DTYPE, DEVICE


#A quantum gate: stores its matrix, name, and parameters (for parametric gates).
class Gate: 
    def __init__(self, matrix: torch.Tensor, name: str = "Not specified", params=None):
        self.matrix = matrix.to(device=DEVICE, dtype=DTYPE)
        self.name = name
        self.params = list(params) if params is not None else []

    def __matmul__(self, other: "Gate") -> "Gate":
        if not isinstance(other, Gate):
            raise TypeError("Not a gate!")
        new_matrix = self.matrix @ other.matrix
        new_name = f"({self.name} @ {other.name})"
        new_params = self.params + other.params
        return Gate(new_matrix, new_name, new_params)

    def __repr__(self):
        return f"Gate name {self.name}, params = {self.params}"

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return self.matrix @ state

    def dagger(self) -> "Gate":
        # Hermitian conjugate
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


def CNOT(nq, ctr, trg):
    assert 0 <= ctr < nq
    assert 0 <= trg  < nq
    assert ctr != trg

    dim = 2 ** nq
    U = torch.zeros((dim, dim), dtype=DTYPE, device=DEVICE)

    for basis in range(dim):
        # decode to bits
        b = [(basis >> k) & 1 for k in range(nq)]

        if b[ctr] == 1:
            # flip target if control=1
            b[trg] ^= 1

        # re-encode bits to integer
        out = 0
        for k in range(nq):
            out |= (b[k] << k)

        U[out, basis] = 1.0

    return Gate(U, f"CNOT(c={ctr},t={trg})")



I = Gate.I
X = Gate.X
Y = Gate.Y
Z = Gate.Z
H = Gate.H




__all__ = [
    "Gate",
    "I", "X", "Y", "Z", "H", "CNOT"
]
