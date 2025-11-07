import math
import torch
from qml_lib.config import DTYPE, DEVICE


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
        new_matrix = self.matrix.mH  # same as self.matrix.conj().transpose(-2, -1)
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

Gate.CNOT = Gate(
    torch.tensor(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=DTYPE, device=DEVICE
    ),
    "CNOT"
)

I = Gate.I
X = Gate.X
Y = Gate.Y
Z = Gate.Z
H = Gate.H
CNOT = Gate.CNOT




__all__ = [
    "Gate",
    "I", "X", "Y", "Z", "H", "CNOT",
    "rx", "ry", "rz",
]
