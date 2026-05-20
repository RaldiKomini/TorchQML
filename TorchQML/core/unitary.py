import torch
from dataclasses import dataclass

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.circuit import _kron_all
from TorchQML.gates import Gate, I, X, Y, Z, H, S, Sdg, T, Tdg, CNOT


ONE_QUBIT_GATES = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "Sdg": Sdg,
    "T": T,
    "Tdg": Tdg,
}


@dataclass(frozen=True)
class AppliedGate:
    name: str
    qubits: tuple[int, ...]

class UnitarySimulator:
    def __init__(self, num_qubits):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be >= 1!")

        self.num_qubits = num_qubits
        self.dim = 1 << num_qubits
        self.gate_layers: list[AppliedGate] = []

        #stats
        self.depth = 0
        self.t_count = 0
        self.cnot_count = 0

        #caching optimization
        self._eye = torch.eye(self.dim, dtype=DTYPE, device=DEVICE)
        self.unitary = self._eye.clone()
        self._expand_gate_cache ={}
        self._cnot_cache = {}


    def reset(self) ->torch.Tensor:
        self.unitary = self._eye.clone()
        self.gate_layers.clear()

        self.depth = 0
        self.t_count = 0
        self.cnot_count = 0
        return self.unitary.clone()

    def add_full(self, g: Gate, qubits: tuple[int, ...])->torch.Tensor:
        if g.matrix.shape != (self.dim, self.dim):
            raise ValueError("Gate has incompatible shape!")

        self.unitary = g.matrix @ self.unitary
        self.gate_layers.append(AppliedGate(g.name, qubits))

        self.depth += 1
        name = g.name.lower()

        if name.startswith("t(") or name.startswith("tdg(") or name in {"t", "tdg"}:
            self.t_count += 1

        if name.startswith("cnot"):
            self.cnot_count += 1

        return self.unitary.clone()


    def _check_qubit(self, qubit):
        if not 0 <= qubit < self.num_qubits:
            raise ValueError("Invalid qubit!")

    def _expand_gate(self, g, qubit)->Gate:
        self._check_qubit(qubit)

        if g.matrix.shape != (2, 2):
            raise ValueError("Gate matrix has wrong shape!")

        cache_key = (g.name, qubit)
        if cache_key in self._expand_gate_cache:
            return self._expand_gate_cache[cache_key]

        wire = self.num_qubits - 1 - qubit
        matrices = [
            g.matrix if i == wire else I.matrix
            for i in range(self.num_qubits)
        ]

        full_gate = Gate(_kron_all(matrices), f"{g.name}({qubit})")
        self._expand_gate_cache[cache_key] = full_gate
        return full_gate

    def add_gate(self, g, qubit)->torch.Tensor:
        exp_gate = self._expand_gate(g, qubit)
        return self.add_full(exp_gate, (qubit, ))

    def is_unitary(self, atol: float = 1e-6) -> bool:
        return torch.allclose(self.unitary.mH @ self.unitary, self._eye, atol=atol)

    def _cnot_matrix(self, control: int, target: int) -> Gate:
        self._check_qubit(control)
        self._check_qubit(target)

        if control == target:
            raise ValueError("control and target must be different")

        cache_key = (control, target)
        if cache_key in self._cnot_cache:
            return self._cnot_cache[cache_key]

        gate = CNOT(self.num_qubits, control, target)
        self._cnot_cache[cache_key] = gate
        return gate

    def add_cnot(self, control: int, target: int) -> torch.Tensor:
        gate = self._cnot_matrix(control, target)
        return self.add_full(gate, (control, target))


    def counts(self) -> dict[str, int]:
        return {
            "depth": self.depth,
            "t_count": self.t_count,
            "cnot_count": self.cnot_count,
        }
