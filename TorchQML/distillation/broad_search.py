from dataclasses import dataclass
from itertools import product

from TorchQML.core.circuit import Circuit, CircuitSpec
from TorchQML.core.sym import Tsym
from TorchQML.distillation.circuit_ops import (
    ENT_OPS,
    LOCAL_OPS,
    add_entangler,
    add_global_local,
    local_param_count,
)
from TorchQML.gates import H


@dataclass(frozen=True)
class BroadArchitecture:
    depth: int
    local_pre: str
    entangler_pre: str
    local_mid: str
    entangler_post: str
    local_post: str
    readout: str = "vector_z"
    initial_hadamards: bool = False


@dataclass(frozen=True)
class BroadSearchSpaceConfig:
    ncomp: int = 32
    num_qubits: int = 5
    depths: tuple[int, ...] = (2, 3, 4)
    local_ops: tuple[str, ...] = LOCAL_OPS
    entanglers: tuple[str, ...] = ENT_OPS
    readouts: tuple[str, ...] = ("vector_z", "pauli_head")
    initial_hadamards: tuple[bool, ...] = (False, True)


class BroadAmpSearchSpace:
    """Amplitude-encoding architecture space from the broad FMNIST searches."""

    def __init__(self, config: BroadSearchSpaceConfig):
        self.config = config

    def all_architectures(self) -> list[BroadArchitecture]:
        arches = []
        for values in product(
            self.config.depths,
            self.config.local_ops,
            self.config.entanglers,
            self.config.local_ops,
            self.config.entanglers,
            self.config.local_ops,
            self.config.readouts,
            self.config.initial_hadamards,
        ):
            arch = BroadArchitecture(*values)
            if self._is_valid_architecture(arch):
                arches.append(arch)
        return arches

    def make_spec(self, arch: BroadArchitecture) -> CircuitSpec:
        return CircuitSpec(self.config.num_qubits, self.config.ncomp, self._tlen(arch))

    def build_circuit(self, arch: BroadArchitecture) -> Circuit:
        spec = self.make_spec(arch)
        t = Tsym()
        circ = Circuit(num_qubits=spec.num_qubits, specs=spec)
        if arch.initial_hadamards:
            circ.add_gates([H] * spec.num_qubits)
        t_offset = 0
        for _ in range(arch.depth):
            t_offset = add_global_local(circ, arch.local_pre, t, t_offset)
            add_entangler(circ, arch.entangler_pre)
            t_offset = add_global_local(circ, arch.local_mid, t, t_offset)
            add_entangler(circ, arch.entangler_post)
            t_offset = add_global_local(circ, arch.local_post, t, t_offset)
        return circ

    def _tlen(self, arch: BroadArchitecture) -> int:
        return arch.depth * sum(
            local_param_count(op, self.config.num_qubits)
            for op in (arch.local_pre, arch.local_mid, arch.local_post)
        )

    def _is_valid_architecture(self, arch: BroadArchitecture) -> bool:
        has_trainable = self._tlen(arch) > 0
        has_entangler = arch.entangler_pre != "none" or arch.entangler_post != "none"
        return has_trainable or has_entangler


__all__ = [
    "BroadAmpSearchSpace",
    "BroadArchitecture",
    "BroadSearchSpaceConfig",
]
