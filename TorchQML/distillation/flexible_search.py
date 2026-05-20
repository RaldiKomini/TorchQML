from dataclasses import dataclass
from itertools import product

from TorchQML.core.circuit import Circuit, CircuitSpec
from TorchQML.core.sym import Tsym, Xsym
from TorchQML.distillation.circuit_ops import (
    ENT_OPS,
    LOCAL_OPS,
    add_entangler,
    add_global_data,
    add_global_local,
    local_param_count,
)
from TorchQML.gates import H


@dataclass(frozen=True)
class QuantumArchitecture:
    depth: int
    data_op: str
    local_op: str
    entangler: str
    block_type: str = "basic"
    readout: str = "vector_z"
    reupload_pattern: str = "every_layer"
    initial_hadamards: bool = True


@dataclass(frozen=True)
class FlexibleSearchSpaceConfig:
    ncomp: int = 16
    num_qubits: int = 6
    depths: tuple[int, ...] = (2, 3, 4)
    data_ops: tuple[str, ...] = ("rx", "rz", "rx_rz", "zz")
    local_ops: tuple[str, ...] = LOCAL_OPS
    entanglers: tuple[str, ...] = ENT_OPS
    block_types: tuple[str, ...] = ("basic", "sandwich", "ent_first")
    readouts: tuple[str, ...] = ("vector_z", "sum_z", "pauli_head")
    reupload_patterns: tuple[str, ...] = ("every_layer", "every_other", "first_only", "staggered")
    initial_hadamards: tuple[bool, ...] = (True, False)


class FlexibleQuantumSearchSpace:
    """Broad data-reuploading circuit architecture space."""

    def __init__(self, config: FlexibleSearchSpaceConfig):
        self.config = config

    def all_architectures(self) -> list[QuantumArchitecture]:
        arches = []
        for values in product(
            self.config.depths,
            self.config.data_ops,
            self.config.local_ops,
            self.config.entanglers,
            self.config.block_types,
            self.config.readouts,
            self.config.reupload_patterns,
            self.config.initial_hadamards,
        ):
            arch = QuantumArchitecture(*values)
            if self._is_valid_architecture(arch):
                arches.append(arch)
        return arches

    def make_spec(self, arch: QuantumArchitecture) -> CircuitSpec:
        return CircuitSpec(self.config.num_qubits, self.config.ncomp, self._tlen(arch))

    def build_circuit(self, arch: QuantumArchitecture) -> Circuit:
        spec = self.make_spec(arch)
        x = Xsym()
        t = Tsym()
        circ = Circuit(num_qubits=spec.num_qubits, specs=spec)
        if arch.initial_hadamards:
            circ.add_gates([H] * spec.num_qubits)
        x_offset = 0
        t_offset = 0
        for layer_idx in range(arch.depth):
            if arch.block_type == "basic":
                x_offset = self._maybe_add_data(circ, arch, x, x_offset, layer_idx, "pre")
                t_offset = add_global_local(circ, arch.local_op, t, t_offset)
                add_entangler(circ, arch.entangler)
            elif arch.block_type == "sandwich":
                x_offset = self._maybe_add_data(circ, arch, x, x_offset, layer_idx, "pre")
                t_offset = add_global_local(circ, arch.local_op, t, t_offset)
                add_entangler(circ, arch.entangler)
                x_offset = self._maybe_add_data(circ, arch, x, x_offset, layer_idx, "post")
                t_offset = add_global_local(circ, arch.local_op, t, t_offset)
            elif arch.block_type == "ent_first":
                add_entangler(circ, arch.entangler)
                x_offset = self._maybe_add_data(circ, arch, x, x_offset, layer_idx, "post")
                t_offset = add_global_local(circ, arch.local_op, t, t_offset)
            else:
                raise ValueError(f"unknown block_type: {arch.block_type}")
        return circ

    def _tlen(self, arch: QuantumArchitecture) -> int:
        repeats = {"basic": 1, "sandwich": 2, "ent_first": 1}[arch.block_type]
        return arch.depth * repeats * local_param_count(arch.local_op, self.config.num_qubits)

    def _is_valid_architecture(self, arch: QuantumArchitecture) -> bool:
        return self._tlen(arch) > 0 or arch.entangler != "none"

    def _maybe_add_data(self, circ, arch, x, x_offset, layer_idx, position):
        if not self._should_reupload(arch.reupload_pattern, layer_idx, position):
            return x_offset
        return add_global_data(circ, arch.data_op, x, x_offset)

    @staticmethod
    def _should_reupload(pattern: str, layer_idx: int, position: str) -> bool:
        if pattern == "every_layer":
            return True
        if pattern == "every_other":
            return layer_idx % 2 == 0
        if pattern == "first_only":
            return layer_idx == 0
        if pattern == "staggered":
            return (layer_idx % 2 == 0 and position == "pre") or (
                layer_idx % 2 == 1 and position == "post"
            )
        raise ValueError(f"unknown reupload pattern: {pattern}")


__all__ = [
    "FlexibleQuantumSearchSpace",
    "FlexibleSearchSpaceConfig",
    "QuantumArchitecture",
]
