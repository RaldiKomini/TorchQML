from itertools import product
import math

from TorchQML.core.circuit import Circuit, CircuitSpec
from TorchQML.core.sym import Tsym, Xsym
from TorchQML.distillation.circuit_ops import (
    Cell,
    DATA_OPS,
    LOCAL_OPS,
    add_entangler,
    add_global_data,
    add_global_local,
    add_single_qubit_gate,
    local_param_count,
)
from TorchQML.gates import CNOT, H, rx, ry, rz


class RepeatedCellSearchSpace:
    """Global repeated-cell architecture space used by the KD experiments."""

    def __init__(self, num_qubits: int, xlen: int, n_repeats: int, add_initial_h: bool = True):
        self.num_qubits = num_qubits
        self.xlen = xlen
        self.n_repeats = n_repeats
        self.add_initial_h = add_initial_h

    def all_cells(self) -> list[Cell]:
        from TorchQML.distillation.circuit_ops import ENT_OPS

        return list(product(DATA_OPS, LOCAL_OPS, ENT_OPS, DATA_OPS, LOCAL_OPS))

    def repeated_cell_tlen(self, cell: Cell) -> int:
        _, local1, _, _, local2 = cell
        return self.n_repeats * (
            local_param_count(local1, self.num_qubits)
            + local_param_count(local2, self.num_qubits)
        )

    def make_spec(self, cell: Cell) -> CircuitSpec:
        return CircuitSpec(self.num_qubits, self.xlen, self.repeated_cell_tlen(cell))

    def build_circuit(self, cell: Cell) -> Circuit:
        spec = self.make_spec(cell)
        x = Xsym()
        t = Tsym()
        circ = Circuit(num_qubits=spec.num_qubits, specs=spec)
        if self.add_initial_h:
            circ.add_gates([H] * spec.num_qubits)
        data1, local1, ent, data2, local2 = cell
        x_offset = 0
        t_offset = 0
        for _ in range(self.n_repeats):
            x_offset = add_global_data(circ, data1, x, x_offset)
            t_offset = add_global_local(circ, local1, t, t_offset)
            add_entangler(circ, ent)
            x_offset = add_global_data(circ, data2, x, x_offset)
            t_offset = add_global_local(circ, local2, t, t_offset)
        return circ


def get_ala_partitions(num_qubits: int, periodic: bool = True):
    even = [(q, q + 1) for q in range(0, num_qubits - 1, 2)]
    odd = [(q, q + 1) for q in range(1, num_qubits - 1, 2)]
    if periodic and num_qubits % 2 == 0 and num_qubits >= 4:
        odd.append((num_qubits - 1, 0))
    return even, odd


def add_block_entangler(circ: Circuit, qa: int, qb: int, entangler: str):
    if entangler == "none":
        return circ
    if entangler == "cnot_01":
        circ.add_full(CNOT(circ.num_qubits, qa, qb))
        return circ
    if entangler == "cnot_10":
        circ.add_full(CNOT(circ.num_qubits, qb, qa))
        return circ
    if entangler == "cnot_both":
        circ.add_full(CNOT(circ.num_qubits, qa, qb))
        circ.add_full(CNOT(circ.num_qubits, qb, qa))
        return circ
    raise ValueError(f"unknown block entangler: {entangler}")


def block_local_param_count(op_name: str) -> int:
    if op_name == "none":
        return 0
    if op_name in ("ry", "rz"):
        return 2
    if op_name == "ry_rz":
        return 4
    raise ValueError(f"unknown local op: {op_name}")


class AlternatingLayeredSearchSpace:
    """Two-qubit-block ALA search space from the older student-search script."""

    data_ops = ("none", "rx", "rz", "rx_rz", "zz")
    local_ops = LOCAL_OPS
    ent_ops = ("none", "cnot_01", "cnot_10", "cnot_both")

    def __init__(self, num_qubits: int, xlen: int, depth: int, periodic: bool = True, add_initial_h: bool = True):
        self.num_qubits = num_qubits
        self.xlen = xlen
        self.depth = depth
        self.periodic = periodic
        self.add_initial_h = add_initial_h

    def all_cells(self) -> list[Cell]:
        return list(product(self.data_ops, self.local_ops, self.ent_ops, self.data_ops, self.local_ops))

    def total_blocks(self) -> int:
        even, odd = get_ala_partitions(self.num_qubits, periodic=self.periodic)
        return sum(len(even if idx % 2 == 0 else odd) for idx in range(self.depth))

    def cell_tlen(self, cell: Cell) -> int:
        _, local1, _, _, local2 = cell
        return self.total_blocks() * (
            block_local_param_count(local1) + block_local_param_count(local2)
        )

    def make_spec(self, cell: Cell) -> CircuitSpec:
        return CircuitSpec(self.num_qubits, self.xlen, self.cell_tlen(cell))

    def build_circuit(self, cell: Cell) -> Circuit:
        spec = self.make_spec(cell)
        x = Xsym()
        t = Tsym()
        circ = Circuit(num_qubits=spec.num_qubits, specs=spec)
        if self.add_initial_h:
            circ.add_gates([H] * spec.num_qubits)
        even, odd = get_ala_partitions(spec.num_qubits, periodic=self.periodic)
        x_offset = 0
        t_offset = 0
        data1, local1, ent, data2, local2 = cell
        for idx in range(self.depth):
            blocks = even if idx % 2 == 0 else odd
            for qa, qb in blocks:
                x_offset = self._add_block_data(circ, data1, qa, qb, x, x_offset)
                t_offset = self._add_block_local(circ, local1, qa, qb, t, t_offset)
                add_block_entangler(circ, qa, qb, ent)
                x_offset = self._add_block_data(circ, data2, qa, qb, x, x_offset)
                t_offset = self._add_block_local(circ, local2, qa, qb, t, t_offset)
        return circ

    def _add_block_data(self, circ, op_name, qa, qb, x, x_offset):
        if op_name == "none":
            return x_offset
        a = x_offset % circ.specs.xlen
        b = (x_offset + 1) % circ.specs.xlen
        if op_name in {"rx", "rz", "rx_rz"}:
            gates = [rx] if op_name == "rx" else [rz] if op_name == "rz" else [rx, rz]
            for gate in gates:
                add_single_qubit_gate(circ, qa, gate(math.pi * x[a]))
                add_single_qubit_gate(circ, qb, gate(math.pi * x[b]))
            return x_offset + 2
        if op_name == "zz":
            circ.add_full(CNOT(circ.num_qubits, qa, qb))
            add_single_qubit_gate(circ, qb, rz(math.pi * x[a] * x[b]))
            circ.add_full(CNOT(circ.num_qubits, qa, qb))
            return x_offset + 2
        raise ValueError(f"unknown data op: {op_name}")

    def _add_block_local(self, circ, op_name, qa, qb, t, t_offset):
        if op_name == "none":
            return t_offset
        gates = [ry] if op_name == "ry" else [rz] if op_name == "rz" else [ry, rz]
        for gate in gates:
            add_single_qubit_gate(circ, qa, gate(math.pi * t[t_offset]))
            add_single_qubit_gate(circ, qb, gate(math.pi * t[t_offset + 1]))
            t_offset += 2
        return t_offset


__all__ = [
    "AlternatingLayeredSearchSpace",
    "RepeatedCellSearchSpace",
    "add_block_entangler",
    "block_local_param_count",
    "get_ala_partitions",
]
