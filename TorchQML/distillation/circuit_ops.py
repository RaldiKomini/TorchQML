import math

from TorchQML.core.circuit import Circuit
from TorchQML.core.sym import Tsym, Xsym
from TorchQML.gates import CNOT, H, I, rx, ry, rz


Cell = tuple[str, str, str, str, str]
DATA_OPS = ("none", "rx", "rz", "rx_rz", "zz")
LOCAL_OPS = ("none", "ry", "rz", "ry_rz")
ENT_OPS = ("none", "chain", "ring", "pairs")


def add_single_qubit_gate(circ: Circuit, qubit: int, gate):
    layer = [I] * circ.num_qubits
    layer[qubit] = gate
    circ.add_gates(layer)
    return circ


def add_entangler(circ: Circuit, entangler: str) -> None:
    nq = circ.num_qubits
    if entangler == "none":
        return
    if entangler == "chain":
        for q in range(nq - 1):
            circ.add_full(CNOT(nq, q, q + 1))
        return
    if entangler == "ring":
        for q in range(nq):
            circ.add_full(CNOT(nq, q, (q + 1) % nq))
        return
    if entangler == "pairs":
        for q in range(0, nq, 2):
            if q + 1 < nq:
                circ.add_full(CNOT(nq, q, q + 1))
        return
    raise ValueError(f"unknown entangler: {entangler}")


def local_param_count(op_name: str, num_qubits: int) -> int:
    if op_name == "none":
        return 0
    if op_name in ("ry", "rz"):
        return num_qubits
    if op_name == "ry_rz":
        return 2 * num_qubits
    raise ValueError(f"unknown local op: {op_name}")


def add_global_local(circ: Circuit, op_name: str, t: Tsym, t_offset: int) -> int:
    nq = circ.num_qubits
    if op_name == "none":
        return t_offset
    if op_name == "ry":
        circ.add_gates([ry(math.pi * t[t_offset + q]) for q in range(nq)])
        return t_offset + nq
    if op_name == "rz":
        circ.add_gates([rz(math.pi * t[t_offset + q]) for q in range(nq)])
        return t_offset + nq
    if op_name == "ry_rz":
        circ.add_gates([ry(math.pi * t[t_offset + q]) for q in range(nq)])
        circ.add_gates([rz(math.pi * t[t_offset + nq + q]) for q in range(nq)])
        return t_offset + 2 * nq
    raise ValueError(f"unknown local op: {op_name}")


def add_global_data(circ: Circuit, op_name: str, x: Xsym, x_offset: int) -> int:
    nq = circ.num_qubits
    xlen = circ.specs.xlen
    if op_name == "none":
        return x_offset
    if op_name == "rx":
        circ.add_gates([rx(math.pi * x[(x_offset + q) % xlen]) for q in range(nq)])
        return x_offset + nq
    if op_name == "rz":
        circ.add_gates([rz(math.pi * x[(x_offset + q) % xlen]) for q in range(nq)])
        return x_offset + nq
    if op_name == "rx_rz":
        circ.add_gates([rx(math.pi * x[(x_offset + q) % xlen]) for q in range(nq)])
        circ.add_gates([rz(math.pi * x[(x_offset + q) % xlen]) for q in range(nq)])
        return x_offset + nq
    if op_name == "zz":
        for q in range(nq - 1):
            a = (x_offset + q) % xlen
            b = (x_offset + q + 1) % xlen
            angle = math.pi * x[a] * x[b]
            circ.add_full(CNOT(nq, q, q + 1))
            layer = [I] * nq
            layer[q + 1] = rz(angle)
            circ.add_gates(layer)
            circ.add_full(CNOT(nq, q, q + 1))
        return x_offset + nq
    raise ValueError(f"unknown data op: {op_name}")


__all__ = [
    "Cell",
    "DATA_OPS",
    "ENT_OPS",
    "LOCAL_OPS",
    "add_entangler",
    "add_global_data",
    "add_global_local",
    "add_single_qubit_gate",
    "local_param_count",
]
