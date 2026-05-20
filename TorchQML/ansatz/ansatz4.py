import math

from TorchQML.core import Circuit, Tsym, Xsym
from TorchQML.gates import CNOT, H, I, X, rx, ry, rz


def state4_big_tlen(num_blocks: int = 6) -> int:
    """Return the parameter count expected by `state4_big`."""
    return 8 + 24 * num_blocks


def state4_big(specs, num_blocks: int = 6):
    """Build the four-qubit data-reuploading ansatz used by the models."""
    x = Xsym()
    t = Tsym()

    num_qubits = 4
    xlen = specs.xlen
    circuit = Circuit(num_qubits=num_qubits)

    circuit.add_gates([H] * num_qubits)
    circuit.add_gates([X if q % 2 == 0 else I for q in range(num_qubits)])

    t_idx = 0
    gates_ry0 = []
    gates_rz0 = []
    for _ in range(num_qubits):
        gates_ry0.append(ry(math.pi * t[t_idx]))
        t_idx += 1
        gates_rz0.append(rz(math.pi * t[t_idx]))
        t_idx += 1
    circuit.add_gates(gates_ry0)
    circuit.add_gates(gates_rz0)

    for block in range(num_blocks):
        gates_ry = []
        gates_rz = []
        gates_rx = []

        for q in range(num_qubits):
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + 7) % xlen

            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
            t_idx += 2
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1])
            t_idx += 2
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1])
            t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        circuit.add_gates(gates_ry)
        circuit.add_gates(gates_rz)
        circuit.add_gates(gates_rx)

        if block % 2 == 0:
            for q in range(num_qubits):
                circuit.add_full(CNOT(num_qubits, q, (q + 1) % num_qubits))
        else:
            circuit.add_full(CNOT(num_qubits, 0, 2))
            circuit.add_full(CNOT(num_qubits, 1, 3))
            circuit.add_full(CNOT(num_qubits, 0, 3))

    expected_tlen = state4_big_tlen(num_blocks)
    if hasattr(specs, "tlen") and specs.tlen != expected_tlen:
        raise ValueError(f"state4_big expects specs.tlen={expected_tlen}, got {specs.tlen}")

    return circuit


__all__ = ["state4_big", "state4_big_tlen"]
