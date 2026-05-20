import torch


def unitary_fidelity(unitary, target)->float:
    """Phase-invariant overlap between two unitary matrices."""
    if unitary.shape != target.shape:
        raise ValueError("Shape mismatch!")
    dim = unitary.shape[0]
    overlap = torch.trace(target.mH @ unitary)
    return float(torch.abs(overlap) / dim)

def unitary_distance(unitary, target)->float:
    """Convert fidelity into a simple distance-like score."""
    return 1.0 - unitary_fidelity(unitary, target)


SELF_INVERSE_GATES = {"H", "X", "Y", "Z", "CNOT"}
INVERSE_GATES = {
    "S": "Sdg",
    "Sdg": "S",
    "T": "Tdg",
    "Tdg": "T",
}


def base_gate_name(name):
    """Remove qubit labels from logged gate names."""
    return name.split("(", 1)[0]


def gates_cancel(left, right) -> bool:
    """Check whether two logged gates cancel if adjacent."""
    left_name = base_gate_name(left["name"])
    right_name = base_gate_name(right["name"])

    if left["qubits"] != right["qubits"]:
        return False

    if left_name == right_name and left_name in SELF_INVERSE_GATES:
        return True

    return INVERSE_GATES.get(left_name) == right_name


def simplified_circuit(circuit):
    """Remove only obvious adjacent cancellations from a circuit log."""
    stack = []

    for gate in circuit:
        if stack and gates_cancel(stack[-1], gate):
            stack.pop()
        else:
            stack.append(gate)

    return stack


def simplified_depth(circuit) -> int:
    """Depth after the lightweight cancellation pass."""
    return len(simplified_circuit(circuit))
