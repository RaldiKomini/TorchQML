from TorchQML.synthesis.actions import CircuitAction
from TorchQML.gates import H, S, Sdg, T, Tdg, X, Y, Z


ONE_QUBIT_PAULI_ACTION_GATES = [H, S, Sdg, T, Tdg, X, Y, Z]


def build_pauli_actions(num_qubits: int) -> list[CircuitAction]:
    """Build the base action set with X/Y/Z added."""
    actions = []

    for qubit in range(num_qubits):
        for gate in ONE_QUBIT_PAULI_ACTION_GATES:
            actions.append(CircuitAction(gate.name, (qubit,), gate))

    for ctrl in range(num_qubits):
        for trg in range(num_qubits):
            if ctrl != trg:
                actions.append(CircuitAction("CNOT", (ctrl, trg), None))

    return actions
