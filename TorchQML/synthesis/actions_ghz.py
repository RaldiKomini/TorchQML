from TorchQML.gates import H
from TorchQML.synthesis.actions import CircuitAction


def build_ghz_actions(num_qubits: int = 3) -> list[CircuitAction]:
    """Use only the gates needed for GHZ-style preparation."""
    actions = []

    for qubit in range(num_qubits):
        actions.append(CircuitAction(H.name, (qubit,), H))

    for ctrl in range(num_qubits):
        for trg in range(num_qubits):
            if ctrl != trg:
                actions.append(CircuitAction("CNOT", (ctrl, trg), None))

    return actions
