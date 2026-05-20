from TorchQML.gates import H, T, Tdg
from TorchQML.synthesis.actions import CircuitAction


def build_toffoli_actions() -> list[CircuitAction]:
    """Build the small action set used for Toffoli probes."""
    actions = [CircuitAction(H.name, (2,), H)]

    for qubit in range(3):
        actions.append(CircuitAction(T.name, (qubit,), T))
        actions.append(CircuitAction(Tdg.name, (qubit,), Tdg))

    for ctrl in range(3):
        for trg in range(3):
            if ctrl != trg:
                actions.append(CircuitAction("CNOT", (ctrl, trg), None))

    return actions
