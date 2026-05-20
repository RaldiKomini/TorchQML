from TorchQML.synthesis.actions import CircuitAction, actions_cancel
from TorchQML.synthesis.actions_toffoli import build_toffoli_actions


def _format_action(action: CircuitAction) -> str:
    """Make compact names for two-gate macro actions."""
    if action.name == "CNOT":
        ctrl, trg = action.qubits
        return f"CNOT({ctrl},{trg})"

    return f"{action.name}({action.qubits[0]})"


def build_toffoli_macro_actions() -> list[CircuitAction]:
    """Add filtered two-gate macros to the restricted Toffoli actions."""
    primitives = build_toffoli_actions()
    actions = list(primitives)

    for first in primitives:
        for second in primitives:
            # Do not spend an action on pairs that cancel immediately.
            if actions_cancel(first, second):
                continue

            name = f"{_format_action(first)}+{_format_action(second)}"
            actions.append(CircuitAction(name, (), parts=(first, second)))

    return actions
