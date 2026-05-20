from dataclasses import dataclass

from TorchQML.core.unitary import UnitarySimulator
from TorchQML.gates import Gate, H, S, Sdg, T, Tdg


ONE_QUBIT_ACTION_GATES = [H, S, Sdg, T, Tdg]


@dataclass(frozen=True)
class CircuitAction:
    """One discrete action the RL policy can choose."""

    name: str
    qubits: tuple[int, ...]
    gate: Gate | None = None
    parts: tuple = ()


def build_actions(num_qubits: int) -> list[CircuitAction]:
    """Build the default Clifford+T action set for `num_qubits`."""
    actions = []

    for qubit in range(num_qubits):
        for gate in ONE_QUBIT_ACTION_GATES:
            actions.append(CircuitAction(gate.name, (qubit,), gate))

    for ctrl in range(num_qubits):
        for trg in range(num_qubits):
            if ctrl != trg:
                actions.append(CircuitAction("CNOT", (ctrl, trg), None))

    return actions


def primitive_actions(action: CircuitAction) -> tuple[CircuitAction, ...]:
    """Flatten a macro action into the primitive gates it applies."""
    if not action.parts:
        return (action,)

    primitives = []
    for part in action.parts:
        primitives.extend(primitive_actions(part))
    return tuple(primitives)


def action_depth(action: CircuitAction) -> int:
    """Count how many primitive gates this action adds."""
    return len(primitive_actions(action))


def action_t_count(action: CircuitAction) -> int:
    """Count T/Tdg primitives inside an action."""
    return sum(1 for part in primitive_actions(action) if part.name in {"T", "Tdg"})


def action_cnot_count(action: CircuitAction) -> int:
    """Count CNOT primitives inside an action."""
    return sum(1 for part in primitive_actions(action) if base_action_name(part.name) == "CNOT")


SELF_INVERSE_ACTIONS = {"H", "X", "Y", "Z", "CNOT"}
INVERSE_ACTIONS = {
    "S": "Sdg",
    "Sdg": "S",
    "T": "Tdg",
    "Tdg": "T",
}


def base_action_name(name: str) -> str:
    """Strip display details from an action name."""
    if name.startswith("CNOT"):
        return "CNOT"
    return name.split("(", 1)[0]


def actions_cancel(left: CircuitAction, right: CircuitAction) -> bool:
    """Return True for obvious adjacent cancellations."""
    left_name = base_action_name(left.name)
    right_name = base_action_name(right.name)

    if left.qubits != right.qubits:
        return False

    if left_name == right_name and left_name in SELF_INVERSE_ACTIONS:
        return True

    return INVERSE_ACTIONS.get(left_name) == right_name


def apply_action(simulator: UnitarySimulator, action: CircuitAction):
    """Apply an action to the simulator and return the new unitary."""
    if action.parts:
        result = None
        for part in primitive_actions(action):
            result = apply_action(simulator, part)
        return result

    if action.name == "CNOT":
        ctrl, trg = action.qubits
        return simulator.add_cnot(ctrl, trg)
    qubit = action.qubits[0]
    return simulator.add_gate(action.gate, qubit)
