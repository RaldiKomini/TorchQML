from TorchQML.core.unitary import UnitarySimulator
from TorchQML.synthesis.actions import actions_cancel, apply_action, build_actions
from TorchQML.synthesis.actions_ghz import build_ghz_actions
from TorchQML.synthesis.actions_pauli import build_pauli_actions
from TorchQML.synthesis.actions_toffoli import build_toffoli_actions
from TorchQML.synthesis.actions_toffoli_macro import build_toffoli_macro_actions


def test_build_actions_has_expected_size():
    assert len(build_actions(num_qubits=1)) == 5
    assert len(build_actions(num_qubits=2)) == 12
    assert len(build_actions(num_qubits=3)) == 21


def test_actions_include_cnot_for_both_directions():
    actions = build_actions(num_qubits=2)
    cnot_qubits = {action.qubits for action in actions if action.name == "CNOT"}

    assert cnot_qubits == {(0, 1), (1, 0)}


def test_apply_action_updates_simulator_counts():
    simulator = UnitarySimulator(num_qubits=2)
    cnot_action = next(
        action for action in build_actions(num_qubits=2)
        if action.name == "CNOT" and action.qubits == (0, 1)
    )

    apply_action(simulator, cnot_action)

    assert simulator.counts()["depth"] == 1
    assert simulator.counts()["cnot_count"] == 1
    assert simulator.is_unitary()


def test_pauli_action_set_includes_xyz():
    actions = build_pauli_actions(num_qubits=3)
    one_qubit_names = {action.name for action in actions if len(action.qubits) == 1}

    assert {"X", "Y", "Z"}.issubset(one_qubit_names)
    assert len(actions) == 30


def test_ghz_action_set_only_has_h_and_cnot():
    actions = build_ghz_actions(num_qubits=3)
    names = {action.name for action in actions}

    assert names == {"H", "CNOT"}
    assert len(actions) == 9


def test_restricted_toffoli_action_set_is_small():
    actions = build_toffoli_actions()

    assert len(actions) == 13
    assert actions[0].name == "H"
    assert actions[0].qubits == (2,)


def test_toffoli_macro_actions_skip_obvious_cancellations():
    actions = build_toffoli_macro_actions()
    macro_actions = [action for action in actions if action.parts]

    assert len(actions) == 169
    assert macro_actions
    assert all(
        not actions_cancel(action.parts[0], action.parts[1])
        for action in macro_actions
    )
