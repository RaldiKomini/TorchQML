from dataclasses import dataclass

from TorchQML.core.unitary import UnitarySimulator
from TorchQML.gates import H, T, Tdg


@dataclass(frozen=True)
class BaselineCircuit:
    """Known reference numbers for a synthesis target."""

    name: str
    num_qubits: int
    description: str
    optimal_depth: int
    optimal_t_count: int
    optimal_cnot_count: int


BASELINES = {
    "t": BaselineCircuit(
        name="t",
        num_qubits=1,
        description="Single T gate on qubit 0.",
        optimal_depth=1,
        optimal_t_count=1,
        optimal_cnot_count=0,
    ),
    "bell": BaselineCircuit(
        name="bell",
        num_qubits=2,
        description="Bell-preparation unitary: H(0), CNOT(0, 1).",
        optimal_depth=2,
        optimal_t_count=0,
        optimal_cnot_count=1,
    ),
    "ghz": BaselineCircuit(
        name="ghz",
        num_qubits=3,
        description="GHZ-preparation unitary: H(0), CNOT(0, 1), CNOT(1, 2).",
        optimal_depth=3,
        optimal_t_count=0,
        optimal_cnot_count=2,
    ),
    "toffoli": BaselineCircuit(
        name="toffoli",
        num_qubits=3,
        description="CCNOT with controls q0/q1 and target q2; standard 7-T decomposition.",
        optimal_depth=15,
        optimal_t_count=7,
        optimal_cnot_count=6,
    ),
}


def build_baseline_unitary(name: str):
    """Build the known reference unitary for a named target."""
    simulator = UnitarySimulator(BASELINES[name].num_qubits)

    if name == "t":
        simulator.add_gate(T, 0)
    elif name == "bell":
        simulator.add_gate(H, 0)
        simulator.add_cnot(0, 1)
    elif name == "ghz":
        simulator.add_gate(H, 0)
        simulator.add_cnot(0, 1)
        simulator.add_cnot(1, 2)
    elif name == "toffoli":
        simulator.add_gate(H, 2)
        simulator.add_cnot(1, 2)
        simulator.add_gate(Tdg, 2)
        simulator.add_cnot(0, 2)
        simulator.add_gate(T, 2)
        simulator.add_cnot(1, 2)
        simulator.add_gate(Tdg, 2)
        simulator.add_cnot(0, 2)
        simulator.add_gate(T, 1)
        simulator.add_gate(T, 2)
        simulator.add_cnot(0, 1)
        simulator.add_gate(H, 2)
        simulator.add_gate(T, 0)
        simulator.add_gate(Tdg, 1)
        simulator.add_cnot(0, 1)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    return simulator.unitary.clone()
