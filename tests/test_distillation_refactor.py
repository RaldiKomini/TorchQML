import torch
from torch.utils.data import DataLoader, TensorDataset

from TorchQML import CircuitSpec
from TorchQML.distillation import (
    AlternatingLayeredSearchSpace,
    BroadAmpSearchSpace,
    BroadSearchSpaceConfig,
    FlexibleQuantumSearchSpace,
    FlexibleSearchSpaceConfig,
    RepeatedCellSearchSpace,
    TrainLoopConfig,
    VectorZMultiClass,
    build_direct_rank_hamiltonian_circuit,
    direct_rank_tlen,
    fit_model,
    fit_mf_surrogate,
    predict_high_batch,
    teacher_similarity_matrix,
)
from TorchQML.synthesis import smoke_unitary_distillation


def test_reusable_multiclass_kd_loop_smoke():
    torch.manual_seed(0)
    X = torch.randn(6, 2)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    teacher_logits = torch.randn(6, 3)
    loader = DataLoader(TensorDataset(X, y, teacher_logits), batch_size=3)

    space = RepeatedCellSearchSpace(num_qubits=2, xlen=2, n_repeats=1)
    cell = ("rx", "ry", "chain", "none", "none")
    spec = space.make_spec(cell)
    circ = space.build_circuit(cell)
    model = VectorZMultiClass(circ, spec, num_classes=3)
    result = fit_model(
        model,
        loader,
        loader,
        torch.device("cpu"),
        "kd",
        TrainLoopConfig(epochs=1, lr=0.01),
    )

    assert len(result.history) == 1
    assert 0.0 <= result.best_val_acc <= 1.0


def test_refactored_search_spaces_build_circuits():
    repeated = RepeatedCellSearchSpace(num_qubits=2, xlen=2, n_repeats=1)
    repeated_circ = repeated.build_circuit(("rx", "ry", "chain", "none", "none"))

    ala = AlternatingLayeredSearchSpace(num_qubits=4, xlen=4, depth=1)
    ala_circ = ala.build_circuit(("rx", "ry", "cnot_01", "none", "none"))

    flexible = FlexibleQuantumSearchSpace(
        FlexibleSearchSpaceConfig(
            ncomp=4,
            num_qubits=2,
            depths=(1,),
            data_ops=("rx",),
            local_ops=("ry",),
            entanglers=("none",),
            block_types=("basic",),
            readouts=("vector_z",),
            reupload_patterns=("every_layer",),
            initial_hadamards=(False,),
        )
    )
    flexible_circ = flexible.build_circuit(flexible.all_architectures()[0])

    broad = BroadAmpSearchSpace(
        BroadSearchSpaceConfig(
            ncomp=4,
            num_qubits=2,
            depths=(1,),
            local_ops=("ry",),
            entanglers=("none",),
            readouts=("vector_z",),
            initial_hadamards=(False,),
        )
    )
    broad_circ = broad.build_circuit(broad.all_architectures()[0])

    assert len(repeated_circ.gate_layers) > 0
    assert len(ala_circ.gate_layers) > 0
    assert len(flexible_circ.gate_layers) > 0
    assert len(broad_circ.gate_layers) > 0


def test_direct_rank_and_kernel_alignment_smoke():
    spec = CircuitSpec(num_qubits=4, xlen=4, tlen=direct_rank_tlen())
    circ = build_direct_rank_hamiltonian_circuit(spec)
    logits = torch.randn(5, 3)
    target = teacher_similarity_matrix(logits, temperature=2.0)

    assert len(circ.gate_layers) > 0
    assert target.shape == (5, 5)
    assert torch.allclose(target.diag(), torch.ones(5))


def test_multifidelity_surrogate_smoke():
    items = [("a",), ("b",), ("c",)]
    low_db = {items[0]: 0.2}
    high_db = {items[0]: 0.3}
    gp_low, gp_delta, rho = fit_mf_surrogate(low_db, high_db)
    mu, sd = predict_high_batch(items, low_db, gp_low, gp_delta, rho)

    assert mu.shape == (3,)
    assert sd.shape == (3,)


def test_unitary_compilation_distillation_smoke():
    result = smoke_unitary_distillation(teacher="qft", n=2, epochs=1, seed=0)
    assert result["teacher"] == "qft"
    assert 0.0 <= result["test_fidelity"] <= 1.0
