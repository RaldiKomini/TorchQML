import argparse

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from TorchQML import Circuit, CircuitSpec, H, Xsym, Tsym, rx
from TorchQML.distillation import fit_binary_kd
from TorchQML.encoding import AmpEnc
from TorchQML.kernels import kernel_matrix
from TorchQML.models import VectorZLinear
from TorchQML.qas import build_patterns, search
from TorchQML.synthesis.search_mitm import apply_action_pruning, build_forward_table, search_backward
from TorchQML.synthesis.analyze_results import summarize
from TorchQML.synthesis.train import evaluate_random
from TorchQML.synthesis.train_all import make_env as make_all_env
from TorchQML.synthesis.train_distill import make_env as make_distill_env
from TorchQML.training import Trainer, TrainConfig, batch_accuracy, softplus


def test_continuous_variational_training_smoke():
    torch.manual_seed(0)
    spec = CircuitSpec(num_qubits=2, xlen=2, tlen=2)
    x = Xsym()
    t = Tsym()

    circ = Circuit(num_qubits=2, specs=spec)
    circ.add_gates([H, H])
    circ.add_gates([rx(x[0]), rx(x[1])])
    circ.add_gates([rx(t[0]), rx(t[1])])

    model = VectorZLinear(circ, spec, amp_enc=False)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                [[0.0, 0.1], [0.6, -0.2], [-0.3, 0.4], [0.9, 0.8]],
                dtype=torch.float32,
            ),
            torch.tensor([0, 1, 0, 1], dtype=torch.long),
        ),
        batch_size=2,
        shuffle=False,
    )
    trainer = Trainer(
        model,
        softplus,
        batch_accuracy,
        TrainConfig(epochs=1, lr=0.01, patience=2),
    )

    history, best_state, best_val = trainer.fit(loader, loader)
    out = trainer.evaluate(loader, best_state)

    assert len(history["train_loss"]) == 1
    assert best_val >= 0.0
    assert 0.0 <= out["acc"] <= 100.0


def test_knowledge_distillation_binary_smoke():
    torch.manual_seed(0)
    X = torch.tensor(
        [
            [-1.0, 0.0, 0.5],
            [-0.5, 0.3, 0.2],
            [0.6, -0.4, 0.1],
            [1.0, 0.2, -0.2],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    teacher_logits = torch.tensor([-2.0, -1.0, 1.5, 2.0], dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y, teacher_logits), batch_size=2, shuffle=False)
    model = torch.nn.Linear(3, 1)

    fitted, history = fit_binary_kd(
        model,
        loader,
        loader,
        torch.device("cpu"),
        epochs=1,
        lr=0.05,
        alpha=0.5,
        temperature=2.0,
    )

    assert fitted is model
    assert len(history) == 1
    assert 0.0 <= history[0]["val_acc"] <= 1.0


def test_discrete_qas_search_smoke():
    torch.manual_seed(0)
    spec = CircuitSpec(num_qubits=2, xlen=4, tlen=8)
    c0 = Circuit(num_qubits=2, specs=spec)
    patterns = build_patterns(spec.num_qubits)[:6]

    def criterion(circ):
        return float(len(circ.gate_layers))

    best_circ, best_patterns, actual_tlen, used_t_idx, best_score = search(
        c0,
        Xsym(),
        Tsym(),
        spec,
        patterns,
        criterion,
        depth=1,
        beam_size=2,
    )

    assert best_circ.num_qubits == 2
    assert len(best_patterns) == 1
    assert actual_tlen <= spec.tlen
    assert all(0 <= idx < spec.tlen for idx in used_t_idx)
    assert best_score >= 0.0


def test_quantum_kernel_and_qsvm_smoke():
    pytest.importorskip("sklearn.svm")
    from TorchQML.quantum_svm import fit_qsvm

    spec = CircuitSpec(num_qubits=2, xlen=4, tlen=0)
    circ = Circuit(num_qubits=2, specs=spec)
    theta = torch.zeros(0, dtype=torch.complex64)
    amp_enc = AmpEnc(2)
    Xtr = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    ytr = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    Xte = Xtr[[0, 2]]

    K = kernel_matrix(Xtr, circ, theta, amp_enc=amp_enc)
    result = fit_qsvm(
        Xtr,
        ytr,
        Xte,
        circ,
        theta,
        amp_enc=amp_enc,
        center=False,
        blockA=2,
        blockB=2,
    )

    assert K.shape == (4, 4)
    assert result["Ktr"].shape == (4, 4)
    assert result["Kte"].shape == (2, 4)
    assert result["pred"].shape == (2,)


def test_random_synthesis_experiment_smoke():
    summary = evaluate_random(
        target_name="t",
        episodes=2,
        max_depth=2,
        t_penalty=0.02,
        cnot_penalty=0.0,
        action_set="base",
    )
    assert summary["episodes"] == 2
    assert 0.0 <= summary["mean_best_fidelity"] <= 1.0


def test_behavior_env_experiment_smoke():
    env = make_all_env(
        target_name="ghz",
        max_depth=3,
        num_probe_states=4,
        seed=0,
        action_set="minimal",
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert info["valid_action_count"] > 0


def test_distill_env_experiment_smoke():
    env = make_distill_env("bell", max_depth=2, num_probe_states=4, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert "state_fidelity" in info


def test_mitm_search_experiment_smoke():
    env = make_all_env(
        target_name="bell",
        max_depth=2,
        num_probe_states=4,
        seed=0,
        action_set="minimal",
    )
    args = argparse.Namespace(
        forward_depth=1,
        backward_depth=1,
        decimals=5,
        skip_immediate_inverse=True,
        max_forward_states=0,
        max_backward_states=0,
        success_threshold=0.999,
    )
    apply_action_pruning(env, "none", 1e-5)
    actions_np = [unitary.detach().cpu().numpy() for unitary in env.action_unitaries]
    table, _ = build_forward_table(env, actions_np, args)
    result = search_backward(env, actions_np, table, args)
    assert result["success"] is True
    assert result["fidelity"] >= 0.999


def test_analyzer_accepts_mixed_experiment_schemas():
    rows = summarize(
        [
            {
                "file": "random_base_t_depth3.json",
                "gate_set": "base",
                "target": "t",
                "max_depth": 3,
                "success_rate": 0.0,
                "mean_final_fidelity": 0.5,
                "mean_best_fidelity": 0.9,
                "mean_depth": 3.0,
            },
            {
                "file": "mitm_minimal_bell_none_f1_b1_seed0.json",
                "target": "bell",
                "action_set": "minimal",
                "success": True,
                "fidelity": 0.999999,
                "depth": 2,
            },
            {
                "file": "ppo_distill_bell_depth2_probes4_seed0.json",
                "target": "bell",
                "success_rate": 0.0,
                "mean_state_fidelity": 0.4,
                "mean_unitary_fidelity": 0.3,
                "mean_depth": 2.0,
            },
        ]
    )
    assert len(rows) == 3
