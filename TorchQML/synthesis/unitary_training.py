from dataclasses import dataclass
import math
import random

import numpy as np
import torch

from TorchQML.synthesis.unitary_dense import (
    cnot_full,
    cp_full,
    cz_full,
    full_single,
    h_gate,
    qft_full,
    rx_gate,
    ry_gate,
    rz_gate,
    swap_full,
)
from TorchQML.synthesis.unitary_states import make_states, normalize


@dataclass(frozen=True)
class Architecture:
    """Compact description of a student ansatz used for unitary distillation."""

    depth: int
    local: str = "ry_rz"
    entangler: str = "cnot_chain"
    tail_swap: bool = False


@dataclass
class TrainResult:
    best_val_fidelity: float
    test_fidelity: float
    final_train_fidelity: float
    trained_epochs: int
    best_theta: list[float]
    history: list[dict[str, float]]


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def local_param_count(local: str, n: int) -> int:
    if local in {"none", "h"}:
        return 0
    if local in {"rx", "ry", "rz"}:
        return n
    if local in {"ry_rz", "rx_ry", "rx_rz"}:
        return 2 * n
    if local == "h_ry_rz":
        return 2 * n
    if local in {"rx_ry_rz", "h_rx_ry_rz"}:
        return 3 * n
    raise ValueError(f"unknown local block: {local}")


def entangler_param_count(entangler: str, n: int) -> int:
    if entangler.startswith("cp_trainable"):
        return max(0, n - 1)
    return 0


def param_count(arch: Architecture, n: int) -> int:
    return arch.depth * (local_param_count(arch.local, n) + entangler_param_count(arch.entangler, n))


def twoq_count(arch: Architecture, n: int) -> int:
    per_layer = {
        "none": 0,
        "cnot_chain": max(0, n - 1),
        "cnot_ring": n,
        "cz_chain": max(0, n - 1),
        "cz_full": n * (n - 1) // 2,
        "cp_chain": max(0, n - 1),
        "cp_ring": n,
        "cp_full": n * (n - 1) // 2,
    }.get(arch.entangler)
    if per_layer is None:
        raise ValueError(f"unknown entangler: {arch.entangler}")
    return arch.depth * per_layer + (n // 2 if arch.tail_swap else 0)


def all_architectures(
    *,
    depths=(1, 2, 3),
    locals=("h", "ry_rz", "rx_ry_rz"),
    entanglers=("none", "cnot_chain", "cnot_ring"),
    tail_swaps=(False,),
) -> list[Architecture]:
    return [
        Architecture(depth, local, entangler, tail_swap)
        for depth in depths
        for local in locals
        for entangler in entanglers
        for tail_swap in tail_swaps
    ]


def apply_local_block(states: torch.Tensor, theta: torch.Tensor, offset: int, local: str, n: int):
    if local == "none":
        return states, offset
    for q in range(n):
        if local in {"h", "h_ry_rz", "h_rx_ry_rz"}:
            states = states @ full_single(h_gate(), q, n).T
        if local in {"rx", "rx_ry", "rx_rz", "rx_ry_rz", "h_rx_ry_rz"}:
            states = states @ full_single(rx_gate(theta[offset]), q, n).T
            offset += 1
        if local in {"ry", "ry_rz", "rx_ry", "rx_ry_rz", "h_ry_rz", "h_rx_ry_rz"}:
            states = states @ full_single(ry_gate(theta[offset]), q, n).T
            offset += 1
        if local in {"rz", "ry_rz", "rx_rz", "rx_ry_rz", "h_ry_rz", "h_rx_ry_rz"}:
            states = states @ full_single(rz_gate(theta[offset]), q, n).T
            offset += 1
    return states, offset


def apply_entangler(states: torch.Tensor, theta: torch.Tensor, offset: int, entangler: str, n: int):
    if entangler == "none":
        return states, offset
    if entangler in {"cnot_chain", "cz_chain", "cp_chain"}:
        pairs = [(q, q + 1) for q in range(n - 1)]
    elif entangler in {"cnot_ring", "cp_ring"}:
        pairs = [(q, (q + 1) % n) for q in range(n)]
    elif entangler in {"cz_full", "cp_full"}:
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    else:
        raise ValueError(f"unknown entangler: {entangler}")
    for a, b in pairs:
        if entangler.startswith("cnot"):
            gate = cnot_full(a, b, n)
        elif entangler.startswith("cz"):
            gate = cz_full(a, b, n)
        elif entangler.startswith("cp"):
            angle = theta[offset] if "trainable" in entangler else math.pi / 2
            gate = cp_full(a, b, angle, n)
            if "trainable" in entangler:
                offset += 1
        states = states @ gate.T
    return states, offset


def apply_student(states: torch.Tensor, theta: torch.Tensor, arch: Architecture, n: int) -> torch.Tensor:
    theta = theta.to(torch.float32)
    offset = 0
    for _ in range(arch.depth):
        # Each layer is local rotations followed by the selected entangling pattern.
        states, offset = apply_local_block(states, theta, offset, arch.local, n)
        states, offset = apply_entangler(states, theta, offset, arch.entangler, n)
    if arch.tail_swap:
        for q in range(0, n - 1, 2):
            states = states @ swap_full(q, q + 1, n).T
    return normalize(states)


def apply_teacher(name: str, states: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    if name == "identity":
        return states
    if name == "qft":
        return states @ qft_full(n).T
    if name == "random_heavy":
        set_seed(seed)
        out = states
        for q in range(n):
            out = out @ full_single(h_gate(), q, n).T
        for q in range(n - 1):
            out = out @ cnot_full(q, q + 1, n).T
        for q in range(n):
            out = out @ full_single(rz_gate(torch.tensor(0.37 * (q + 1))), q, n).T
        return normalize(out)
    raise ValueError(f"unknown teacher: {name}")


def mean_fidelity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((target.conj() * pred).sum(dim=1).abs().square().real).mean()


def train_architecture(
    arch: Architecture,
    *,
    n: int,
    train_states: torch.Tensor,
    val_states: torch.Tensor,
    test_states: torch.Tensor,
    train_targets: torch.Tensor,
    val_targets: torch.Tensor,
    test_targets: torch.Tensor,
    epochs: int = 20,
    lr: float = 0.04,
    init_scale: float = 0.1,
    seed: int = 0,
) -> TrainResult:
    set_seed(seed)
    theta = torch.nn.Parameter(init_scale * torch.randn(param_count(arch, n)))
    optimizer = torch.optim.Adam([theta], lr=lr)
    best_theta = theta.detach().clone()
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_fid = mean_fidelity(apply_student(train_states, theta, arch, n), train_targets)
        loss = 1.0 - train_fid
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_fid = mean_fidelity(apply_student(val_states, theta, arch, n), val_targets)
            history.append({"epoch": float(epoch), "train_fidelity": float(train_fid), "val_fidelity": float(val_fid)})
            if float(val_fid) > best_val:
                best_val = float(val_fid)
                best_theta = theta.detach().clone()
    with torch.no_grad():
        final_train = mean_fidelity(apply_student(train_states, best_theta, arch, n), train_targets)
        test_fid = mean_fidelity(apply_student(test_states, best_theta, arch, n), test_targets)
    return TrainResult(best_val, float(test_fid), float(final_train), len(history), [float(x) for x in best_theta], history)


def smoke_unitary_distillation(teacher: str = "qft", n: int = 2, arch: Architecture | None = None, epochs: int = 1, seed: int = 0):
    arch = arch or Architecture(depth=1, local="ry_rz", entangler="none")
    train = make_states("basis_random", 4, n, seed)
    val = make_states("product", 4, n, seed + 1)
    test = make_states("haar", 4, n, seed + 2)
    result = train_architecture(
        arch,
        n=n,
        train_states=train,
        val_states=val,
        test_states=test,
        train_targets=apply_teacher(teacher, train, n, seed),
        val_targets=apply_teacher(teacher, val, n, seed),
        test_targets=apply_teacher(teacher, test, n, seed),
        epochs=epochs,
        seed=seed,
    )
    return {
        "teacher": teacher,
        "n": n,
        "arch": str(arch),
        "val_fidelity": result.best_val_fidelity,
        "test_fidelity": result.test_fidelity,
    }


__all__ = [
    "Architecture",
    "TrainResult",
    "all_architectures",
    "apply_student",
    "apply_teacher",
    "mean_fidelity",
    "param_count",
    "set_seed",
    "smoke_unitary_distillation",
    "train_architecture",
    "twoq_count",
]
