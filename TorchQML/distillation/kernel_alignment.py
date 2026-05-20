from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.runtime import compute_states, fidelity_kernel_from_states
from TorchQML.distillation.common import clone_state_dict_to_cpu
from TorchQML.distillation.experiments import QSVMConfig
from TorchQML.encoding.amplitude import AmpEnc
from TorchQML.kernels.quantum import fidelity_kernel_matrix, fit_precomputed_svc


@dataclass(frozen=True)
class PairwiseTrainConfig:
    epochs: int
    lr: float = 0.01
    weight_decay: float = 0.0
    patience: int | None = None
    teacher_temperature: float = 4.0
    verbose: bool = False


@dataclass(frozen=True)
class KernelSubsetConfig:
    search_train_limit: int = 512
    search_val_limit: int = 512
    final_train_limit: int = 1024
    final_val_limit: int = 512
    final_test_limit: int | None = 2000


@dataclass
class PairwiseTrainResult:
    best_val_loss: float
    history: list[dict[str, float]]
    best_state: dict[str, torch.Tensor]
    trained_epochs: int


@dataclass(frozen=True)
class KernelView:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


class AmplitudeFeatureMap(nn.Module):
    """Trainable amplitude-encoded feature map for QSVM-first distillation."""

    def __init__(self, circ, spec):
        super().__init__()
        self.circ = circ
        self.spec = spec
        self.theta = nn.Parameter(
            0.1 * torch.randn(spec.tlen, device=DEVICE, dtype=DTYPE)
        )
        self.amp_enc = AmpEnc(spec.num_qubits)

    def states(self, xb):
        xb = xb.to(device=self.theta.device, dtype=torch.float32)
        return compute_states(xb, self.theta, self.circ, amp_enc=self.amp_enc)


def stratified_subset_indices(labels: torch.Tensor, limit: int | None, seed: int) -> np.ndarray:
    from sklearn.model_selection import train_test_split

    y = labels.detach().cpu().numpy().astype(np.int64)
    indices = np.arange(len(y))
    if limit is None or limit >= len(indices):
        return indices
    chosen, _ = train_test_split(indices, train_size=limit, stratify=y, random_state=seed)
    return np.sort(chosen)


def subset_tensor(x: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    return x.index_select(0, torch.as_tensor(indices, dtype=torch.long))


def build_kernel_view(dataset, config: KernelSubsetConfig, seed: int, *, final: bool) -> KernelView:
    train_limit = config.final_train_limit if final else config.search_train_limit
    val_limit = config.final_val_limit if final else config.search_val_limit
    test_limit = config.final_test_limit if final else config.search_val_limit
    train_idx = stratified_subset_indices(dataset.y_train, train_limit, seed)
    val_idx = stratified_subset_indices(dataset.y_val, val_limit, seed + 1)
    test_idx = stratified_subset_indices(dataset.y_test, test_limit, seed + 2)
    return KernelView(
        X_train=subset_tensor(dataset.X_train, train_idx),
        y_train=subset_tensor(dataset.y_train, train_idx),
        X_val=subset_tensor(dataset.X_val, val_idx),
        y_val=subset_tensor(dataset.y_val, val_idx),
        X_test=subset_tensor(dataset.X_test, test_idx),
        y_test=subset_tensor(dataset.y_test, test_idx),
    )


def teacher_similarity_matrix(teacher_logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    probs = torch.softmax(teacher_logits / temperature, dim=1)
    target = (probs @ probs.T).real
    target.fill_diagonal_(1.0)
    return target


def teacher_kernel_alignment_loss(states, teacher_logits, *, temperature: float):
    kernel = fidelity_kernel_from_states(states, states).real
    target = teacher_similarity_matrix(teacher_logits, temperature=temperature)
    loss = torch.mean((kernel - target).pow(2))
    teacher_labels = teacher_logits.argmax(dim=1)
    same = teacher_labels[:, None].eq(teacher_labels[None, :])
    eye = torch.eye(same.shape[0], dtype=torch.bool, device=same.device)
    same = same & ~eye
    diff = (~same) & ~eye
    kernel_gap = torch.zeros((), device=states.device)
    target_gap = torch.zeros((), device=states.device)
    if same.any() and diff.any():
        kernel_gap = kernel[same].mean() - kernel[diff].mean()
        target_gap = target[same].mean() - target[diff].mean()
    return loss, kernel_gap.detach(), target_gap.detach()


def run_kernel_epoch(model: AmplitudeFeatureMap, loader, *, temperature: float, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total = 0
    loss_sum = 0.0
    kernel_gap_sum = 0.0
    target_gap_sum = 0.0
    with torch.set_grad_enabled(is_training):
        for xb, _, teacher_logits in loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            teacher_logits = teacher_logits.to(DEVICE, dtype=torch.float32)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            states = model.states(xb)
            loss, kernel_gap, target_gap = teacher_kernel_alignment_loss(
                states,
                teacher_logits,
                temperature=temperature,
            )
            if is_training:
                loss.backward()
                optimizer.step()
            batch_size = xb.shape[0]
            total += batch_size
            loss_sum += float(loss.item()) * batch_size
            kernel_gap_sum += float(kernel_gap.item()) * batch_size
            target_gap_sum += float(target_gap.item()) * batch_size
    return {
        "loss": loss_sum / max(1, total),
        "kernel_gap": kernel_gap_sum / max(1, total),
        "target_gap": target_gap_sum / max(1, total),
    }


def fit_feature_map(model: AmplitudeFeatureMap, train_loader, val_loader, config: PairwiseTrainConfig):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    best_val_loss = float("inf")
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, config.epochs + 1):
        train = run_kernel_epoch(
            model,
            train_loader,
            temperature=config.teacher_temperature,
            optimizer=optimizer,
        )
        val = run_kernel_epoch(
            model,
            val_loader,
            temperature=config.teacher_temperature,
            optimizer=None,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train["loss"],
            "train_kernel_gap": train["kernel_gap"],
            "train_target_gap": train["target_gap"],
            "val_loss": val["loss"],
            "val_kernel_gap": val["kernel_gap"],
            "val_target_gap": val["target_gap"],
        }
        history.append(row)
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = clone_state_dict_to_cpu(model)
            stale = 0
        else:
            stale += 1
            if config.patience is not None and stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("feature-map training finished without a best state")
    model.load_state_dict(best_state)
    return PairwiseTrainResult(best_val_loss, history, best_state, len(history))


def qsvm_accuracy_on_view(model: AmplitudeFeatureMap, view: KernelView, qsvm_config: QSVMConfig, *, split: str):
    y_train = view.y_train.detach().cpu().numpy().astype(np.int64)
    if split == "val":
        X_eval = view.X_val
        y_eval = view.y_val.detach().cpu().numpy().astype(np.int64)
    elif split == "test":
        X_eval = view.X_test
        y_eval = view.y_test.detach().cpu().numpy().astype(np.int64)
    else:
        raise ValueError(f"unknown split: {split}")
    K_train = fidelity_kernel_matrix(
        view.X_train.to(DEVICE),
        view.X_train.to(DEVICE),
        model.theta.detach(),
        model.circ,
        amp_enc=model.amp_enc,
        blockA=qsvm_config.block_a,
        blockB=qsvm_config.block_b,
        symmetric=True,
    )
    K_eval = fidelity_kernel_matrix(
        X_eval.to(DEVICE),
        view.X_train.to(DEVICE),
        model.theta.detach(),
        model.circ,
        amp_enc=model.amp_enc,
        blockA=qsvm_config.block_a,
        blockB=qsvm_config.block_b,
    )
    _, pred = fit_precomputed_svc(K_train, y_train, K_eval, C=qsvm_config.C)
    return float((pred == y_eval).mean())


__all__ = [
    "AmplitudeFeatureMap",
    "KernelSubsetConfig",
    "KernelView",
    "PairwiseTrainConfig",
    "PairwiseTrainResult",
    "build_kernel_view",
    "fit_feature_map",
    "qsvm_accuracy_on_view",
    "run_kernel_epoch",
    "stratified_subset_indices",
    "subset_tensor",
    "teacher_kernel_alignment_loss",
    "teacher_similarity_matrix",
]
