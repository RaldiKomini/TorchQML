import numpy as np
import torch

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.runtime import compute_states, z_expectations_from_states
from TorchQML.kernels.losses import kernel_alignment
from TorchQML.kernels.quantum import fidelity_kernel_matrix


def _balanced_subset(X, y, max_per_class: int = 32):
    labels = torch.unique(y)
    if labels.numel() < 2:
        return X, y
    parts = []
    for label in labels[:2]:
        idx = torch.nonzero(y == label, as_tuple=False).view(-1)[:max_per_class]
        parts.append(idx)
    idx = torch.cat(parts)
    idx = idx[torch.randperm(idx.numel())]
    return X[idx], y[idx]


def make_z_trainability_score(X, theta, nq: int, *, amp_enc=None, eps: float = 1e-8):
    """Criterion factory that scores circuits by trainable Z gradients."""
    X = X[:32].to(DEVICE)
    theta = theta.to(device=DEVICE, dtype=DTYPE)

    def criterion(circ):
        theta_var = theta.clone().detach().requires_grad_(True)
        states = compute_states(X, theta_var, circ, amp_enc=amp_enc)
        zvals = z_expectations_from_states(states, nq)
        loss = zvals.mean() + 0.0 * theta_var.real.sum()
        loss.backward()
        grad = theta_var.grad
        if grad is None:
            return -1e9
        grad = grad[torch.isfinite(grad)]
        grad = grad[grad.abs() > 1e-10]
        if grad.numel() == 0:
            return -1e9
        return torch.log(torch.var(grad.real, unbiased=False) + eps).item()

    return criterion


def make_z_separation_score(X, y, theta, nq: int, *, amp_enc=None, eps: float = 1e-8):
    """Criterion factory that scores class separation in Z-expectation space."""
    Xs, ys = _balanced_subset(X.to(DEVICE), y.to(DEVICE))
    theta = theta.to(device=DEVICE, dtype=DTYPE)

    def criterion(circ):
        with torch.no_grad():
            states = compute_states(Xs, theta, circ, amp_enc=amp_enc)
            zvals = z_expectations_from_states(states, nq).real
        labels = torch.unique(ys)
        if labels.numel() < 2:
            return -1e9
        z0 = zvals[ys == labels[0]]
        z1 = zvals[ys == labels[1]]
        if z0.numel() == 0 or z1.numel() == 0:
            return -1e9
        mean_gap = (z1.mean(dim=0) - z0.mean(dim=0)).norm()
        pooled_var = z0.var(dim=0, unbiased=False).mean() + z1.var(dim=0, unbiased=False).mean()
        return torch.log(mean_gap / (pooled_var.sqrt() + eps) + eps).item()

    return criterion


def make_kernel_alignment_criterion(X, y, theta, *, amp_enc=None, max_samples: int = 96):
    """Criterion factory for centered quantum kernel-target alignment."""
    Xs = X[:max_samples].to(DEVICE)
    ys = y[:max_samples].to(DEVICE)
    theta = theta.to(device=DEVICE, dtype=DTYPE)

    def criterion(circ):
        K = fidelity_kernel_matrix(Xs, Xs, theta, circ, amp_enc=amp_enc, symmetric=True)
        Kt = torch.as_tensor(K, device=DEVICE)
        return float(kernel_alignment(Kt, ys).item())

    return criterion


def make_overlap_criterion(Xtr, ytr, Xval, yval, theta, *, amp_enc=None):
    """Nearest-class overlap score using train/validation fidelity kernels."""
    Xtr = Xtr.to(DEVICE)
    Xval = Xval.to(DEVICE)
    ytr_np = ytr.detach().cpu().numpy() if torch.is_tensor(ytr) else np.asarray(ytr)
    yval_np = yval.detach().cpu().numpy() if torch.is_tensor(yval) else np.asarray(yval)
    labels = np.unique(ytr_np)
    if labels.size < 2:
        raise ValueError("overlap criterion needs at least two classes")
    neg_label, pos_label = labels[0], labels[-1]
    theta = theta.to(device=DEVICE, dtype=DTYPE)

    def criterion(circ):
        K = fidelity_kernel_matrix(Xval, Xtr, theta, circ, amp_enc=amp_enc)
        sim_pos = K[:, ytr_np == pos_label].mean(axis=1)
        sim_neg = K[:, ytr_np == neg_label].mean(axis=1)
        pred = np.where(sim_pos > sim_neg, pos_label, neg_label)
        acc = float((pred == yval_np).mean())
        margin_pos = (sim_pos[yval_np == pos_label] - sim_neg[yval_np == pos_label]).mean()
        margin_neg = (sim_neg[yval_np == neg_label] - sim_pos[yval_np == neg_label]).mean()
        return acc + 0.1 * float(0.5 * (margin_pos + margin_neg))

    return criterion


__all__ = [
    "make_kernel_alignment_criterion",
    "make_overlap_criterion",
    "make_z_separation_score",
    "make_z_trainability_score",
]
