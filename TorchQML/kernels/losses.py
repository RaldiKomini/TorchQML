import torch
import torch.nn.functional as F


def _same_diff_masks(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = y.view(-1)
    eye = torch.eye(y.numel(), device=y.device, dtype=torch.bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = (y[:, None] != y[None, :]) & ~eye
    return same, diff


def _pm_labels(y: torch.Tensor) -> torch.Tensor:
    y = y.float().view(-1)
    values = torch.unique(y.detach())
    if torch.all((values == 0) | (values == 1)):
        return 2.0 * y - 1.0
    return torch.sign(y).clamp(min=-1.0, max=1.0)


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    return K - K.mean(dim=1, keepdim=True) - K.mean(dim=0, keepdim=True) + K.mean()


def label_kernel(y: torch.Tensor) -> torch.Tensor:
    y_pm = _pm_labels(y)
    return y_pm[:, None] * y_pm[None, :]


def kernel_alignment(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Centered kernel-target alignment."""
    Kc = center_kernel(K)
    Yc = center_kernel(label_kernel(y).to(device=K.device, dtype=K.dtype))
    num = (Kc * Yc).sum()
    den = torch.sqrt((Kc * Kc).sum() * (Yc * Yc).sum()).clamp_min(1e-12)
    return num / den


def alignment_loss(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -kernel_alignment(K, y)


def mu_gap_loss(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Maximize the gap between same-class and different-class similarities."""
    same, diff = _same_diff_masks(y.to(K.device))
    if not same.any() or not diff.any():
        return torch.zeros((), device=K.device, dtype=K.dtype)
    return -(K[same].mean() - K[diff].mean())


def kernel_triplet_loss(K: torch.Tensor, y: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    y = y.to(K.device).view(-1)
    losses = []
    for i in range(K.shape[0]):
        pos = torch.nonzero(y == y[i], as_tuple=False).view(-1)
        neg = torch.nonzero(y != y[i], as_tuple=False).view(-1)
        pos = pos[pos != i]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        p = pos[torch.randint(pos.numel(), (1,), device=K.device)]
        n = neg[torch.randint(neg.numel(), (1,), device=K.device)]
        losses.append(F.softplus(margin - (K[i, p] - K[i, n])).view(()))
    if not losses:
        return torch.zeros((), device=K.device, dtype=K.dtype)
    return torch.stack(losses).mean()


def hs_loss(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE between centered quantum kernel and same-class label kernel."""
    same = (y.view(-1, 1) == y.view(1, -1)).to(device=K.device, dtype=K.dtype)
    eye = torch.eye(K.size(0), device=K.device, dtype=torch.bool)
    return ((center_kernel(K).masked_fill(eye, 0.0) - center_kernel(same).masked_fill(eye, 0.0)) ** 2).mean()


def hsic_loss(K: torch.Tensor, y: torch.Tensor, margin_weight: float = 0.1) -> torch.Tensor:
    """Negative HSIC with a small same/different-class margin reward."""
    n = K.size(0)
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    Y = label_kernel(y).to(device=K.device, dtype=K.dtype)
    Kc = H @ K @ H
    Yc = H @ Y @ H
    hsic = torch.trace(Kc @ Yc) / ((n - 1) ** 2 + 1e-8)
    return -hsic + margin_weight * mu_gap_loss(K, y)


def margin_loss(K: torch.Tensor, y: torch.Tensor, target: float = 0.05, std_weight: float = 0.2) -> torch.Tensor:
    same, diff = _same_diff_masks(y.to(K.device))
    if not same.any() or not diff.any():
        return torch.zeros((), device=K.device, dtype=K.dtype)
    margin = K[same].mean() - K[diff].mean()
    spread = K[same].std(unbiased=False) + K[diff].std(unbiased=False)
    return F.softplus(torch.as_tensor(target, device=K.device, dtype=K.dtype) - margin) - std_weight * spread


def compute_kernel_loss(name: str, K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    losses = {
        "align": alignment_loss,
        "alignment": alignment_loss,
        "mugap": mu_gap_loss,
        "triplet": kernel_triplet_loss,
        "hs": hs_loss,
        "hsic": hsic_loss,
        "margin": margin_loss,
    }
    try:
        return losses[name](K, y)
    except KeyError as exc:
        raise ValueError(f"Unknown kernel loss: {name}") from exc
