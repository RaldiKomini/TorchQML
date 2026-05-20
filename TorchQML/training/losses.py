import torch
import torch.nn.functional as F


def softplus(S: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """Binary softplus loss for scores where class 0 should be positive."""
    y_pm = 1 - 2 * yb.to(S.dtype)
    return F.softplus(-y_pm * S).mean()


def contrastive(l_vec: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pull same-label entanglement vectors together and push different labels apart."""
    batch_size = l_vec.size(0)
    li = l_vec.unsqueeze(1)
    lj = l_vec.unsqueeze(0)
    dist = torch.norm(li - lj, dim=-1)

    same = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    diff = 1.0 - same
    mask = 1.0 - torch.eye(batch_size, device=l_vec.device)

    return ((same * dist) - (diff * dist)).mul(mask).mean()


def hard_entanglement_loss(
    l_vec: torch.Tensor,
    y: torch.Tensor,
    target_high: float = 0.5,
) -> torch.Tensor:
    """Penalize class 0 for high entanglement and class 1 for low entanglement."""
    l_mean = l_vec.mean(dim=1)
    y0 = (y == 0).float()
    y1 = (y == 1).float()
    return (y0 * l_mean + y1 * (target_high - l_mean)).mean()


def spooky_loss(output, y: torch.Tensor) -> torch.Tensor:
    """Loss used by `SpookyModel` training."""
    _, l_vec = output
    return hard_entanglement_loss(l_vec, y)


__all__ = ["softplus", "spooky_loss"]
