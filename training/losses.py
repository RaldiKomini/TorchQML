import torch
import torch.nn.functional as F
def softplus(S: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    y_pm = 1 - 2 * yb.to(S.dtype)   # 0->+1, 1->-1
    return F.softplus(-y_pm * S).mean()

__all__ = ["softplus"]