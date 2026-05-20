import torch
import os

DTYPE = torch.complex64


def _pick_device(env: str | None) -> torch.device:
    """Choose the runtime device, keeping CPU as the safe default."""
    if env:
        return torch.device(env)
    return torch.device("cpu")


DEVICE = _pick_device(os.getenv("QML_DEVICE"))
__all__ = ["DEVICE", "DTYPE"]
