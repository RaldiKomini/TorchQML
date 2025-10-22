import torch
import os

DTYPE = torch.complex64


_env = os.getenv("QML_DEVICE")

def _pick_device(env: str | None) -> torch.device:
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _pick_device(_env)