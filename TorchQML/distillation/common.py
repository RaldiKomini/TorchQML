import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset


def reset_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def dataset_to_numpy(dataset: TensorDataset) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        features, labels = dataset.tensors[:2]
        return (
            features.detach().cpu().numpy().astype(np.float32),
            labels.detach().cpu().numpy(),
        )

    features, labels = [], []
    for xb, yb in dataset:
        features.append(torch.as_tensor(xb).view(-1).cpu())
        labels.append(torch.as_tensor(yb).view(-1).cpu())

    return (
        torch.stack(features).numpy().astype(np.float32),
        torch.cat(labels).numpy(),
    )


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_model_theta(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "effective_theta"):
        return model.effective_theta().detach()
    if hasattr(model, "theta"):
        return model.theta.detach()
    raise AttributeError("Could not find circuit parameters on model.")


__all__ = [
    "as_numpy",
    "clone_state_dict_to_cpu",
    "dataset_to_numpy",
    "get_model_theta",
    "reset_seed",
]
