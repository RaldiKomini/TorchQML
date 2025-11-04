import math
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from qml_lib.config import DEVICE

def make_moons_torch(
    n_samples=1000,
    noise=0.2,
    test_size=0.2,
    val_size=0.1,
    standardize=True,
    to_angle="pi",   # None | "pi" | "halfpi"
    seed=402,
    device=DEVICE,
):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    if standardize:
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    if to_angle == "pi":
        X = X / (abs(X).max(axis=0, keepdims=True) + 1e-8)
        X = X * math.pi          # <- no .item()
    elif to_angle == "halfpi":
        X = X / (abs(X).max(axis=0, keepdims=True) + 1e-8)
        X = X * (0.5 * math.pi)  # <- no .item()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size+val_size), stratify=y, random_state=seed
    )
    rel_val = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.0
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1-rel_val), stratify=y_tmp, random_state=seed
    )

    tt = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    tl = lambda a: torch.tensor(a, dtype=torch.long, device=device)

    return {
        "X_train": tt(X_train), "y_train": tl(y_train),
        "X_val":   tt(X_val),   "y_val":   tl(y_val),
        "X_test":  tt(X_test),  "y_test":  tl(y_test),
    }
