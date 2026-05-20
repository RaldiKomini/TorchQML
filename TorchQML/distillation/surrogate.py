from dataclasses import asdict, is_dataclass
from typing import Callable, Hashable, Sequence

import numpy as np


class ConstantRegressor:
    """Small fallback model used before there is enough data for a GP."""

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = float(mean)
        self.std = float(std)

    def predict(self, X, return_std: bool = False):
        X = np.asarray(X)
        mu = np.full(len(X), self.mean, dtype=np.float64)
        if return_std:
            return mu, np.full(len(X), self.std, dtype=np.float64)
        return mu


def default_encoder(item) -> np.ndarray:
    """Encode architecture descriptors into a numeric vector for sklearn models."""
    if is_dataclass(item):
        values = asdict(item).values()
    elif isinstance(item, tuple):
        values = item
    else:
        values = vars(item).values()

    encoded = []
    for value in values:
        if isinstance(value, bool):
            encoded.append(float(value))
        elif isinstance(value, (int, float)):
            encoded.append(float(value))
        else:
            # This keeps categorical choices usable without creating a large dependency.
            encoded.append(float(abs(hash(value)) % 10000) / 10000.0)
    return np.asarray(encoded, dtype=np.float32)


def encode_items(items: Sequence[Hashable], encoder: Callable[[Hashable], np.ndarray] = default_encoder):
    if not items:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([encoder(item) for item in items], axis=0)


def fit_gp(X: np.ndarray, y: np.ndarray):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return ConstantRegressor(mean=0.5, std=0.25)
    if len(y) == 1:
        return ConstantRegressor(mean=float(y[0]), std=0.10)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def fit_mf_surrogate(
    low_db: dict[Hashable, float],
    high_db: dict[Hashable, float],
    encoder: Callable[[Hashable], np.ndarray] = default_encoder,
):
    """Fit a low-fidelity GP plus a GP for the high-minus-low correction."""
    low_items = list(low_db.keys())
    X_low = encode_items(low_items, encoder)
    y_low = np.array([low_db[item] for item in low_items], dtype=np.float64)
    gp_low = fit_gp(X_low, y_low)

    overlap = [item for item in high_db if item in low_db]
    if not overlap:
        raise RuntimeError("Need at least one item with both low and high fidelity.")

    X_overlap = encode_items(overlap, encoder)
    y_low_overlap = np.array([low_db[item] for item in overlap], dtype=np.float64)
    y_high_overlap = np.array([high_db[item] for item in overlap], dtype=np.float64)
    rho = float(np.dot(y_low_overlap, y_high_overlap) / (np.dot(y_low_overlap, y_low_overlap) + 1e-8))
    gp_delta = fit_gp(X_overlap, y_high_overlap - rho * y_low_overlap)
    return gp_low, gp_delta, rho


def predict_high_batch(
    items: Sequence[Hashable],
    low_db: dict[Hashable, float],
    gp_low,
    gp_delta,
    rho: float,
    encoder: Callable[[Hashable], np.ndarray] = default_encoder,
):
    X = encode_items(items, encoder)
    mu_low, sd_low = gp_low.predict(X, return_std=True)
    mu_low = np.asarray(mu_low, dtype=np.float64)
    sd_low = np.asarray(sd_low, dtype=np.float64)

    for idx, item in enumerate(items):
        if item in low_db:
            mu_low[idx] = low_db[item]
            sd_low[idx] = 1e-8

    mu_delta, sd_delta = gp_delta.predict(X, return_std=True)
    mu_high = rho * mu_low + np.asarray(mu_delta, dtype=np.float64)
    sd_high = np.sqrt((rho * sd_low) ** 2 + np.asarray(sd_delta, dtype=np.float64) ** 2)
    return mu_high, sd_high


def select_ucb(items, low_db, high_db, beta: float = 1.0, encoder=default_encoder):
    """Pick the untested item with the largest upper-confidence-bound score."""
    gp_low, gp_delta, rho = fit_mf_surrogate(low_db, high_db, encoder)
    candidates = [item for item in items if item not in high_db]
    mu, sd = predict_high_batch(candidates, low_db, gp_low, gp_delta, rho, encoder)
    acquisition = mu + beta * sd
    idx = int(np.argmax(acquisition))
    return candidates[idx], float(mu[idx]), float(sd[idx]), float(acquisition[idx]), rho


__all__ = [
    "ConstantRegressor",
    "default_encoder",
    "encode_items",
    "fit_gp",
    "fit_mf_surrogate",
    "predict_high_batch",
    "select_ucb",
]
