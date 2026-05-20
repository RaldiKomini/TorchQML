from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from TorchQML.distillation.common import dataset_to_numpy


FMNIST_HANDLE = "zalando-research/fashionmnist"
DEFAULT_KEEP = tuple(range(10))
DEFAULT_SEED = 0
DEFAULT_VAL_FRACTION = 0.2


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    batch_size_train: int = 32
    batch_size_eval: int = 64


@dataclass
class DatasetBundle:
    train_set: TensorDataset
    val_set: TensorDataset
    test_set: TensorDataset
    train_loader_plain: DataLoader
    val_loader_plain: DataLoader
    test_loader_plain: DataLoader
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    X_train_np: np.ndarray
    y_train_np: np.ndarray
    X_val_np: np.ndarray
    y_val_np: np.ndarray
    X_test_np: np.ndarray
    y_test_np: np.ndarray
    train_loader_kd: DataLoader | None = None
    val_loader_kd: DataLoader | None = None
    test_loader_kd: DataLoader | None = None


def load_dataset_bundle(config: DatasetConfig) -> DatasetBundle:
    if not config.path.exists():
        raise FileNotFoundError(f"dataset not found: {config.path}")

    data = torch.load(config.path, weights_only=False)
    train_set = data["train"]
    val_set = data["val"]
    test_set = data["test"]
    X_train, y_train = train_set.tensors[:2]
    X_val, y_val = val_set.tensors[:2]
    X_test, y_test = test_set.tensors[:2]
    X_train_np, y_train_np = dataset_to_numpy(train_set)
    X_val_np, y_val_np = dataset_to_numpy(val_set)
    X_test_np, y_test_np = dataset_to_numpy(test_set)

    return DatasetBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        train_loader_plain=DataLoader(
            train_set, batch_size=config.batch_size_train, shuffle=True
        ),
        val_loader_plain=DataLoader(
            val_set, batch_size=config.batch_size_eval, shuffle=False
        ),
        test_loader_plain=DataLoader(
            test_set, batch_size=config.batch_size_eval, shuffle=False
        ),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_train_np=X_train_np,
        y_train_np=y_train_np,
        X_val_np=X_val_np,
        y_val_np=y_val_np,
        X_test_np=X_test_np,
        y_test_np=y_test_np,
    )


def get_cached_fmnist_dir(handle: str = FMNIST_HANDLE) -> Path:
    from kagglehub.cache import get_cached_path
    from kagglehub.handle import parse_dataset_handle

    dataset_dir = Path(get_cached_path(parse_dataset_handle(handle)))
    version_dirs = sorted((dataset_dir / "versions").glob("*"))
    if not version_dirs:
        raise FileNotFoundError(
            f"Cached FashionMNIST dataset not found for handle {handle}."
        )
    return version_dirs[-1]


def _load_fmnist_frames(handle: str = FMNIST_HANDLE):
    import pandas as pd

    cached_dir = get_cached_fmnist_dir(handle)
    train_csv = cached_dir / "fashion-mnist_train.csv"
    test_csv = cached_dir / "fashion-mnist_test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Expected FashionMNIST CSVs at {cached_dir}.")
    return pd.read_csv(train_csv), pd.read_csv(test_csv)


def _split_fmnist(
    *,
    keep: tuple[int, ...],
    seed: int,
    val_fraction: float,
    remap_labels: bool,
    handle: str,
):
    from sklearn.model_selection import train_test_split

    train_df, test_df = _load_fmnist_frames(handle)
    train_df = train_df[train_df["label"].isin(keep)].reset_index(drop=True)
    test_df = test_df[test_df["label"].isin(keep)].reset_index(drop=True)

    if remap_labels:
        label_map = {label: idx for idx, label in enumerate(keep)}
        train_df["label"] = train_df["label"].map(label_map)
        test_df["label"] = test_df["label"].map(label_map)

    X_train_full = train_df.drop(columns=["label"]).to_numpy(dtype="float32")
    y_train_full = train_df["label"].to_numpy(dtype="int64")
    X_test_full = test_df.drop(columns=["label"]).to_numpy(dtype="float32")
    y_test = test_df["label"].to_numpy(dtype="int64")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_train_full,
    )
    return X_train, y_train, X_val, y_val, X_test_full, y_test


def build_fmnist_pca_dataset(
    path: Path,
    *,
    keep: tuple[int, ...] = (0, 1),
    n_components: int = 16,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    remap_labels: bool = False,
    handle: str = FMNIST_HANDLE,
    label_dtype: torch.dtype | None = None,
) -> Path:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_train, y_train, X_val, y_val, X_test, y_test = _split_fmnist(
        keep=keep,
        seed=seed,
        val_fraction=val_fraction,
        remap_labels=remap_labels,
        handle=handle,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = torch.tensor(pca.fit_transform(X_train_scaled), dtype=torch.float32)
    X_val_pca = torch.tensor(pca.transform(X_val_scaled), dtype=torch.float32)
    X_test_pca = torch.tensor(pca.transform(X_test_scaled), dtype=torch.float32)

    mean = X_train_pca.mean(dim=0, keepdim=True)
    std = X_train_pca.std(dim=0, keepdim=True) + 1e-6
    X_train_pca = (X_train_pca - mean) / std
    X_val_pca = (X_val_pca - mean) / std
    X_test_pca = (X_test_pca - mean) / std

    if label_dtype is None:
        label_dtype = torch.float32 if len(keep) == 2 and not remap_labels else torch.long

    bundle = {
        "train": TensorDataset(X_train_pca, torch.tensor(y_train, dtype=label_dtype)),
        "val": TensorDataset(X_val_pca, torch.tensor(y_val, dtype=label_dtype)),
        "test": TensorDataset(X_test_pca, torch.tensor(y_test, dtype=label_dtype)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return path


def pad_and_normalize_for_amplitude(X: np.ndarray, target_dim: int) -> torch.Tensor:
    X = X.astype(np.float32) / 255.0
    if X.shape[1] > target_dim:
        raise ValueError(f"expected at most {target_dim} features, got {X.shape[1]}")
    if X.shape[1] < target_dim:
        X = np.pad(X, ((0, 0), (0, target_dim - X.shape[1])), mode="constant")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return torch.tensor(X / np.maximum(norms, 1e-8), dtype=torch.float32)


def build_fmnist_amplitude_dataset(
    path: Path,
    *,
    keep: tuple[int, ...] = DEFAULT_KEEP,
    num_qubits: int = 10,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    remap_labels: bool = True,
    handle: str = FMNIST_HANDLE,
) -> Path:
    target_dim = 1 << num_qubits
    X_train, y_train, X_val, y_val, X_test, y_test = _split_fmnist(
        keep=keep,
        seed=seed,
        val_fraction=val_fraction,
        remap_labels=remap_labels,
        handle=handle,
    )
    bundle = {
        "train": TensorDataset(
            pad_and_normalize_for_amplitude(X_train, target_dim),
            torch.tensor(y_train, dtype=torch.long),
        ),
        "val": TensorDataset(
            pad_and_normalize_for_amplitude(X_val, target_dim),
            torch.tensor(y_val, dtype=torch.long),
        ),
        "test": TensorDataset(
            pad_and_normalize_for_amplitude(X_test, target_dim),
            torch.tensor(y_test, dtype=torch.long),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return path


def ensure_dataset(path: Path, builder, *args, **kwargs) -> Path:
    return path if path.exists() else builder(path, *args, **kwargs)


__all__ = [
    "DEFAULT_KEEP",
    "DEFAULT_SEED",
    "DEFAULT_VAL_FRACTION",
    "DatasetBundle",
    "DatasetConfig",
    "FMNIST_HANDLE",
    "build_fmnist_amplitude_dataset",
    "build_fmnist_pca_dataset",
    "ensure_dataset",
    "get_cached_fmnist_dir",
    "load_dataset_bundle",
    "pad_and_normalize_for_amplitude",
]
