from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from TorchQML.distillation.datasets import DatasetBundle, DatasetConfig


@dataclass(frozen=True)
class TeacherConfig:
    n_estimators: int = 1000
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 0
    n_jobs: int = -1
    tree_method: str = "hist"


@dataclass
class TeacherBundle:
    classifier: object
    train_logits: np.ndarray
    val_logits: np.ndarray
    test_logits: np.ndarray


def fit_xgb_teacher(dataset: DatasetBundle, config: TeacherConfig) -> TeacherBundle:
    from xgboost import XGBClassifier

    y_train = dataset.y_train_np.astype(np.int64)
    n_classes = int(np.max(y_train)) + 1 if y_train.size else 2
    if n_classes <= 2:
        objective = "binary:logistic"
        extra = {}
    else:
        objective = "multi:softprob"
        extra = {"num_class": n_classes}

    classifier = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective=objective,
        eval_metric="logloss" if n_classes <= 2 else "mlogloss",
        random_state=config.random_state,
        tree_method=config.tree_method,
        n_jobs=config.n_jobs,
        **extra,
    )
    classifier.fit(
        dataset.X_train_np,
        y_train,
        eval_set=[(dataset.X_val_np, dataset.y_val_np.astype(np.int64))],
        verbose=False,
    )

    def margins(X):
        raw = classifier.predict(X, output_margin=True).astype(np.float32)
        return raw.reshape(-1) if n_classes <= 2 else raw

    return TeacherBundle(
        classifier=classifier,
        train_logits=margins(dataset.X_train_np),
        val_logits=margins(dataset.X_val_np),
        test_logits=margins(dataset.X_test_np),
    )


def attach_kd_loaders(
    dataset: DatasetBundle,
    teacher: TeacherBundle,
    config: DatasetConfig,
) -> None:
    def make_set(X, y, logits):
        return TensorDataset(
            X.detach().cpu().to(dtype=torch.float32),
            y.detach().cpu(),
            torch.as_tensor(logits, dtype=torch.float32),
        )

    dataset.train_loader_kd = DataLoader(
        make_set(dataset.X_train, dataset.y_train, teacher.train_logits),
        batch_size=config.batch_size_train,
        shuffle=True,
    )
    dataset.val_loader_kd = DataLoader(
        make_set(dataset.X_val, dataset.y_val, teacher.val_logits),
        batch_size=config.batch_size_eval,
        shuffle=False,
    )
    dataset.test_loader_kd = DataLoader(
        make_set(dataset.X_test, dataset.y_test, teacher.test_logits),
        batch_size=config.batch_size_eval,
        shuffle=False,
    )


__all__ = ["TeacherBundle", "TeacherConfig", "attach_kd_loaders", "fit_xgb_teacher"]
