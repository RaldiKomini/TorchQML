import copy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from TorchQML.distillation.common import clone_state_dict_to_cpu


Objective = Literal["task", "kd"]


@dataclass(frozen=True)
class TrainLoopConfig:
    epochs: int
    lr: float = 1e-2
    weight_decay: float = 0.0
    patience: int | None = None
    alpha: float = 0.5
    temperature: float = 4.0
    verbose: bool = False


@dataclass
class TrainResult:
    objective: Objective
    best_val_acc: float
    history: list[dict[str, float]]
    best_state: dict[str, torch.Tensor]
    trained_epochs: int


def kd_binary_loss(student_logit, y_true, teacher_logit, alpha=0.5, temperature=4.0):
    """Blend binary task loss with a soft teacher-logit target."""
    student_logit = student_logit.float().view(-1)
    y_true = y_true.float().view(-1)
    teacher_logit = teacher_logit.float().view(-1)
    task_loss = F.binary_cross_entropy_with_logits(student_logit, y_true)
    soft_target = torch.sigmoid(teacher_logit / temperature)
    kd_loss = F.binary_cross_entropy_with_logits(
        student_logit / temperature,
        soft_target,
    )
    loss = (1.0 - alpha) * task_loss + alpha * (temperature**2) * kd_loss
    return loss, task_loss.detach(), kd_loss.detach()


def kd_multiclass_loss(logits, y_true, teacher_logits, alpha=0.5, temperature=4.0):
    """Blend multiclass cross entropy with KL distillation."""
    y_true = y_true.long().view(-1)
    teacher_logits = teacher_logits.float()
    task_loss = F.cross_entropy(logits, y_true)
    kd_loss = F.kl_div(
        F.log_softmax(logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
    loss = (1.0 - alpha) * task_loss + alpha * (temperature**2) * kd_loss
    return loss, task_loss.detach(), kd_loss.detach()


def unpack_batch(batch):
    if len(batch) == 2:
        xb, yb = batch
        return xb, yb, None
    if len(batch) == 3:
        xb, yb, teacher = batch
        return xb, yb, teacher
    raise ValueError(f"unexpected batch size: {len(batch)}")


def _as_logits(output):
    return output[0] if isinstance(output, tuple) else output


def _is_multiclass(logits: torch.Tensor) -> bool:
    return logits.ndim > 1 and logits.shape[-1] > 1


def _accuracy(logits: torch.Tensor, yb: torch.Tensor) -> float:
    if _is_multiclass(logits):
        pred = logits.argmax(dim=1)
        y = yb.long().view(-1)
    else:
        pred = (torch.sigmoid(logits.view(-1)) >= 0.5).to(yb.device)
        y = yb.float().view(-1)
    return float((pred == y).sum().item()) / max(1, y.numel())


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective: Objective,
    optimizer: torch.optim.Optimizer | None,
    alpha: float,
    temperature: float,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total = 0
    correct = 0.0
    loss_sum = 0.0
    task_sum = 0.0
    kd_sum = 0.0

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            xb, yb, teacher_logits = unpack_batch(batch)
            xb = xb.to(device)
            yb = yb.to(device)
            teacher_logits = None if teacher_logits is None else teacher_logits.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = _as_logits(model(xb))
            # Binary and multiclass models share the same loop; only the loss changes.
            if objective == "kd":
                if teacher_logits is None:
                    raise ValueError("KD training requires teacher logits.")
                if _is_multiclass(logits):
                    loss, task_loss, kd_loss = kd_multiclass_loss(
                        logits,
                        yb,
                        teacher_logits,
                        alpha=alpha,
                        temperature=temperature,
                    )
                else:
                    loss, task_loss, kd_loss = kd_binary_loss(
                        logits,
                        yb,
                        teacher_logits,
                        alpha=alpha,
                        temperature=temperature,
                    )
            elif _is_multiclass(logits):
                loss = F.cross_entropy(logits, yb.long().view(-1))
                task_loss = loss.detach()
                kd_loss = torch.zeros((), device=logits.device)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    yb.float().view(-1),
                )
                task_loss = loss.detach()
                kd_loss = torch.zeros((), device=logits.device)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = yb.numel()
            total += batch_size
            correct += _accuracy(logits.detach(), yb) * batch_size
            loss_sum += float(loss.item()) * batch_size
            task_sum += float(task_loss.item()) * batch_size
            kd_sum += float(kd_loss.item()) * batch_size

    return {
        "loss": loss_sum / max(1, total),
        "task_loss": task_sum / max(1, total),
        "kd_loss": kd_sum / max(1, total),
        "acc": correct / max(1, total),
    }


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader | None,
    val_loader: DataLoader | None,
    device: torch.device,
    objective: Objective,
    config: TrainLoopConfig,
) -> TrainResult:
    if train_loader is None or val_loader is None:
        raise ValueError("train_loader and val_loader must both be provided")

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    best_state = None
    best_val_acc = -1.0
    history = []
    stale = 0

    for epoch in range(1, config.epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            device,
            objective,
            optimizer,
            config.alpha,
            config.temperature,
        )
        val = run_epoch(
            model,
            val_loader,
            device,
            objective,
            None,
            config.alpha,
            config.temperature,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train["loss"],
            "train_task": train["task_loss"],
            "train_kd": train["kd_loss"],
            "train_acc": train["acc"],
            "val_loss": val["loss"],
            "val_acc": val["acc"],
        }
        history.append(row)

        if config.verbose:
            print(
                f"ep {epoch:02d} | train_loss={row['train_loss']:.4f} | "
                f"train_acc={row['train_acc']:.4f} | val_acc={row['val_acc']:.4f}"
            )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            best_state = clone_state_dict_to_cpu(model)
            stale = 0
        else:
            stale += 1
            if config.patience is not None and stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training finished without a best state")

    model.load_state_dict(best_state)
    return TrainResult(
        objective=objective,
        best_val_acc=best_val_acc,
        history=history,
        best_state=copy.deepcopy(best_state),
        trained_epochs=len(history),
    )


__all__ = [
    "Objective",
    "TrainLoopConfig",
    "TrainResult",
    "fit_model",
    "kd_binary_loss",
    "kd_multiclass_loss",
    "run_epoch",
    "unpack_batch",
]
