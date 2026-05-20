import copy
import random

import numpy as np
import torch
import torch.nn.functional as F


def reset_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kd_binary_loss(student_logit, y_true, teacher_logit, *, alpha: float = 0.5, temperature: float = 4.0):
    """Binary task loss plus softened teacher-logit distillation."""
    student_logit = student_logit.float().view(-1)
    y_true = y_true.float().view(-1)
    teacher_logit = teacher_logit.float().view(-1)

    task_loss = F.binary_cross_entropy_with_logits(student_logit, y_true)
    soft_target = torch.sigmoid(teacher_logit / temperature)
    kd_loss = F.binary_cross_entropy_with_logits(student_logit / temperature, soft_target)
    loss = (1.0 - alpha) * task_loss + alpha * (temperature**2) * kd_loss
    return loss, task_loss.detach(), kd_loss.detach()


def _as_logits(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def evaluate_binary_logits(model, loader, device, *, alpha: float = 0.5, temperature: float = 4.0):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for xb, yb, tb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        tb = tb.to(device).float()
        logits = _as_logits(model(xb)).view(-1)
        loss, _, _ = kd_binary_loss(logits, yb, tb, alpha=alpha, temperature=temperature)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()

    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


def fit_binary_kd(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int = 15,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    alpha: float = 0.5,
    temperature: float = 4.0,
):
    """Train a binary student model from labels and teacher logits."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_acc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        task_sum = 0.0
        kd_sum = 0.0

        for xb, yb, tb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            tb = tb.to(device).float()

            logits = _as_logits(model(xb)).view(-1)
            loss, task_loss, distill_loss = kd_binary_loss(
                logits,
                yb,
                tb,
                alpha=alpha,
                temperature=temperature,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pred = (torch.sigmoid(logits.detach()) >= 0.5).float()
            batch = yb.numel()
            correct += (pred == yb).sum().item()
            total += batch
            loss_sum += loss.item() * batch
            task_sum += task_loss.item() * batch
            kd_sum += distill_loss.item() * batch

        val = evaluate_binary_logits(
            model,
            val_loader,
            device,
            alpha=alpha,
            temperature=temperature,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, total),
            "train_task": task_sum / max(1, total),
            "train_kd": kd_sum / max(1, total),
            "train_acc": correct / max(1, total),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
        }
        history.append(row)

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
