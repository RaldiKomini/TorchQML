import json
import os
from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    """Basic training settings for `Trainer`."""

    epochs: int = 50
    lr: float = 0.02
    patience: int = 5
    device: torch.device | None = None


class Trainer:
    """Small training loop for score-based TorchQML models."""

    def __init__(self, model, loss_fn, metric_fn, cfg: TrainConfig):
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.cfg = cfg

        if self.cfg.device is None:
            self.cfg.device = next(model.parameters()).device

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def fit(self, train_loader, val_loader=None):
        """Train the model and return history, best weights, and best validation accuracy."""
        best_val = -1.0
        best_state = None
        no_improve = 0
        history = {"train_loss": [], "val_acc": [], "train_acc": []}

        for epoch in range(self.cfg.epochs):
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            val_acc = None

            if val_loader is not None:
                val_acc = self._eval_metric(val_loader)
                if val_acc > best_val:
                    best_val = val_acc
                    no_improve = 0
                    best_state = self._state_dict_cpu()
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.patience:
                        print(f"Early stopping at epoch {epoch:03d} (best val_acc={best_val:.2f})")
                        break

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_acc is None:
                print(f"epoch {epoch:03d} | loss={train_loss:.4f} | acc={train_acc:.2f}")
            else:
                print(
                    f"epoch {epoch:03d} | loss={train_loss:.4f} | "
                    f"train_acc={train_acc:.2f} | val_acc={val_acc:.2f}"
                )

        if best_state is None:
            best_state = self._state_dict_cpu()

        return history, best_state, best_val

    def _train_epoch(self, train_loader, epoch: int) -> tuple[float, float]:
        self.model.train()
        loss_sum = 0.0
        acc_sum = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(self.cfg.device)
            yb = yb.to(self.cfg.device)

            output = self.model(xb)
            loss = self.loss_fn(output, yb)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                loss_sum += float(loss.item())
                acc_sum += float(self._metric(output, yb))
                n_batches += 1
                if (n_batches % 50) == 0:
                    pct = 100.0 * n_batches / len(train_loader)
                    print(f"epoch {epoch:03d} {pct:5.1f}% ({n_batches}/{len(train_loader)})")

        return loss_sum / max(1, n_batches), acc_sum / max(1, n_batches)

    def _metric(self, output, yb):
        if isinstance(output, tuple):
            score, l_vec = output
            metric_input = (score.detach(), l_vec.detach(), self.model.tau.item())
            return self.metric_fn(metric_input, yb)
        return self.metric_fn(output.detach(), yb)

    def _state_dict_cpu(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def _eval_metric(self, loader):
        """Evaluate only the configured metric over a loader."""
        self.model.eval()
        acc_sum = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(self.cfg.device)
            yb = yb.to(self.cfg.device)
            acc_sum += float(self._metric(self.model(xb), yb))
            n_batches += 1
        return acc_sum / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, loader, state_dict=None):
        """Run a loader and return average accuracy, scores, and labels."""
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        acc_sum = 0.0
        n_batches = 0
        scores_all, labels_all, preds_all = [], [], []

        for xb, yb in loader:
            xb = xb.to(self.cfg.device)
            yb = yb.to(self.cfg.device)

            output = self.model(xb)
            acc_sum += float(self._metric(output, yb))
            n_batches += 1

            if isinstance(output, tuple):
                score, _ = output
                preds = (score.detach() > self.model.tau.detach()).long()
                scores_all.append(score.detach().cpu())
            else:
                preds = (output.detach() <= 0).long()
                scores_all.append(output.detach().cpu())

            labels_all.append(yb.detach().cpu())
            preds_all.append(preds.detach().cpu())

        return {
            "acc": acc_sum / max(1, n_batches),
            "scores": torch.cat(scores_all),
            "labels": torch.cat(labels_all),
            "preds": torch.cat(preds_all),
        }


def save_run(out_dir, *, history, best_state, best_val_acc, test_out, extra_metrics=None, run_cfg=None):
    """Save model weights, history, metrics, and test tensors for one run."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "best_state.pt"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    metrics = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_out["acc"]),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    if run_cfg:
        metrics.update(run_cfg)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "scores": test_out["scores"],
            "labels": test_out["labels"],
            "preds": test_out["preds"],
        },
        os.path.join(out_dir, "test_scores_labels.pt"),
    )


__all__ = ["TrainConfig", "Trainer", "save_run"]
