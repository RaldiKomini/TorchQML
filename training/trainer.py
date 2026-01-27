from dataclasses import dataclass
import torch

@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 0.02
    patience: int = 5
    device: torch.device = None


class Trainer:
    def __init__(self, model, loss_fn, metric_fn, cfg):
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.cfg = cfg

        if self.cfg.device is None:
            self.cfg.device = next(model.parameters()).device

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def fit(self, train_loader, val_loader = None):
        best_val = -1.0
        best_state = None
        no_imporve = 0
        history = {"train_loss": [], "val_acc": [], "train_acc": []}

        for epoch in range(self.cfg.epochs):
            self.model.train()
            loss_sum = 0.0
            acc_sum = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(self.cfg.device)
                yb = yb.to(self.cfg.device)

                S = self.model(xb)
                loss = self.loss_fn(S, yb)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                with torch.no_grad():
                    loss_sum += float(loss.item())
                    acc_sum += float(self.metric_fn(S.detach(), yb))
                    n_batches += 1

            train_loss = loss_sum / max(1, n_batches)
            train_acc = acc_sum / max(1, n_batches)

            #validation
            val_acc = None
            if val_loader is not None:
                val_acc = self._eval_metric(val_loader)
                if val_acc > best_val:
                    best_val = val_acc
                    no_imporve = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    no_imporve += 1
                    if no_imporve >= self.cfg.patience:
                        print(f"Early stopping at epoch {epoch:03d} (best val_acc={best_val:.2f}")
                        break

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_acc is None:
                print(f"epoch {epoch:03d} | loss={train_loss:.4f} | acc={train_acc:.2f}")
            else:
                print(f"epoch {epoch:03d} | loss={train_loss:.4f} | train_acc={train_acc:.2f} | val_acc={val_acc:.2f}")


        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        return history, best_state, best_val

    @torch.no_grad()
    def _eval_metric(self, loader):
        self.model.eval()
        acc_sum = 0.0
        n_bacthes = 0
        for xb, yb in loader:
            xb = xb.to(self.cfg.device)
            yb = yb.to(self.cfg.device)

            S = self.model(xb)
            acc_sum += float(self.metric_fn(S, yb))
            n_bacthes += 1

        return  acc_sum / max(1, n_bacthes)
        
    
    def evaluate(self, loader, state_dict = None):
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        acc_sum = 0.0
        n_batches = 0
        Sall , yall= [], []
        for xb, yb in loader:
            xb = xb.to(self.cfg.device)
            yb = yb.to(self.cfg.device)

            S = self.model(xb)
            acc_sum += float(self.metric_fn(S, yb))
            n_batches += 1
            Sall.append(S.detach().cpu())
            yall.append(yb.detach().cpu())
            

        return {
            "acc": acc_sum / max(1, n_batches),
            "scores": torch.cat(Sall),
            "labels": torch.cat(yall),
        }
    



import os, json, torch

def save_run(out_dir, *, history, best_state, best_val_acc, test_out, extra_metrics=None, run_cfg=None):
    """
    Writes:
      - best_state.pt
      - history.json
      - metrics.json
      - test_scores_labels.pt
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) best model weights
    torch.save(best_state, os.path.join(out_dir, "best_state.pt"))

    # 2) history
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 3) metrics
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

    # 4) raw test tensors
    torch.save(
        {"scores": test_out["scores"], "labels": test_out["labels"], "preds": test_out["preds"]},
        os.path.join(out_dir, "test_scores_labels.pt")
    )






__all__ = ["TrainConfig", "Trainer", "save_run"]