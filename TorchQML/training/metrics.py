import torch


def binary_metrics_from_preds(y_true, y_pred):
    """Return confusion counts plus precision, recall, and F1 for binary labels."""
    y_true = y_true.long()
    y_pred = y_pred.long()

    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def batch_accuracy(S: torch.Tensor, yb: torch.Tensor) -> float:
    """Accuracy for score tensors where non-positive means class 1."""
    preds = (S <= 0).long()
    return 100.0 * (preds == yb).sum().item() / yb.shape[0]


@torch.no_grad()
def contrastive_pair_accuracy(l_vec: torch.Tensor, y: torch.Tensor) -> float:
    """Check whether same-label pairs are closer than different-label pairs."""
    li = l_vec.unsqueeze(1)
    lj = l_vec.unsqueeze(0)
    dist = torch.norm(li - lj, dim=-1)

    same = y.unsqueeze(1) == y.unsqueeze(0)
    diff = ~same
    return float(dist[same].mean() < dist[diff].mean()) * 100.0


def spooky_accuracy(output, yb: torch.Tensor) -> float:
    """Accuracy for `SpookyModel` outputs that include a learned threshold."""
    score, _, tau = output
    preds = (score > tau).long()
    return 100.0 * (preds == yb).sum().item() / yb.size(0)


__all__ = ["batch_accuracy", "binary_metrics_from_preds", "spooky_accuracy"]
