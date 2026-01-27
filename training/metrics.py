import torch
def binary_metrics_from_preds(y_true, y_pred):

    """
    y_true, y_pred: 1D torch tensors (0/1), on CPU or GPU
    returns dict with TP/TN/FP/FN + precision/recall/f1
    """
    y_true = y_true.long()
    y_pred = y_pred.long()

    TP = int(((y_true == 1) & (y_pred == 1)).sum().item())
    TN = int(((y_true == 0) & (y_pred == 0)).sum().item())
    FP = int(((y_true == 0) & (y_pred == 1)).sum().item())
    FN = int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def batch_accuracy(S: torch.Tensor, yb: torch.Tensor) -> float:
    """
    S:  [B]  (sum of Z per sample)
    yb: [B]  (0/1 labels)
    """
    preds = (S <= 0).long()         
    correct = (preds == yb).sum().item()
    total = yb.shape[0]
    return 100.0 * correct / total

__all__ = ["batch_accuracy", "binary_metrics_from_preds"]