import numpy as np
import torch

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.core.runtime import compute_states, fidelity_kernel_from_states, same_tensor


def kernel_matrix(X, circ, theta, amp_enc=None) -> torch.Tensor:
    """Differentiable fidelity kernel for one batch."""
    psi = compute_states(X, theta, circ, amp_enc=amp_enc)
    return fidelity_kernel_from_states(psi)


def rbf_kernel_matrix(X, circ, theta, gamma: float = 1.0, amp_enc=None) -> torch.Tensor:
    """RBF transform of the quantum fidelity kernel."""
    fidelity = kernel_matrix(X, circ, theta, amp_enc=amp_enc)
    return torch.exp(-gamma * (1.0 - fidelity))


@torch.no_grad()
def fidelity_kernel_matrix(
    XA,
    XB,
    theta,
    circ,
    *,
    amp_enc=None,
    blockA: int = 256,
    blockB: int = 256,
    symmetric: bool = False,
) -> np.ndarray:
    """Blocked NumPy fidelity kernel for larger train/test splits."""
    XA = XA.to(device=DEVICE)
    XB = XB.to(device=DEVICE)
    theta = theta.to(device=DEVICE, dtype=DTYPE)

    na = XA.shape[0]
    nb = XB.shape[0]
    out = np.empty((na, nb), dtype=np.float32)
    shared_cache = {} if same_tensor(XA, XB) else None

    def get_states(X, i0, i1):
        if shared_cache is not None and i0 in shared_cache:
            return shared_cache[i0]
        states = compute_states(X[i0:i1], theta, circ, amp_enc=amp_enc)
        if shared_cache is not None:
            shared_cache[i0] = states
        return states

    for i0 in range(0, na, blockA):
        i1 = min(i0 + blockA, na)
        psia = get_states(XA, i0, i1)
        j_start = i0 if symmetric else 0
        for j0 in range(j_start, nb, blockB):
            j1 = min(j0 + blockB, nb)
            psib = get_states(XB, j0, j1)
            block = fidelity_kernel_from_states(psia, psib).detach().cpu().numpy()
            out[i0:i1, j0:j1] = block
            if symmetric:
                out[j0:j1, i0:i1] = block.T

    return out


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-center a square kernel matrix."""
    return K - K.mean(dim=1, keepdim=True) - K.mean(dim=0, keepdim=True) + K.mean()


def center_train_test(Ktr: torch.Tensor, Kte: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center train and test precomputed kernels using train statistics."""
    train_row = Ktr.mean(dim=1, keepdim=True)
    train_col = Ktr.mean(dim=0, keepdim=True)
    train_mean = Ktr.mean()
    test_row = Kte.mean(dim=1, keepdim=True)
    return Ktr - train_row - train_col + train_mean, Kte - test_row - train_col + train_mean


def fit_precomputed_svc(Ktr, ytr, Kte=None, *, C: float = 1.0, **svc_kwargs):
    """Fit an sklearn SVC on a precomputed kernel and optionally predict test labels."""
    from sklearn.svm import SVC

    clf = SVC(kernel="precomputed", C=C, **svc_kwargs)
    y_np = ytr.detach().cpu().numpy() if torch.is_tensor(ytr) else np.asarray(ytr)
    clf.fit(np.asarray(Ktr), y_np)
    if Kte is None:
        return clf
    return clf, clf.predict(np.asarray(Kte))
