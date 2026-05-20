import numpy as np
import torch

from TorchQML.kernels.quantum import center_train_test, fidelity_kernel_matrix, fit_precomputed_svc


def _as_tensor_kernel(K):
    return K if torch.is_tensor(K) else torch.as_tensor(K)


def fit_qsvm(
    Xtr,
    ytr,
    Xte,
    circ,
    theta,
    *,
    amp_enc=None,
    C: float = 1.0,
    center: bool = True,
    blockA: int = 256,
    blockB: int = 256,
    **svc_kwargs,
):
    """Fit an sklearn SVC using a TorchQML fidelity kernel."""
    Ktr = fidelity_kernel_matrix(
        Xtr,
        Xtr,
        theta,
        circ,
        amp_enc=amp_enc,
        blockA=blockA,
        blockB=blockB,
        symmetric=True,
    )
    Kte = fidelity_kernel_matrix(
        Xte,
        Xtr,
        theta,
        circ,
        amp_enc=amp_enc,
        blockA=blockA,
        blockB=blockB,
    )

    if center:
        Ktr_t, Kte_t = center_train_test(_as_tensor_kernel(Ktr), _as_tensor_kernel(Kte))
        Ktr = Ktr_t.detach().cpu().numpy()
        Kte = Kte_t.detach().cpu().numpy()

    clf, pred = fit_precomputed_svc(Ktr, ytr, Kte, C=C, **svc_kwargs)
    return {
        "classifier": clf,
        "pred": pred,
        "Ktr": np.asarray(Ktr),
        "Kte": np.asarray(Kte),
    }


__all__ = ["fit_qsvm"]
