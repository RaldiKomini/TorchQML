"""Compatibility wrappers for older TorchQML quantum_svm imports."""

from TorchQML.kernels.quantum import (
    center_kernel,
    center_train_test,
    fidelity_kernel_matrix,
    kernel_matrix,
    rbf_kernel_matrix,
)

kernel = fidelity_kernel_matrix

__all__ = [
    "center_kernel",
    "center_train_test",
    "fidelity_kernel_matrix",
    "kernel",
    "kernel_matrix",
    "rbf_kernel_matrix",
]
