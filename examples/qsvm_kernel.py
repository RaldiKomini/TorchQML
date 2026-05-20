import torch

from TorchQML import Circuit, CircuitSpec
from TorchQML.encoding import AmpEnc
from TorchQML.kernels import kernel_matrix
from TorchQML.quantum_svm import fit_qsvm


def main():
    spec = CircuitSpec(num_qubits=2, xlen=4, tlen=0)
    circ = Circuit(num_qubits=2, specs=spec)
    theta = torch.zeros(0, dtype=torch.complex64)
    amp_enc = AmpEnc(spec.num_qubits)

    X_train = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    y_train = torch.tensor([0, 0, 1, 1])

    K = kernel_matrix(X_train, circ, theta, amp_enc=amp_enc)
    result = fit_qsvm(X_train, y_train, X_train, circ, theta, amp_enc=amp_enc)
    print({"kernel_shape": tuple(K.shape), "pred": result["pred"].tolist()})


if __name__ == "__main__":
    main()
