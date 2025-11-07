import torch
from qml_lib.circuits.circuit import Circuit
from qml_lib.measurements.expectation_Z import expectation_Z


def evaluate_Z(circ, theta, X, y, n_qubits)->float:
    correct = 0

    for i, x in enumerate(X):
        psi = circ.apply_to(x = x, theta = theta)
        expZ = expectation_Z(psi, 0, n_qubits)
        prob = (expZ+1)/2
        pred = (prob > 0.5).long()
        if pred.item() == y[i].item():
            correct+= 1

    return correct / X.shape[0]