import torch
from qml_lib.config import DEVICE, DTYPE
from functools import reduce
from qml_lib.gates import I, Z

def kron_all(matrices : list[torch.Tensor])-> torch.Tensor:
    return reduce(torch.kron, matrices)


def target_Z(n_qubits, target = 0) -> torch.Tensor:
    obs = []
    for n in range(n_qubits):
        obs.append(I.matrix if target != n else Z.matrix)
    return kron_all(obs)


def expectation_Z(psi: torch.tensor, n_qubits: int, target = 0)->torch.Tensor:

    dim = 2 << n_qubits
    obs_Z = target_Z(n_qubits=n_qubits, target= target)

    m1 = obs_Z @ psi
    m2 = torch.conj(psi) @ m1
    return m2.real
