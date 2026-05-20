import math

import torch

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.gates import I, S, Sdg, T, Tdg


def test_phase_gate_matrices():
    expected_s = torch.tensor([[1, 0], [0, 1j]], dtype=DTYPE, device=DEVICE)
    expected_t = torch.tensor(
        [[1, 0], [0, complex(math.cos(math.pi / 4), math.sin(math.pi / 4))]],
        dtype=DTYPE,
        device=DEVICE,
    )

    assert torch.allclose(S.matrix, expected_s)
    assert torch.allclose(T.matrix, expected_t)


def test_phase_gate_inverses():
    assert torch.allclose(Sdg.matrix, S.matrix.mH)
    assert torch.allclose(Tdg.matrix, T.matrix.mH)
    assert torch.allclose(S.matrix @ Sdg.matrix, I.matrix)
    assert torch.allclose(T.matrix @ Tdg.matrix, I.matrix)
