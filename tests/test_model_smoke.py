import torch

from TorchQML import Circuit, CircuitSpec, H, Tsym, Xsym, rx
from TorchQML.encoding import AmpEnc
from TorchQML.models import (
    PauliHeadModel,
    PauliVectorModel,
    SpookyModel,
    SumZModel,
    VectorZDirHam,
    VectorZLinear,
)


def _tiny_circuit(tlen=6):
    spec = CircuitSpec(num_qubits=2, xlen=2, tlen=tlen)
    x = Xsym()
    t = Tsym()
    circ = Circuit(num_qubits=2, specs=spec)
    circ.add_gates([H, H])
    circ.add_gates([rx(x[0]), rx(x[1])])
    circ.add_gates([rx(t[0]), rx(t[1])])
    return circ, spec


def _assert_score(name, out, batch_size):
    assert out.shape == (batch_size,), name
    assert torch.isfinite(out.real).all(), name


def _assert_backward(name, model, score):
    model.zero_grad(set_to_none=True)
    score.real.mean().backward()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.requires_grad
    ), name


def test_z_readout_model_heads_smoke():
    torch.manual_seed(0)
    circ, spec = _tiny_circuit()
    xb = torch.tensor(
        [[0.0, 0.1], [0.6, -0.2], [-0.3, 0.4]],
        dtype=torch.float32,
    )

    models = {
        "SumZModel": SumZModel(circ, spec, amp_enc=False),
        "VectorZLinear": VectorZLinear(circ, spec, amp_enc=False),
        "VectorZDirHam": VectorZDirHam(circ, spec, n_slots=1, amp_enc=None),
    }

    for name, model in models.items():
        score = model(xb)
        _assert_score(name, score, xb.shape[0])
        _assert_backward(name, model, score)


def test_pauli_and_entanglement_model_heads_smoke():
    torch.manual_seed(0)
    circ, spec = _tiny_circuit()
    xb = torch.tensor(
        [[0.0, 0.1], [0.6, -0.2], [-0.3, 0.4]],
        dtype=torch.float32,
    )

    for name, model in {
        "PauliHeadModel": PauliHeadModel(circ, spec, amp_enc=False),
        "PauliVectorModel": PauliVectorModel(circ, spec, amp_enc=False),
    }.items():
        score = model(xb)
        _assert_score(name, score, xb.shape[0])
        _assert_backward(name, model, score)

    spooky = SpookyModel(circ, spec, amp_enc=False)
    score, entanglement = spooky(xb)
    _assert_score("SpookyModel.score", score, xb.shape[0])
    assert entanglement.shape == (xb.shape[0], spec.num_qubits)
    assert torch.isfinite(entanglement).all()
    _assert_backward("SpookyModel", spooky, score)


def test_model_heads_with_amplitude_encoding_smoke():
    torch.manual_seed(0)
    spec = CircuitSpec(num_qubits=2, xlen=4, tlen=6)
    t = Tsym()
    circ = Circuit(num_qubits=2, specs=spec)
    circ.add_gates([rx(t[0]), rx(t[1])])
    xb = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    amp_enc = AmpEnc(spec.num_qubits)

    dir_model = VectorZDirHam(circ, spec, n_slots=1, amp_enc=amp_enc)
    dir_score = dir_model(xb)
    _assert_score("VectorZDirHam.amp", dir_score, xb.shape[0])
    _assert_backward("VectorZDirHam.amp", dir_model, dir_score)

    for name, model in {
        "SumZModel.amp": SumZModel(circ, spec, amp_enc=True),
        "VectorZLinear.amp": VectorZLinear(circ, spec, amp_enc=True),
        "PauliHeadModel.amp": PauliHeadModel(circ, spec, amp_enc=True),
        "PauliVectorModel.amp": PauliVectorModel(circ, spec, amp_enc=True),
    }.items():
        score = model(xb)
        _assert_score(name, score, xb.shape[0])
        _assert_backward(name, model, score)

    spooky = SpookyModel(circ, spec, amp_enc=True)
    score, entanglement = spooky(xb)
    _assert_score("SpookyModel.amp.score", score, xb.shape[0])
    assert entanglement.shape == (xb.shape[0], spec.num_qubits)
    _assert_backward("SpookyModel.amp", spooky, score)
