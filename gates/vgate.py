import torch
from .gate import Gate, CNOT, I
from TorchQML.core.sym import Sym
from TorchQML.core.config import DTYPE, DEVICE


class VGate:
    def __init__(self, rot, angle):
        self.rot = rot
        self.angle = angle

    def resolve(self, x, t):
        ang = self.angle.eval(x, t)
        if self.rot == "rx":
            return _rxn(ang)
        elif self.rot == "ry":
            return _ryn(ang)
        elif self.rot == "rz":
            return _rzn(ang)
        raise TypeError("not rx, ry, rz")
        

def _to_real_angle(theta: torch.Tensor) -> torch.Tensor:
    """
    Ensure the rotation angle is a real float32 tensor on DEVICE.
    If a complex tensor sneaks in, we take its real part explicitly
    (no PyTorch warning).
    """
    t = torch.as_tensor(theta, device=DEVICE)
    if torch.is_complex(t):
        t = t.real
    return t.to(torch.float32)


# --- Parameterized 1-qubit rotations ---
def _rxn(theta) -> Gate:
    # Rx(θ) = [[cos θ/2, -i sin θ/2], [-i sin θ/2, cos θ/2]]
    t = _to_real_angle(theta)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    mat = torch.stack([
        torch.stack([c, -1j * s]),
        torch.stack([-1j * s, c])
    ]).to(dtype=DTYPE, device=DEVICE)
    return Gate(mat, "RotX", [t])


def _ryn(theta) -> Gate:
    # Ry(θ) = [[cos θ/2, -sin θ/2], [sin θ/2, cos θ/2]]
    t = _to_real_angle(theta)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    mat = torch.stack([
        torch.stack([c, -s]),
        torch.stack([s,  c])
    ]).to(dtype=DTYPE, device=DEVICE)
    return Gate(mat, "RotY", [t])


def _rzn(theta) -> Gate:
    # Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
    t = _to_real_angle(theta)
    e_m = torch.exp(-0.5j * t)
    e_p = torch.exp( 0.5j * t)
    zero = torch.zeros((), device=DEVICE, dtype=DTYPE)
    mat = torch.stack([
        torch.stack([e_m.to(DTYPE), zero]),
        torch.stack([zero, e_p.to(DTYPE)])
    ])
    return Gate(mat, "RotZ", [t])


def rx(w):
    if isinstance(w, Sym):
        return VGate("rx", w)
    return _rxn(w)


def ry(w):
    if isinstance(w, Sym):
        return VGate("ry", w)
    return _ryn(w)


def rz(w):
    if isinstance(w, Sym):
        return VGate("rz", w)
    return _rzn(w)



def _CRYn(theta, nq, ctr, trg):
    t = _to_real_angle(theta)

    dim = 2 ** nq
    U = torch.zeros((dim, dim), dtype=DTYPE, device=DEVICE)

    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    Ry = torch.stack([
        torch.stack([ c, -s]),
        torch.stack([ s,  c])
    ])

    for basis in range(dim):
        b = [(basis >> k) & 1 for k in range(nq)]

        if b[ctr] == 0:
            U[basis, basis] = 1.0
        else:
            bt = b[trg]
            for new_bt in (0, 1):
                amp = Ry[new_bt, bt]
                b2 = b.copy()
                b2[trg] = new_bt

                out = 0
                for k in range(nq):
                    out |= (b2[k] << k)

                U[out, basis] = amp

    return Gate(U, f"CRY(c={ctr},t={trg})", [t])


def CRY(circ, theta, nq, ctr, trg):
    # RY(theta/2) on target
    layer1 = [I]*nq
    layer1[trg] = ry(theta/2)
    circ.add_gates(layer1)

    circ.add_full(CNOT(nq, ctr, trg))

    # RY(-theta/2) on target
    layer2 = [I]*nq
    layer2[trg] = ry(-theta/2)
    circ.add_gates(layer2)

    circ.add_full(CNOT(nq, ctr, trg))



__all__ = ["rx", "ry", "rz", "CRY"]
