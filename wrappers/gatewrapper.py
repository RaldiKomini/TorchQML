import torch
from qml_lib.gates.gate import Gate
from qml_lib.wrappers.symcalc import Sym
from qml_lib.config import DTYPE, DEVICE


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

    

# --- Parameterized 1-qubit rotations ---
def _rxn(theta) -> Gate:
    # Rx(θ) = [[cos θ/2, -i sin θ/2], [-i sin θ/2, cos θ/2]]
    t = torch.as_tensor(theta, device=DEVICE, dtype=torch.float32)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    mat = torch.stack([
        torch.stack([c, -1j * s]),
        torch.stack([-1j * s, c])
    ]).to(dtype=DTYPE, device=DEVICE)
    return Gate(mat, f"RotX(t={float(t) if t.numel()==1 else '...'})", [t])

def _ryn(theta) -> Gate:
    # Ry(θ) = [[cos θ/2, -sin θ/2], [sin θ/2, cos θ/2]]
    t = torch.as_tensor(theta, device=DEVICE, dtype=torch.float32)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    mat = torch.stack([
        torch.stack([c, -s]),
        torch.stack([s,  c])
    ]).to(dtype=DTYPE, device=DEVICE)
    return Gate(mat, f"RotY(t={float(t) if t.numel()==1 else '...'})", [t])

def _rzn(theta) -> Gate:
    # Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
    t = torch.as_tensor(theta, device=DEVICE, dtype=torch.float32)
    e_m = torch.exp(-0.5j * t)
    e_p = torch.exp( 0.5j * t)
    zero = torch.zeros((), device=DEVICE, dtype=DTYPE)
    mat = torch.stack([
        torch.stack([e_m.to(DTYPE), zero]),
        torch.stack([zero, e_p.to(DTYPE)])
    ])
    return Gate(mat, f"RotZ(t={float(t) if t.numel()==1 else '...'})", [t])

def rx(w):
    if isinstance(w, Sym):
        return VGate("rx", w)
    return _rxn(torch.as_tensor(w))

def ry(w):
    if isinstance(w, Sym):
        return VGate("ry", w)
    return _ryn(torch.as_tensor(w))

def rz(w):
    if isinstance(w, Sym):
        return VGate("rz", w)
    return _rzn(torch.as_tensor(w))