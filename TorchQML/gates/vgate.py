import torch
from .gate import Gate, CNOT, I, X, Y, Z, H
from TorchQML.core.sym import Sym
from TorchQML.core.config import DTYPE, DEVICE


I2 = torch.eye(2, dtype=DTYPE, device=DEVICE)
I4 = torch.eye(4, dtype=DTYPE, device=DEVICE)
_EMBED_2Q_CACHE: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


class VGate:
    """Gate template whose angle is resolved from data or trainable parameters."""

    def __init__(self, rot, angle):
        self.rot = rot
        self.angle = angle

    def resolve(self, x, t):
        """Evaluate the stored symbolic angle and return a concrete gate."""
        if isinstance(self.angle, (list, tuple)):
            ang = [a.eval(x, t) if isinstance(a, Sym) else a for a in self.angle]
        else:
            ang = self.angle.eval(x, t) if isinstance(self.angle, Sym) else self.angle

        if self.rot == "rx":
            return _rxn(ang)
        elif self.rot == "ry":
            return _ryn(ang)
        elif self.rot == "rz":
            return _rzn(ang)
        elif self.rot == "ham1q":
            return _ham1q(*ang)
        elif self.rot == "ham2q":
            return _ham2q(*ang)
        elif self.rot == "ham2qfull":
            return _ham2q_full(*ang)
        raise TypeError("not rx, ry, rz, ham1q, ham2q")

def _to_real_angle(theta: torch.Tensor) -> torch.Tensor:
    """Move an angle to DEVICE as a real float tensor."""
    t = torch.as_tensor(theta, device=DEVICE)
    if torch.is_complex(t):
        t = t.real
    return t.to(torch.float32)


def _as_matrix_coeff(t: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or batch of scalars over matrix dimensions."""
    return t.to(dtype=DTYPE, device=DEVICE).unsqueeze(-1).unsqueeze(-1)


def _pauli_exp(theta, pauli: torch.Tensor, ident: torch.Tensor) -> torch.Tensor:
    """Compute exp(-i theta P / 2) for a Pauli-like matrix P."""
    t = _to_real_angle(theta)
    cos_term = _as_matrix_coeff(torch.cos(t / 2))
    sin_term = _as_matrix_coeff(-1j * torch.sin(t / 2))
    return cos_term * ident + sin_term * pauli


# --- Parameterized 1-qubit rotations ---
def _rxn(theta) -> Gate:
    """Return a concrete X-rotation gate."""
    t = _to_real_angle(theta)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    mat = torch.stack([
        torch.stack([c, -1j * s], dim=-1),
        torch.stack([-1j * s, c], dim=-1)
    ], dim=-2).to(dtype=DTYPE, device=DEVICE)
    return Gate(mat, "RotX", [t])


def _ryn(theta) -> Gate:
    """Return a concrete Y-rotation gate."""
    t = _to_real_angle(theta)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    mat = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1)
    ], dim=-2).to(dtype=DTYPE, device=DEVICE)
    return Gate(mat, "RotY", [t])


def _rzn(theta) -> Gate:
    """Return a concrete Z-rotation gate."""
    t = _to_real_angle(theta)
    e_m = torch.exp(-0.5j * t)
    e_p = torch.exp( 0.5j * t)
    zero = torch.zeros_like(e_m, device=DEVICE, dtype=DTYPE)
    mat = torch.stack([
        torch.stack([e_m.to(DTYPE), zero], dim=-1),
        torch.stack([zero, e_p.to(DTYPE)], dim=-1)
    ], dim=-2)
    return Gate(mat, "RotZ", [t])


def rx(w):
    """Create an X rotation, symbolic when `w` is a `Sym`."""
    if isinstance(w, Sym):
        return VGate("rx", w)
    return _rxn(w)


def ry(w):
    """Create a Y rotation, symbolic when `w` is a `Sym`."""
    if isinstance(w, Sym):
        return VGate("ry", w)
    return _ryn(w)


def rz(w):
    """Create a Z rotation, symbolic when `w` is a `Sym`."""
    if isinstance(w, Sym):
        return VGate("rz", w)
    return _rzn(w)



def _CRYn(theta, nq, ctr, trg):
    """Build a dense controlled-RY matrix."""
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
    """Append a controlled-RY decomposition to `circ`."""
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

def CRX(circ, theta, nq, ctr, trg):
    """Append a controlled-RX decomposition to `circ`."""
    # RX(theta/2) on target
    layer1 = [I] * nq
    layer1[trg] = rx(theta / 2)
    circ.add_gates(layer1)

    circ.add_full(CNOT(nq, ctr, trg))

    # RX(-theta/2) on target
    layer2 = [I] * nq
    layer2[trg] = rx(-theta / 2)
    circ.add_gates(layer2)

    circ.add_full(CNOT(nq, ctr, trg))



def CRZ(circ, theta, nq, ctr, trg):
    """Append a controlled-RZ decomposition to `circ`."""
    # RZ(theta/2) on target
    layer1 = [I] * nq
    layer1[trg] = rz(theta / 2)
    circ.add_gates(layer1)

    circ.add_full(CNOT(nq, ctr, trg))

    # RZ(-theta/2) on target
    layer2 = [I] * nq
    layer2[trg] = rz(-theta / 2)
    circ.add_gates(layer2)

    circ.add_full(CNOT(nq, ctr, trg))


def _ham1q(cx,cy,cz):
    """Return exp(-i (cx X + cy Y + cz Z) / 2)."""
    cx = _to_real_angle(cx)
    cy = _to_real_angle(cy)
    cz = _to_real_angle(cz)

    H1 = (
        _as_matrix_coeff(cx) * X.matrix
        + _as_matrix_coeff(cy) * Y.matrix
        + _as_matrix_coeff(cz) * Z.matrix
    )
    norm = torch.sqrt(cx * cx + cy * cy + cz * cz)
    cos_term = _as_matrix_coeff(torch.cos(norm / 2))
    scale = torch.where(
        norm > 1e-12,
        torch.sin(norm / 2) / norm,
        torch.full_like(norm, 0.5),
    )
    U = cos_term * I2 + (-1j * _as_matrix_coeff(scale)) * H1
    return Gate(U, "Ham1Q", [cx,cy,cz])



XX = torch.kron(X.matrix,X.matrix)
YY = torch.kron(Y.matrix,Y.matrix)
ZZ = torch.kron(Z.matrix, Z.matrix)

def _ham2q(cxx, cyy,czz):
    """Return a two-qubit XX/YY/ZZ Hamiltonian gate."""
    cxx = _to_real_angle(cxx)
    cyy = _to_real_angle(cyy)
    czz = _to_real_angle(czz)
    U = (
        _pauli_exp(cxx, XX, I4)
        @ _pauli_exp(cyy, YY, I4)
        @ _pauli_exp(czz, ZZ, I4)
    )
    return Gate(U, "Ham2Q", [cxx, cyy, czz])

def ham1q(cx, cy = None, cz = None):
    """Create a one-qubit Hamiltonian gate, symbolic when needed."""
    if isinstance(cx, (list, tuple)):
        if len(cx) != 3:
            raise ValueError("cx needs 3 its a list")
        if any(isinstance(a, Sym) for a in cx):
            return VGate("ham1q", list(cx))
        return _ham1q(*cx)

    if cy is None or cz is None:
        raise ValueError("no cy or cz")

    if any(isinstance(a, Sym) for a in (cx, cy, cz)):
        return VGate("ham1q", [cx, cy, cz])

    return _ham1q(cx, cy, cz)


def ham2q(cx, cy = None, cz = None):
    """Create a two-qubit XX/YY/ZZ Hamiltonian gate."""
    if isinstance(cx, (list, tuple)):
        if len(cx) != 3:
            raise ValueError("cx needs 3 its a list")
        if any(isinstance(a, Sym) for a in cx):
            return VGate("ham2q", list(cx))
        return _ham2q(*cx)

    if cy is None or cz is None:
        raise ValueError("no cy or cz")

    if any(isinstance(a, Sym) for a in (cx, cy, cz)):
        return VGate("ham2q", [cx, cy, cz])

    return _ham2q(cx, cy, cz)


def _embed_2q(U2, nq, q1, q2):
    """Embed a 4x4 two-qubit operator into an `nq`-qubit space."""
    if q1 == q2:
        raise ValueError("q1 and q2 must be different")

    rows, cols, local_rows, local_cols = _embed_2q_indices(nq, q1, q2)
    dim = 1 << nq

    if U2.dim() == 2:
        U = torch.zeros((dim, dim), dtype=DTYPE, device=DEVICE)
        U[rows, cols] = U2[local_rows, local_cols]
        return U

    lead_shape = U2.shape[:-2]
    U = torch.zeros((*lead_shape, dim, dim), dtype=DTYPE, device=DEVICE)
    U_flat = U.reshape(-1, dim, dim)
    U2_flat = U2.reshape(-1, 4, 4)
    U_flat[:, rows, cols] = U2_flat[:, local_rows, local_cols]
    return U


def _embed_2q_indices(nq, q1, q2):
    """Return cached index tensors used by `_embed_2q`."""
    key = (nq, q1, q2)
    if key in _EMBED_2Q_CACHE:
        return _EMBED_2Q_CACHE[key]

    rows, cols, local_rows, local_cols = [], [], [], []
    dim = 1 << nq

    for basis in range(dim):
        bits = [(basis >> k) & 1 for k in range(nq)]
        inp = bits[q1] + 2 * bits[q2]

        for new_q1 in (0, 1):
            for new_q2 in (0, 1):
                out2 = new_q1 + 2 * new_q2
                bits2 = bits.copy()
                bits2[q1] = new_q1
                bits2[q2] = new_q2

                out = 0
                for k in range(nq):
                    out |= (bits2[k] << k)

                rows.append(out)
                cols.append(basis)
                local_rows.append(out2)
                local_cols.append(inp)

    cached = (
        torch.tensor(rows, device=DEVICE, dtype=torch.long),
        torch.tensor(cols, device=DEVICE, dtype=torch.long),
        torch.tensor(local_rows, device=DEVICE, dtype=torch.long),
        torch.tensor(local_cols, device=DEVICE, dtype=torch.long),
    )
    _EMBED_2Q_CACHE[key] = cached
    return cached


def _ham2q_full(nq, q1, q2, cxx, cyy, czz):
    """Return an embedded two-qubit Hamiltonian gate."""
    cxx = _to_real_angle(cxx)
    cyy = _to_real_angle(cyy)
    czz = _to_real_angle(czz)
    U2 = (
        _pauli_exp(cxx, XX, I4)
        @ _pauli_exp(cyy, YY, I4)
        @ _pauli_exp(czz, ZZ, I4)
    )
    Ufull = _embed_2q(U2, nq, q1, q2)
    return Gate(Ufull, f"Ham2Q({q1},{q2})", [cxx, cyy, czz])


def ham2q_full(nq, q1, q2, cxx, cyy=None, czz=None):
    """Create a two-qubit Hamiltonian acting on qubits `q1` and `q2`."""
    if cyy is None or czz is None:
        raise ValueError("need cxx, cyy, czz")

    if any(isinstance(a, Sym) for a in (cxx, cyy, czz)):
        return VGate("ham2qfull", [nq, q1, q2, cxx, cyy, czz])

    return _ham2q_full(nq, q1, q2, cxx, cyy, czz)

__all__ = ["rx", "ry", "rz", "CRY", "CRX", "CRZ", "ham1q", "ham2q", "ham2q_full"]
