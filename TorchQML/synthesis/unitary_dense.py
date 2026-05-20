import math

import torch


CDTYPE = torch.complex64


def kron_all(mats: list[torch.Tensor]) -> torch.Tensor:
    out = mats[0]
    for mat in mats[1:]:
        out = torch.kron(out, mat)
    return out


def eye2() -> torch.Tensor:
    return torch.eye(2, dtype=CDTYPE)


def h_gate() -> torch.Tensor:
    return torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=CDTYPE) / math.sqrt(2.0)


def rx_gate(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    return torch.stack(
        [
            torch.stack([c, -1j * s]),
            torch.stack([-1j * s, c]),
        ]
    ).to(CDTYPE)


def ry_gate(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(CDTYPE)


def rz_gate(theta: torch.Tensor) -> torch.Tensor:
    phases = torch.stack([torch.exp(-0.5j * theta), torch.exp(0.5j * theta)])
    return torch.diag(phases).to(CDTYPE)


def full_single(gate: torch.Tensor, qubit: int, n: int) -> torch.Tensor:
    return kron_all([gate if idx == qubit else eye2() for idx in range(n)])


def cnot_full(control: int, target: int, n: int) -> torch.Tensor:
    dim = 1 << n
    mat = torch.zeros((dim, dim), dtype=CDTYPE)
    for basis in range(dim):
        bits = [(basis >> (n - 1 - idx)) & 1 for idx in range(n)]
        if bits[control]:
            bits[target] ^= 1
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        mat[out, basis] = 1.0
    return mat


def cz_full(a: int, b: int, n: int) -> torch.Tensor:
    dim = 1 << n
    diag = torch.ones(dim, dtype=CDTYPE)
    for basis in range(dim):
        if ((basis >> (n - 1 - a)) & 1) and ((basis >> (n - 1 - b)) & 1):
            diag[basis] = -1.0
    return torch.diag(diag)


def swap_full(a: int, b: int, n: int) -> torch.Tensor:
    dim = 1 << n
    mat = torch.zeros((dim, dim), dtype=CDTYPE)
    for basis in range(dim):
        bits = [(basis >> (n - 1 - idx)) & 1 for idx in range(n)]
        bits[a], bits[b] = bits[b], bits[a]
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        mat[out, basis] = 1.0
    return mat


def cp_full(a: int, b: int, angle: float | torch.Tensor, n: int) -> torch.Tensor:
    angle = torch.as_tensor(angle)
    dim = 1 << n
    diag = torch.ones(dim, dtype=CDTYPE)
    phase = torch.exp(1j * angle).to(CDTYPE)
    for basis in range(dim):
        if ((basis >> (n - 1 - a)) & 1) and ((basis >> (n - 1 - b)) & 1):
            diag[basis] = phase
    return torch.diag(diag)


def qft_full(n: int) -> torch.Tensor:
    dim = 1 << n
    omega = torch.exp(2j * torch.tensor(math.pi, dtype=torch.float32) / dim)
    rows = torch.arange(dim).reshape(-1, 1)
    cols = torch.arange(dim).reshape(1, -1)
    return (omega ** (rows * cols)).to(CDTYPE) / math.sqrt(dim)


__all__ = [
    "CDTYPE",
    "cnot_full",
    "cp_full",
    "cz_full",
    "eye2",
    "full_single",
    "h_gate",
    "kron_all",
    "qft_full",
    "rx_gate",
    "ry_gate",
    "rz_gate",
    "swap_full",
]
