import numpy as np
import torch

from TorchQML.synthesis.unitary_dense import CDTYPE, kron_all


def normalize(states: torch.Tensor) -> torch.Tensor:
    return states / torch.linalg.vector_norm(states, dim=1, keepdim=True).clamp_min(1e-12)


def basis_states(num: int, n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    dim = 1 << n
    idx = rng.choice(dim, size=min(num, dim), replace=False)
    states = torch.zeros((len(idx), dim), dtype=CDTYPE)
    states[torch.arange(len(idx)), torch.as_tensor(idx)] = 1.0
    return states


def haar_states(num: int, n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    dim = 1 << n
    real = torch.randn((num, dim), generator=generator)
    imag = torch.randn((num, dim), generator=generator)
    return normalize((real + 1j * imag).to(CDTYPE))


def product_states(num: int, n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    states = []
    for _ in range(num):
        factors = []
        for _ in range(n):
            z = torch.randn(2, generator=generator) + 1j * torch.randn(2, generator=generator)
            factors.append((z / torch.linalg.vector_norm(z)).to(CDTYPE))
        states.append(kron_all([factor.reshape(2, 1) for factor in factors]).reshape(-1))
    return torch.stack(states)


def make_states(mode: str, num: int, n: int, seed: int = 0) -> torch.Tensor:
    if mode == "basis":
        return basis_states(num, n, seed)
    if mode == "haar":
        return haar_states(num, n, seed)
    if mode == "product":
        return product_states(num, n, seed)
    if mode == "basis_random":
        n_basis = max(1, num // 2)
        return torch.cat(
            [
                basis_states(n_basis, n, seed),
                haar_states(num - n_basis, n, seed + 1),
            ]
        )
    raise ValueError(f"unknown state mode: {mode}")


__all__ = ["basis_states", "haar_states", "make_states", "normalize", "product_states"]
