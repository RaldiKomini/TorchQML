from dataclasses import dataclass
from functools import reduce
import torch

from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.gates import Gate, VGate


def _kron_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Kronecker product that keeps any leading batch dimensions."""
    prod = torch.einsum("...ab,...cd->...acbd", a, b)
    return prod.reshape(*prod.shape[:-4], a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1])


def _kron_all(matrices: list[torch.Tensor]) -> torch.Tensor:
    """Apply the Kronecker product across a list of matrices."""
    return reduce(_kron_pair, matrices)


def _batch_size_of(x) -> int | None:
    """Return the leading batch size for batched tensors."""
    if not torch.is_tensor(x):
        return None
    if x.ndim >= 2:
        return x.shape[0]
    return None

class Circuit:
    """Layered quantum circuit made from fixed gates and resolved variable gates."""

    def __init__(self, num_qubits: int = 1, specs=None):
        """Create an empty circuit for `num_qubits` qubits."""
        self.num_qubits = num_qubits
        self.gate_layers: list[list[Gate]] = []
        self.specs = specs
        # Fixed layers are cached once; data/theta layers are resolved per batch.
        self._layer_matrix_cache: list[torch.Tensor | None] = []
        self._zero_state_cache: torch.Tensor | None = None

    def add_gates(self, layer: list[object]) -> None:
        """Add a layer with one single-qubit gate per qubit."""
        if not isinstance(layer, list):
            raise TypeError("layer must be a list of gates")
        if len(layer) != self.num_qubits:
            a = len(layer)
            b = self.num_qubits
            raise ValueError(f"layer != num of qubits {a} {b} {type(layer[0])}")

        for i, g in enumerate(layer):
            if isinstance(g, Gate):
                if g.matrix.shape[-2:] != (2, 2):
                    raise ValueError("Gate isnt 2x2 ", g.matrix.shape)
            elif isinstance(g, VGate):
                pass
            else:
                raise TypeError(f"The {i}th gate is not a Gate or VGate: {type(g)}")

        self.gate_layers.append(layer)
        self._layer_matrix_cache.append(self._static_layer_matrix(layer))

    def add_full(self, g: object) -> None:
        """Add a gate that already acts on the full Hilbert space."""
        if not isinstance(g, (Gate, VGate)):
            raise TypeError(f"full gate must be Gate or VGate, got {type(g)}")
        layer = [g]
        self.gate_layers.append(layer)
        self._layer_matrix_cache.append(self._static_layer_matrix(layer))

    def _zero_state(self) -> torch.Tensor:
        """Return the cached |0...0> state."""
        if self._zero_state_cache is None:
            dim = 1 << self.num_qubits
            state = torch.zeros(dim, device=DEVICE, dtype=DTYPE)
            state[0] = 1 + 0j
            self._zero_state_cache = state
        return self._zero_state_cache

    def _layer_matrix(self, layer_index: int, layer: list[object], x, theta) -> torch.Tensor:
        """Resolve one stored layer into its matrix form."""
        cached = self._layer_matrix_cache[layer_index]
        if cached is not None:
            return cached

        if len(layer) == self.num_qubits:
            resolved = [self._resolve_gate(g, x, theta) for g in layer]
            matrices = [rs.matrix for rs in resolved]
            return _kron_all(matrices)

        if len(layer) == 1:
            g = self._resolve_gate(layer[0], x, theta)
            return g.matrix

        raise TypeError("layer_matrix error!")

    def _static_layer_matrix(self, layer: list[object]) -> torch.Tensor | None:
        """Precompute layers that do not depend on data or trainable parameters."""
        if (
            len(layer) == self.num_qubits
            and all(isinstance(g, Gate) and g.matrix.ndim == 2 for g in layer)
        ):
            return _kron_all([g.matrix for g in layer])

        if len(layer) == 1 and isinstance(layer[0], Gate) and layer[0].matrix.ndim == 2:
            return layer[0].matrix

        return None

    def _resolve_gate(self, xgate, x, theta):
        """Turn a fixed or variable gate into a concrete `Gate`."""
        if isinstance(xgate, Gate):
            return xgate
        elif isinstance(xgate, VGate):
            return xgate.resolve(x, theta)
        else:
            raise TypeError("Unknown Type")

    def apply_to(self, state: torch.Tensor | None = None, *, x=None, theta=None) -> torch.Tensor:
        """Apply every circuit layer to `state`, or to |0...0> when no state is given."""
        dim = 1 << self.num_qubits
        batch_size = _batch_size_of(state)
        if batch_size is None:
            batch_size = _batch_size_of(x)

        if state is None:
            current = self._zero_state()
            if batch_size is not None:
                current = current.unsqueeze(0).expand(batch_size, -1)
        else:
            current = state.to(dtype=DTYPE, device=DEVICE)

        if current.ndim == 1 and current.numel() != dim:
            raise TypeError("apply_to wrong dimensions")
        if current.ndim == 2 and current.shape[-1] != dim:
            raise TypeError("apply_to wrong dimensions")

        for layer_index, layer in enumerate(self.gate_layers):
            curmat = self._layer_matrix(layer_index, layer, x, theta)
            # `curmat` may be [d,d] for fixed layers or [B,d,d] for symbolic layers.
            current = torch.matmul(curmat, current.unsqueeze(-1)).squeeze(-1)
        return current

    def __repr__(self):
        names = []
        for layer in self.gate_layers:
            names.append([getattr(g, "name", "Gate") for g in layer])
        return f"Circuit({self.num_qubits}q, layers={names})"

    def copy(self):
        """Return a shallow copy with independent layer lists."""
        new = Circuit(self.num_qubits, self.specs)
        new.gate_layers = [layer.copy() for layer in self.gate_layers]
        new._layer_matrix_cache = self._layer_matrix_cache.copy()
        new._zero_state_cache = self._zero_state_cache
        return new


@dataclass
class CircuitSpec:
    """Shape metadata used by circuit-building helpers and models."""

    num_qubits: int
    xlen: int
    tlen: int

__all__ = ["Circuit", "CircuitSpec"]
