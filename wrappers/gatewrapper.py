# qml_lib/wrappers/gatewrapper.py
import torch
from typing import Tuple, Union, Callable
from qml_lib.gates.gate import Gate  # your Gate class

Source = Tuple[str, Union[int, float]]  # ("x", k) | ("theta", j) | ("const", v)

class VGate:
    """
    Runtime-resolved single-qubit gate (for data reuploading & trainable params).

    ctor:   a function angle -> Gate (e.g., rx, ry, rz)
    source: ("x", k) | ("theta", j) | ("const", v)
    qubit:  which wire this gate is intended for (informational)
    """
    def __init__(self, ctor: Callable[[torch.Tensor], Gate], source: Source, qubit: int, name: str | None = None):
        self.ctor = ctor
        self.source = source
        self.qubit = int(qubit)
        self.name = name or getattr(ctor, "__name__", "VGate")

    def resolve(self, x: torch.Tensor | None = None, theta: torch.Tensor | None = None) -> Gate:
        kind, pos = self.source
        if kind == "x":
            if x is None:
                raise ValueError("VGate needs x to resolve")
            angle = x[pos]
        elif kind == "theta":
            if theta is None:
                raise ValueError("VGate needs theta to resolve")
            angle = theta[pos]
        elif kind == "const":
            angle = torch.as_tensor(pos, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown source kind: {kind}")

        # ctor builds the concrete Gate (e.g., rx(angle) -> Gate)
        return self.ctor(angle)
