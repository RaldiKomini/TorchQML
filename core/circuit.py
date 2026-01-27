from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.gates import Gate, VGate
import torch
from functools import reduce


def _kron_all(matrices : list[torch.Tensor])->torch.Tensor:
    return reduce(torch.kron, matrices)

class Circuit:
    def __init__(self, num_qubits: int = 1, specs = None):
        self.num_qubits = num_qubits
        self.gate_layers : list[list[Gate]] = []
        self.specs = specs

    #responsible for one qubit gates    
    def add_gates(self, layer : list[object]) -> None:
        #error list check
        if not isinstance(layer, list):
            raise TypeError("layer must be a list of gates")
        if len(layer) != self.num_qubits:
            a = len(layer)
            b = self.num_qubits
            raise ValueError(f"layer != num of qubits {a} {b} {type(layer[0])}")
        
        #error gate check
        for i, g in enumerate(layer):
            if isinstance(g, Gate):
                if g.matrix.shape != (2, 2):
                    raise ValueError("Gate isnt 2x2 ", g.matrix.shape)
            elif isinstance(g, VGate):
                pass
            else:
                raise TypeError("The {i}th Gate is not type Gate, {type(g)}")
            
        self.gate_layers.append(layer)

    #responsible for CNOT and other multiqubit gates
    def add_full(self, g :object) -> None:
        self.gate_layers.append([g])

    def _zero_state(self) -> torch.Tensor: #produces |1, 0, 0, 0, 0,...>
        dim = 1 << self.num_qubits
        state = torch.zeros(dim, device= DEVICE, dtype= DTYPE)
        state[0] = 1 + 0j
        return state
    
    def _layer_matrix(self, layer : list[object], x, theta) ->torch.Tensor: #calculates the matrix of the resultint tensor products
        if len(layer) == self.num_qubits:
            resolved = [self._resolve_gate(g, x, theta) for g in layer]
            matrices = [rs.matrix for rs in resolved]
            return _kron_all(matrices)
        if len(layer) == 1: #case of CNOT etc, matrix is already calculated
            return layer[0].matrix
        raise TypeError("layer_martix error!")
        
    def _resolve_gate(self, xgate, x, theta):
        if isinstance(xgate, Gate):
            return xgate
        elif isinstance(xgate, VGate):
            return xgate.resolve(x, theta)
        else:
            raise TypeError("Unknown Type")
            
            
    def apply_to(self, state : torch.Tensor | None = None, *, x = None, theta = None) -> torch.Tensor:
        current = self._zero_state() if state is None else state.to(dtype=DTYPE, device=DEVICE) #initialize the state to current one
        dim = 1 << self.num_qubits
        if current.numel() != dim:
            raise TypeError("apply_to wrong dimensions")
        
        for layer in self.gate_layers: #applies every layer to the current state
            curmat = self._layer_matrix(layer, x, theta)
            current = curmat @current
        return current
    

   
    def __repr__(self):
        names = []
        for layer in self.gate_layers:
            names.append([getattr(g, "name", "Gate") for g in layer])
        return f"Circuit({self.num_qubits}q, layers={names})"
    
    def copy(self):
        new = Circuit(self.num_qubits, self.specs)
        new.gate_layers = [layer.copy() for layer in self.gate_layers]
        return new




from dataclasses import dataclass

@dataclass(frozen=True)
class CircuitSpec:
    num_qubits: int #num of qubits
    xlen: int #X.shape
    tlen: int #t.lenght

__all__ = ["Circuit", "CircuitSpec"]