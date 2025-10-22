from qml_lib.config import DEVICE, DTYPE
from qml_lib.gates.gate import Gate
import torch
from functools import reduce


def _kron_all(matrices : list[torch.Tensor])->torch.Tensor:
    return reduce(torch.kron, matrices)

class Circuit:
    def __init__(self, num_qubits: int = 1,):
        self.num_qubits = num_qubits
        self.gate_layers : list[list[Gate]] = []

    #responsible for one qubit gates    
    def add_gates(self, layer : list[Gate]) -> None:
        #error list check
        if not isinstance(layer, list):
            raise TypeError("layer must be a list of gates")
        if len(layer) != self.num_qubits:
            raise ValueError("layer != num of qubits")
        
        #error gate check
        for i, g in enumerate(layer):
            if not isinstance(g, Gate):
                raise TypeError("The {i}th Gate is not type Gate, {type(g)}")
            if g.matrix.shape != (2, 2):
                raise ValueError("Gate isnt 2x2")
            
        self.gate_layers.append(layer)

    #responsible for CNOT etc
    def add_full(self, g :Gate) -> None:
        if not isinstance(g, Gate):
            raise TypeError("Full gate is not a Gate")
        dim = 1 << self.num_qubits #dimension of gate should be accoridng to the number of qubits
        if g.matrix.shape != (dim, dim):
            raise TypeError("Full gate has wrong dimensions")
        self.gate_layers.append([g])

    def _zero_state(self) -> torch.Tensor: #produces |1, 0, 0, 0, 0,...>
        dim = 1 << self.num_qubits
        state = torch.zeros(dim, device= DEVICE, dtype= DTYPE)
        state[0] = 1 + 0j
        return state
    
    def _layer_matrix(self, layer : list[Gate]) ->torch.Tensor: #calculates the matrix of the resultint tensor products
        if len(layer) == self.num_qubits:
            matrices = [g.matrix for g in layer]
            return _kron_all(matrices)
        if len(layer) == 1: #case of CNOT etc, matrix is already calculated
            return layer[0].matrix
        raise TypeError("layer_martix error!")
        
                
            
    def apply_to(self, state : torch.Tensor | None = None) -> torch.Tensor:
        current = self._zero_state() if state is None else state.to(dtype=DTYPE, device=DEVICE) #initialize the state to current one
        dim = 1 << self.num_qubits
        if current.numel() != dim: #checks dimensions
            raise TypeError("apply_to wrong dimensions")
        
        for layer in self.gate_layers: #applies every layer to the current state
            curmat = self._layer_matrix(layer)
            current = curmat @current
        return current
    

   
    def __repr__(self):
        names = []
        for layer in self.gate_layers:
            names.append([getattr(g, "name", "Gate") for g in layer])
        return f"Circuit({self.num_qubits}q, layers={names})"

