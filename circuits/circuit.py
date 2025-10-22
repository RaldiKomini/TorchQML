from qml_lib.utils import backend as b
from qml_lib.gates import expand as ex
from qml_lib.gates.gate import Gate
from qml_lib.wrappers.gatewrapper import GateWrapper


class Circuit:
    def __init__(self, num_qubits = 1, initial_state=None):
        self.num_qubits = num_qubits
        self.gate_layers = []
        
    def add_gates(self, *gate_layers):
        for layer in gate_layers:
            if isinstance(layer, list):
                if len(layer) != self.num_qubits:
                    raise ValueError("Layer lenght != num of qubits!")
                
                self.gate_layers.append(layer)
            elif isinstance(layer, (Gate, GateWrapper)):
                dim = 2 ** self.num_qubits
                if isinstance(layer, Gate) and layer.matrix.shape != (dim, dim):
                    raise ValueError("Gate doesnt match circuit dim!")
                self.gate_layers.append([layer])
                
            
    def apply_to(self, state, x=None, t=None):
        current = state
        for layer in self.gate_layers:
            # Resolve each gate in the layer if it is a GateWrapper
            resolved_gates = []
            for g in layer:
                if isinstance(g, GateWrapper):
                    if x is None or t is None:
                        raise ValueError("GateWrapper requires x and t to resolve")
                    resolved_gate = g.resolve(x, t)
                    resolved_gates.append(resolved_gate)
                else:
                    resolved_gates.append(g)
            
            # Expand and apply gates
            if len(resolved_gates) == self.num_qubits:
                full_gate = ex.expand_gate(resolved_gates)
                current = full_gate(current)
            elif len(resolved_gates) == 1:
                current = resolved_gates[0](current)
            else:
                raise ValueError("Invalid gate layer size")
        
        return current
    
    #if you want to remove the gates -> gates = 1        
    """
    def reset(self, state=None, gates = 0):
        dim = 2 ** self.num_qubits
        if state is not None:
            self.state = state
        else:
            self.state = b.zeros((dim,))
            self.state[0] = 1.0
        if gates == 1:
            self.gate_layers = []

       
    def measure(self):
        prob = b.abs(self.state ** 2)
        return prob
    
    """

        
    def __repr__(self):
        layers_names = []
        for layer in self.gate_layers:
            layer_names = []
            for g in layer:
                # GateWrapper might not have matrix, so just use name
                name = getattr(g, 'name', repr(g))
                layer_names.append(name)
            layers_names.append(layer_names)
        return f"Circuit(gates={layers_names})"

