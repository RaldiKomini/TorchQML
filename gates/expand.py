from qml_lib.utils import backend as b
from qml_lib.gates import gate
from functools import reduce

def expand_gate(layer):
    #layer = layer[::-1]
    full_matrix = reduce(lambda x, y: torch.kron(x,y), (g.matrix for g in layer))
    full_name = " x ".join(g.name for g in layer)
    
    full_params = []
    for g in layer:
        full_params.extend(g.params if isinstance(g.params, list) else [g.params])
        
    return gate.Gate(full_matrix, full_name, full_params)

