from qml_lib.utils import backend as b

class Gate:
    def __init__(self, matrix, name="Not specified", params=None):
        self.matrix = matrix
        self.name = name
        self.params = params or []
        
    def __matmul__(self, other):
        if not isinstance(other, Gate):
            raise TypeError("Not a gate!")
            
        newMatrix = b.matmul(self.matrix, other.matrix)
        newName = f"({self.name} @ {other.name})"
        newParams = self.params + other.params
        return Gate(newMatrix, newName, newParams)
        
    def __repr__(self):
        return f"Gate name {self.name}, params = {self.params}"
    
    def __call__(self, state):
        return b.matmul(self.matrix, state)
    
    def dagger(self):
        # Hermitian conjugate: conjugate transpose
        new_matrix = b.conj(b.transpose(self.matrix))
        return Gate(new_matrix, f"{self.name}.dagger()", self.params)
    
    def __invert__(self):
        return self.dagger()



#starting state


#Classic quantum gates
Gate.I = Gate(b.tensor([[1,0], [0,1]], dtype = b.cfloat), "I")
Gate.X = Gate(b.tensor([[0,1], [1,0]], dtype = b.cfloat), "X")
Gate.Y = Gate(b.tensor([[0, -1j], [1j, 0]], dtype=b.cfloat), "Y")
Gate.Z = Gate(b.tensor([[1, 0], [0, -1]], dtype=b.cfloat), "Z")
Gate.H = Gate(b.tensor([[1, 1], [1, -1]], dtype=b.cfloat) / b.sqrt(b.tensor(2.0)), "H")

Gate.CNOT = Gate(b.tensor([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype = b.cfloat), "CNOT")

#X, Y, Z with theta
def Rotx(theta):
    theta = b.tensor(theta, dtype= b.cfloat)
    matrix =  b.tensor([[b.cos(theta/2), -1j*b.sin(theta/2)],
                        [-1j*b.sin(theta/2), b.cos(theta/2)]], dtype = b.cfloat)    
    name = f"RotX(t={theta})"
    params = [theta]
    return Gate(matrix, name, params)

def Roty(theta):
    theta = b.tensor(theta, dtype= b.cfloat)
    matrix =  b.tensor([[b.cos(theta/2), -b.sin(theta/2)],
                        [b.sin(theta/2), b.cos(theta/2)]], dtype = b.cfloat)    
    name = f"RotY(t={theta})"
    params = [theta]
    return Gate(matrix, name, params)

def Rotz(theta):
    theta = b.tensor(theta, dtype= b.cfloat)
    matrix =  b.tensor([[b.exp(-1j * theta / 2), 0],
                        [0, b.exp(1j * theta / 2)]], dtype = b.cfloat)    
    name = f"RotZ(t={theta})"
    params = [theta]
    return Gate(matrix, name, params)