from utils import backend as b

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
