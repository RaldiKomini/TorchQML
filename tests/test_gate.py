import unittest

from qml_lib.utils import backend as b
from qml_lib.gates.gate import Gate



class TestGate(unittest.TestCase):
    def test_gate_creation(self):
        matrix = b.tensor([[0, 1], [1, 0]])
        gate = Gate(matrix, name="X", params=[b.pi])
        
        self.assertTrue(b.allclose(gate.matrix, matrix))  # test matrix is stored
        self.assertEqual(gate.name, "X")                 # test name is correct
        self.assertEqual(gate.params, [b.pi]) 
        
        
    def test_mul(self):
        x = Gate.X
        y = Gate.Y
        res = x @ y
        excepted = x.matrix @ y.matrix
        
        self.assertTrue(b.allclose(res.matrix, excepted))