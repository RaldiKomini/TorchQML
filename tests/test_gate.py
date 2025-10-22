import unittest

from qml_lib.config import torch, DTYPE, DEVICE
from qml_lib.gates.gate import Gate



class TestGate(unittest.TestCase):
    def test_gate_creation(self):
        matrix = torch.tensor([[0, 1], [1, 0]])
        gate = Gate(matrix, name="X", params=[torch.pi])
        
        self.assertTrue(torch.allclose(gate.matrix, matrix))  # test matrix is stored
        self.assertEqual(gate.name, "X")                 # test name is correct
        self.assertEqual(gate.params, [torch.pi]) 
        
        
    def test_mul(self):
        x = Gate.X
        y = Gate.Y
        res = x @ y
        excepted = x.matrix @ y.matrix
        print(res)
        print(excepted)
        
        self.assertTrue(torch.allclose(res.matrix, excepted))