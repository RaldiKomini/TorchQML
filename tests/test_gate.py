import unittest
import numpy as np
from gates.gate import Gate

class TestGate(unittest.TestGate):
    def test_gate_creation(self):
        matrix = np.array([[0, 1], [1, 0]])
        gate = Gate(matrix, name="X", params=[np.pi])