from qml_lib.config import DEVICE, DTYPE
import torch
from qml_lib.gates.gate import *
from qml_lib.circuits.circuit import Circuit


theta = 2
gate1 = ry(theta)
#print(gate1.matrix)

# prepare |1> by applying X first
c = Circuit(1)
c.add_gates([X])
c.add_gates([Z])
out = c.apply_to()
expected = torch.tensor([0, -1], dtype=DTYPE, device=DEVICE)
print(torch.allclose(out, expected))

