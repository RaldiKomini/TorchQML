from qml_lib.config import DEVICE, DTYPE
import torch
from qml_lib.gates.gate import *
from qml_lib.circuits.circuit import Circuit
from qml_lib.measurements.expectation_Z import expectation_Z
from qml_lib.wrappers.gatewrapper import VGate

theta = 2
gate1 = ry(theta)
gate2 = rx(theta + 0.2)
#print(gate1.matrix)

# prepare |1> by applying X first
c = Circuit(2)
c.add_gates([X, I])
c.add_gates([I, Z])

c.add_gates([X, X])
c.add_gates([gate1, gate2])
out = c.apply_to()



res = expectation_Z(out,0, 2)

print(res)