import math
import torch
from TorchQML.circuits.circuit import Circuit
from TorchQML.gates.gate import H 
from TorchQML.wrappers.gatewrapper import rx, ry, rz
from TorchQML.wrappers.symcalc import Sym, Xsym, Tsym

def stateg() -> torch.Tensor:

    x = Xsym()
    t = Tsym()

    c = Circuit(num_qubits=1)

    c.add_gates([H])

    c.add_gates([ry(t[0])])

    c.add_gates([rx(math.pi * x[0])])

    c.add_gates([ry(t[1])])

    c.add_gates([rx(math.pi * x[1])])

    c.add_gates([rz(t[2])])

    c.add_gates([ry(t[3])])

    c.add_gates([rx(0.5 * math.pi * x[0])])

    c.add_gates([ry(0.5 * math.pi * x[1])])

    c.add_gates([rz(t[4])])

    c.add_gates([ry(t[5])])

    c.add_gates([rx(t[6] * math.pi * (x[0] + x[1]))])

 

    return c


def states() -> torch.Tensor:
    x = Xsym()   # image features: x[0], x[1], x[2], x[3]
    t = Tsym()   # trainable parameters t[0]..t[6]

    c = Circuit(num_qubits=1)

    c.add_gates([H])

    c.add_gates([ry(math.pi * x[0])])
    c.add_gates([rx(math.pi * x[1])])
    c.add_gates([rz(math.pi * x[2])])
    c.add_gates([ry(math.pi * x[3])])

 
    c.add_gates([rz(t[0])])
    c.add_gates([ry(t[1])])

    c.add_gates([rx(t[2] * (x[0] + x[1]))])
    c.add_gates([ry(t[3] * (x[2] + x[3]))])

    c.add_gates([rz(t[4])])
    c.add_gates([ry(t[5])])
    c.add_gates([rx(t[6])])

    return c
