import math
import torch
from qml_lib.circuits.circuit import Circuit
from qml_lib.gates.gate import H 
from qml_lib.wrappers.gatewrapper import rx, ry, rz
from qml_lib.wrappers.symcalc import Sym, Xsym, Tsym

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
