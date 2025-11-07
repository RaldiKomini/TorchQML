import torch
#from qml_lib.wrappers.symcalc import Sym, Xsym, Tsym, pii
#from qml_lib.wrappers.gatewrapper import VGate, rx, _rxn
from qml_lib.gates.gate import H
from qml_lib.wrappers.symcalc import pii

from qml_lib.circuits.circuit import Circuit
from qml_lib.wrappers.gatewrapper import rx, ry, rz


x = torch.tensor([2, 3, 5])
t = torch.tensor([3, 4, 6])
"""


x0 = Sym("x", 2)
t1 = Sym("t", 0)

s = 3 + x0* t1 / 2 / x0 -t1
print(s.eval(x, t))

xsym = Xsym()
a = xsym[2]
print(a.eval(x, t))

x_data = torch.tensor([10.0, 20.0, 30.0])
t_data = torch.tensor([1.0, 2.0])

expr = pii + 2 * x[1] - t[0]   # π + 2*20 - 1

print(expr.op, expr.args)     # should be "sub", (add_expr, t[0])
print(expr.eval(x_data, t_data))
"""
import torch
from qml_lib.config import DEVICE, DTYPE

# paste your Sym, to_sym, Xsym, Tsym, pii definitions here (with the fixes)

c = Circuit(1)

# start in H|0>
c.add_gates([H])


c.add_gates([rx(pii * t[1] + x[0])])

# more crazy stuff:
c.add_gates([ry(2 * pii * t[2] - 0.3 * x[1])])
c.add_gates([rz(t[0] * (x[0] + x[1]))])

psi = c.apply_to(x= x, theta=t)
print(psi)