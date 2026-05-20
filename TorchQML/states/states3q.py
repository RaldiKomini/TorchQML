from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT
from TorchQML import Xsym, Tsym
import math

def state3_poly(spec):
    x = Xsym()
    t = Tsym()

    nq = 3
    xlen = spec.xlen
    tlen = spec.tlen

    c = Circuit(num_qubits=nq)

    # --- init ---
    c.add_gates([H, H, H])
    c.add_gates([X, I, X])  # break symmetry harder

    t_idx = 0
    def p():
        nonlocal t_idx
        if t_idx >= tlen:
            raise ValueError("Ran out of parameters vs spec.tlen")
        v = t[t_idx]
        t_idx += 1
        return v

    # Initial trainable layer (6 params)
    c.add_gates([ry(math.pi * p()), ry(math.pi * p()), ry(math.pi * p())])
    c.add_gates([rz(math.pi * p()), rz(math.pi * p()), rz(math.pi * p())])

    stride = 7
    block = 0

    # Each full block uses 18 params:
    # 3 qubits x (Ry(a*x+b) + Rz(c*x+b) + Rx(e*cross+b)) => 3*(2+2+2)=18
    while (tlen - t_idx) >= 18:
        # indices per qubit
        i = [(3*block + q) % xlen for q in range(3)]
        j = [(3*block + q + stride) % xlen for q in range(3)]

        cross01 = x[i[0]] * x[j[1]]
        cross12 = x[i[1]] * x[j[2]]
        cross20 = x[i[2]] * x[j[0]]
        crosses = [cross01, cross12, cross20]

        # Encode each qubit with one linear feature and one cross feature
        gates1 = []
        gates2 = []
        gates3 = []
        for q in range(3):
            gates1.append(ry(math.pi * (p()*x[i[q]] + p())))
            gates2.append(rz(math.pi * (p()*x[j[q]] + p())))
            gates3.append(rx(math.pi * (p()*crosses[q] + p())))

        c.add_gates(gates1)
        c.add_gates(gates2)
        c.add_gates(gates3)

        # Entangling: alternate ring + "parity" links
        if block % 2 == 0:
            c.add_full(CNOT(nq, 0, 1))
            c.add_full(CNOT(nq, 1, 2))
            c.add_full(CNOT(nq, 2, 0))
        else:
            c.add_full(CNOT(nq, 0, 2))
            c.add_full(CNOT(nq, 2, 1))
            c.add_full(CNOT(nq, 1, 0))

        block += 1

    # Spend leftovers as a final mixer layer
    while t_idx < tlen:
        q = (t_idx) % 3
        r = (t_idx) % 3
        if r == 0:
            c.add_gates([ry(math.pi * p()) if k == q else I for k in range(3)])
        elif r == 1:
            c.add_gates([rz(math.pi * p()) if k == q else I for k in range(3)])
        else:
            c.add_gates([rx(math.pi * p()) if k == q else I for k in range(3)])

    assert t_idx == tlen, f"Used {t_idx}, expected {tlen}"
    return c
