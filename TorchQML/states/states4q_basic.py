
from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT, CRY, CRX, CRZ
from TorchQML import Xsym, Tsym
import math



def state4_small(specs): #tlen = 8 + 24B
    x = Xsym()   # symbolic inputs
    t = Tsym()   # symbolic params

    num_qubits = 4
    num_blocks = 2   # more blocks = more data reuploading
    xlen = specs.xlen

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry with some X's
    init_gates = []
    for q in range(num_qubits):
        init_gates.append(X if q % 2 == 0 else I)
    c.add_gates(init_gates)

# --- Initial param-only layer (2 t per qubit) ---
    t_idx = 0
    gates_ry0 = []
    gates_rz0 = []
    for q in range(num_qubits):
        angle_y0 = math.pi * t[t_idx]; t_idx += 1
        angle_z0 = math.pi * t[t_idx]; t_idx += 1
        gates_ry0.append(ry(angle_y0))
        gates_rz0.append(rz(angle_z0))
    c.add_gates(gates_ry0)
    c.add_gates(gates_rz0)

    # --- Data reuploading blocks ---
    for block in range(num_blocks):
        gates_ry = []
        gates_rz = []
        gates_rx = []

        for q in range(num_qubits):
            # pick 2 feature indices per (block, qubit)
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + 7) % xlen  # offset to mix features

            # Ry: pi * (a * x[i1] + b)
            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
            t_idx += 2

            # Rz: pi * (c * x[i2] + d)
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1])
            t_idx += 2

            # Rx: pi * (e * (x[i1] + x[i2]) + f)
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1])
            t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        c.add_gates(gates_ry)
        c.add_gates(gates_rz)
        c.add_gates(gates_rx)

        # --- Entangling pattern ---
        if block % 2 == 0:
            # even blocks: nearest-neighbor ring
            for q in range(num_qubits):
                q_next = (q + 1) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))
        else:
            # odd blocks: skip + long range
            c.add_full(CNOT(num_qubits, 0, 2))
            c.add_full(CNOT(num_qubits, 1, 3))
            c.add_full(CNOT(num_qubits, 0, 3))

    # t_idx should end at 152
    # (you can assert this if you want)
    return c





def state4_medium(specs): #tlen = 8 + 24B
    x = Xsym()   # symbolic inputs
    t = Tsym()   # symbolic params

    num_qubits = 4
    num_blocks = 4   # more blocks = more data reuploading
    xlen = specs.xlen

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry with some X's
    init_gates = []
    for q in range(num_qubits):
        init_gates.append(X if q % 2 == 0 else I)
    c.add_gates(init_gates)

# --- Initial param-only layer (2 t per qubit) ---
    t_idx = 0
    gates_ry0 = []
    gates_rz0 = []
    for q in range(num_qubits):
        angle_y0 = math.pi * t[t_idx]; t_idx += 1
        angle_z0 = math.pi * t[t_idx]; t_idx += 1
        gates_ry0.append(ry(angle_y0))
        gates_rz0.append(rz(angle_z0))
    c.add_gates(gates_ry0)
    c.add_gates(gates_rz0)

    # --- Data reuploading blocks ---
    for block in range(num_blocks):
        gates_ry = []
        gates_rz = []
        gates_rx = []

        for q in range(num_qubits):
            # pick 2 feature indices per (block, qubit)
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + 7) % xlen  # offset to mix features

            # Ry: pi * (a * x[i1] + b)
            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
            t_idx += 2

            # Rz: pi * (c * x[i2] + d)
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1])
            t_idx += 2

            # Rx: pi * (e * (x[i1] + x[i2]) + f)
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1])
            t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        c.add_gates(gates_ry)
        c.add_gates(gates_rz)
        c.add_gates(gates_rx)

        # --- Entangling pattern ---
        if block % 2 == 0:
            # even blocks: nearest-neighbor ring
            for q in range(num_qubits):
                q_next = (q + 1) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))
        else:
            # odd blocks: skip + long range
            c.add_full(CNOT(num_qubits, 0, 2))
            c.add_full(CNOT(num_qubits, 1, 3))
            c.add_full(CNOT(num_qubits, 0, 3))

    # t_idx should end at 152
    # (you can assert this if you want)
    return c




def state4_big(specs): #tlen = 8 + 24B
    x = Xsym()   # symbolic inputs
    t = Tsym()   # symbolic params

    num_qubits = 4
    num_blocks = 6   # more blocks = more data reuploading
    xlen = specs.xlen

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry with some X's
    init_gates = []
    for q in range(num_qubits):
        init_gates.append(X if q % 2 == 0 else I)
    c.add_gates(init_gates)

# --- Initial param-only layer (2 t per qubit) ---
    t_idx = 0
    gates_ry0 = []
    gates_rz0 = []
    for q in range(num_qubits):
        angle_y0 = math.pi * t[t_idx]; t_idx += 1
        angle_z0 = math.pi * t[t_idx]; t_idx += 1
        gates_ry0.append(ry(angle_y0))
        gates_rz0.append(rz(angle_z0))
    c.add_gates(gates_ry0)
    c.add_gates(gates_rz0)

    # --- Data reuploading blocks ---
    for block in range(num_blocks):
        gates_ry = []
        gates_rz = []
        gates_rx = []

        for q in range(num_qubits):
            # pick 2 feature indices per (block, qubit)
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + 7) % xlen  # offset to mix features

            # Ry: pi * (a * x[i1] + b)
            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1])
            t_idx += 2

            # Rz: pi * (c * x[i2] + d)
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1])
            t_idx += 2

            # Rx: pi * (e * (x[i1] + x[i2]) + f)
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1])
            t_idx += 2

            gates_ry.append(ry(angle_y))
            gates_rz.append(rz(angle_z))
            gates_rx.append(rx(angle_x))

        c.add_gates(gates_ry)
        c.add_gates(gates_rz)
        c.add_gates(gates_rx)

        # --- Entangling pattern ---
        if block % 2 == 0:
            # even blocks: nearest-neighbor ring
            for q in range(num_qubits):
                q_next = (q + 1) % num_qubits
                c.add_full(CNOT(num_qubits, q, q_next))
        else:
            # odd blocks: skip + long range
            c.add_full(CNOT(num_qubits, 0, 2))
            c.add_full(CNOT(num_qubits, 1, 3))
            c.add_full(CNOT(num_qubits, 0, 3))

    # t_idx should end at 152
    # (you can assert this if you want)
    return c


def state4_bigger(specs, num_blocks=6):
    x = Xsym()
    t = Tsym()

    nq = 4
    xlen = specs.xlen
    c = Circuit(num_qubits=nq)

    c.add_gates([H] * nq)
    c.add_gates([X if q % 2 == 0 else I for q in range(nq)])

    t_idx = 0

    # initial param-only layer: RZ-RY-RZ per qubit
    g1=g2=g3=[]
    g1=[]; g2=[]; g3=[]
    for q in range(nq):
        g1.append(rz(math.pi * t[t_idx])); t_idx += 1
        g2.append(ry(math.pi * t[t_idx])); t_idx += 1
        g3.append(rz(math.pi * t[t_idx])); t_idx += 1
    c.add_gates(g1); c.add_gates(g2); c.add_gates(g3)

    for block in range(num_blocks):
        # --- data layer (same structure as you, but keep it) ---
        gy=[]; gz=[]; gx=[]
        for q in range(nq):
            i1 = (block * nq + q) % xlen
            i2 = (block * nq + q + 7) % xlen

            ay = t[t_idx]; by = t[t_idx+1]; t_idx += 2
            az = t[t_idx]; bz = t[t_idx+1]; t_idx += 2
            ax = t[t_idx]; bx = t[t_idx+1]; t_idx += 2

            gy.append(ry(math.pi * (ay * x[i1] + by)))
            gz.append(rz(math.pi * (az * x[i2] + bz)))
            gx.append(rx(math.pi * (ax * (x[i1] + x[i2]) + bx)))

        c.add_gates(gy); c.add_gates(gz); c.add_gates(gx)

        # --- NEW: param-only mixer after data embedding ---
        m1=[]; m2=[]; m3=[]
        for q in range(nq):
            m1.append(rz(math.pi * t[t_idx])); t_idx += 1
            m2.append(ry(math.pi * t[t_idx])); t_idx += 1
            m3.append(rz(math.pi * t[t_idx])); t_idx += 1
        c.add_gates(m1); c.add_gates(m2); c.add_gates(m3)

        # --- alternating entanglers ---
        if block % 2 == 0:
            for q in range(nq):
                c.add_full(CNOT(nq, q, (q+1) % nq))
        else:
            c.add_full(CNOT(nq, 0, 2))
            c.add_full(CNOT(nq, 1, 3))
            c.add_full(CNOT(nq, 0, 3))

    return c



import math


__all__ = ['state4_small', 'state4_medium', 'state4_big', 'state4_bigger']
