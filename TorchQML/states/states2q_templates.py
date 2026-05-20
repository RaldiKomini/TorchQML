from TorchQML import Circuit
from TorchQML import H, X, I, Z, rx, ry, rz, CNOT
from TorchQML import Xsym, Tsym
import math






def state2_big(spec):
    x = Xsym()
    t = Tsym()

    num_qubits = 2
    xlen = spec.xlen
    tlen = spec.tlen

    # tlen must match: 4 + 12*num_blocks
    if (tlen - 4) % 12 != 0:
        raise ValueError(f"For this template, tlen must be 4 + 12*k. Got tlen={tlen}.")
    num_blocks = (tlen - 4) // 12

    c = Circuit(num_qubits=num_qubits)

    # --- Initial superposition ---
    c.add_gates([H] * num_qubits)

    # Break symmetry (optional but usually helps)
    c.add_gates([X, I])

    # --- Initial param-only layer: (Ry, Rz) per qubit ---
    t_idx = 0
    ry0, rz0 = [], []
    for q in range(num_qubits):
        ry0.append(ry(math.pi * t[t_idx])); t_idx += 1
        rz0.append(rz(math.pi * t[t_idx])); t_idx += 1
    c.add_gates(ry0)
    c.add_gates(rz0)

    # --- Data reuploading blocks ---
    stride = 5  # change this to mix features differently (3,5,7,...)
    for block in range(num_blocks):
        gry, grz, grx = [], [], []

        for q in range(num_qubits):
            i1 = (block * num_qubits + q) % xlen
            i2 = (block * num_qubits + q + stride) % xlen

            # Ry: pi*(a*x[i1] + b)
            angle_y = math.pi * (t[t_idx] * x[i1] + t[t_idx + 1]); t_idx += 2
            # Rz: pi*(c*x[i2] + d)
            angle_z = math.pi * (t[t_idx] * x[i2] + t[t_idx + 1]); t_idx += 2
            # Rx: pi*(e*(x[i1]+x[i2]) + f)
            angle_x = math.pi * (t[t_idx] * (x[i1] + x[i2]) + t[t_idx + 1]); t_idx += 2

            gry.append(ry(angle_y))
            grz.append(rz(angle_z))
            grx.append(rx(angle_x))

        c.add_gates(gry)
        c.add_gates(grz)
        c.add_gates(grx)

        # --- Entangling pattern (as "complex" as 2 qubits allows) ---
        if block % 3 == 0:
            c.add_full(CNOT(num_qubits, 0, 1))
        elif block % 3 == 1:
            c.add_full(CNOT(num_qubits, 1, 0))
        else:
            c.add_full(CNOT(num_qubits, 0, 1))
            c.add_full(CNOT(num_qubits, 1, 0))

    # Sanity check: we used exactly spec.tlen parameters
    assert t_idx == tlen, f"t_idx ended at {t_idx}, expected {tlen}"

    return c










def state2_poly(spec):
    x = Xsym()
    t = Tsym()

    nq = 2
    xlen = spec.xlen
    tlen = spec.tlen

    c = Circuit(num_qubits=nq)

    # --- init ---
    c.add_gates([H, H])
    c.add_gates([X, I])  # break symmetry

    t_idx = 0

    # Helper: consume one param and return it
    def p():
        nonlocal t_idx
        if t_idx >= tlen:
            raise ValueError("Ran out of parameters vs spec.tlen")
        v = t[t_idx]
        t_idx += 1
        return v

    # Initial trainable layer (4 params)
    c.add_gates([ry(math.pi * p()), ry(math.pi * p())])
    c.add_gates([rz(math.pi * p()), rz(math.pi * p())])

    # Use the rest in blocks. Each block uses 12 params (2 qubits * 3 gates * (a,b))
    # If tlen doesn't fit exactly, we'll use remaining params in a final mixer.
    stride = 5
    block = 0
    while (tlen - t_idx) >= 12:
        # feature indices
        i0 = (2*block + 0) % xlen
        i1 = (2*block + 1) % xlen
        j0 = (2*block + 0 + stride) % xlen
        j1 = (2*block + 1 + stride) % xlen

        # Nonlinear cross terms (quadratic)
        cross0 = x[i0] * x[j0]
        cross1 = x[i1] * x[j1]

        # --- encode + train (a*x + b), plus (a2*cross + b2) ---
        # qubit 0
        c.add_gates([ry(math.pi * (p()*x[i0]   + p())), I])
        c.add_gates([rz(math.pi * (p()*x[j0]   + p())), I])
        c.add_gates([rx(math.pi * (p()*cross0  + p())), I])

        # qubit 1
        c.add_gates([I, ry(math.pi * (p()*x[i1]   + p()))])
        c.add_gates([I, rz(math.pi * (p()*x[j1]   + p()))])
        c.add_gates([I, rx(math.pi * (p()*cross1  + p()))])

        # Entangle (alternate direction to avoid symmetry traps)
        if block % 2 == 0:
            c.add_full(CNOT(nq, 0, 1))
        else:
            c.add_full(CNOT(nq, 1, 0))

        block += 1

    # Final mixer to spend any leftover params (0..11 params possible)
    # This keeps spec.tlen flexible without forcing a rigid formula.
    while t_idx < tlen:
        # cycle through single-qubit mixers
        if (t_idx % 3) == 0:
            c.add_gates([ry(math.pi * p()), I] if (t_idx % 2 == 0) else [I, ry(math.pi * p())])
        elif (t_idx % 3) == 1:
            c.add_gates([rz(math.pi * p()), I] if (t_idx % 2 == 0) else [I, rz(math.pi * p())])
        else:
            c.add_gates([rx(math.pi * p()), I] if (t_idx % 2 == 0) else [I, rx(math.pi * p())])

    assert t_idx == tlen, f"Used {t_idx}, expected {tlen}"
    return c






import math


def state2cr(spec):
    x = Xsym()
    t = Tsym()

    nq   = spec.nq if hasattr(spec, "nq") else 8
    xlen = spec.xlen
    tlen = spec.tlen

    c = Circuit(num_qubits=nq)

    # ---------- param helper ----------
    t_idx = 0
    def p():
        nonlocal t_idx
        if t_idx >= tlen:
            raise ValueError("Ran out of parameters vs spec.tlen")
        v = t[t_idx]
        t_idx += 1
        return v

    # ---------- init (break symmetry hard) ----------
    c.add_gates([H] * nq)
    # alternating X's to break symmetry
    c.add_gates([X if (q % 2 == 0) else I for q in range(nq)])

    # ---------- helpers ----------
    def entangle_weird(block):
        """
        Ring + cross + stagger, alternating directions.
        This creates lots of long-range correlations and avoids symmetric traps.
        """
        # ring
        if block % 2 == 0:
            for q in range(nq - 1):
                c.add_full(CNOT(nq, q, q + 1))
            c.add_full(CNOT(nq, nq - 1, 0))
        else:
            for q in range(nq - 1, 0, -1):
                c.add_full(CNOT(nq, q, q - 1))
            c.add_full(CNOT(nq, 0, nq - 1))

        # cross (pair q with q+nq/2)
        half = nq // 2
        if half >= 1:
            for q in range(half):
                if (q + block) % 2 == 0:
                    c.add_full(CNOT(nq, q, q + half))
                else:
                    c.add_full(CNOT(nq, q + half, q))

        # stagger (skip connections)
        step = 2 + (block % 3)  # 2,3,4 repeating
        for q in range(nq):
            a = q
            b = (q + step) % nq
            if (q + block) % 3 == 0:
                c.add_full(CNOT(nq, a, b))
            elif (q + block) % 3 == 1:
                c.add_full(CNOT(nq, b, a))
            else:
                # controlled pattern in both directions but not too much
                c.add_full(CNOT(nq, a, b))

    def train_layer():
        # per-qubit trainable mixer: Y then Z (strong but stable)
        c.add_gates([ry(math.pi * p()) for _ in range(nq)])
        c.add_gates([rz(math.pi * p()) for _ in range(nq)])

    def data_layer(block):
        """
        Explicit sin/cos encoding + mild polynomial interactions.
        We encode different combos per qubit so we don't collapse to something linear.
        """
        stride1 = 3 + 2 * (block % 2)   # 3,5 alternating
        stride2 = 7 + (block % 3)       # 7,8,9 repeating

        gatesY = []
        gatesZ = []
        gatesX = []

        for q in range(nq):
            i  = (q + block) % xlen
            j  = (q + stride1 + 2*block) % xlen
            k  = (q + stride2 + 3*block) % xlen

            # explicit nonlinear features
            s  = sin(x[i])
            c0 = cos(x[j])

            # multiplicative interactions (nonlinear + cross-feature)
            cross = x[i] * x[j]
            tri   = x[i] * x[j] * x[k]

            # "random-ish" per-qubit mixture, still deterministic
            # Each angle is (a * feature + b * feature2 + bias)
            a1, b1, d1 = p(), p(), p()
            a2, b2, d2 = p(), p(), p()
            a3, b3, d3 = p(), p(), p()

            angY = math.pi * (a1 * s      + b1 * cross + d1)
            angZ = math.pi * (a2 * c0     + b2 * tri   + d2)
            angX = math.pi * (a3 * x[k]   + b3 * (s*c0) + d3)  # extra nonlinearity

            gatesY.append(ry(angY))
            gatesZ.append(rz(angZ))
            gatesX.append(rx(angX))

        # apply as full layers (vectorized, like your style)
        c.add_gates(gatesY)
        c.add_gates(gatesZ)
        c.add_gates(gatesX)

    # ---------- budgeted blocks ----------
    # Each block uses:
    #   train_layer: 2*nq params
    #   data_layer:  9*nq params  (three gates * (a,b,bias))
    #   train_layer: 2*nq params
    # Total = 13*nq per block
    per_block = 13 * nq

    # spend some initial trainables (helps avoid "all data then tiny train")
    # uses 2*nq params
    if (tlen - t_idx) >= 2 * nq:
        train_layer()

    block = 0
    while (tlen - t_idx) >= per_block:
        # sandwich: train -> data -> entangle -> train -> entangle
        train_layer()
        data_layer(block)
        entangle_weird(block)
        train_layer()
        entangle_weird(block + 1)
        block += 1

    # ---------- leftover spend (graceful) ----------
    # if you have some params left, do a strong mixer + a smaller data splash
    while (tlen - t_idx) >= 2 * nq:
        train_layer()
        entangle_weird(block)
        block += 1

    # if still leftover, just burn them as single-qubit rotations alternating qubits
    while t_idx < tlen:
        q = t_idx % nq
        # build a layer with identity everywhere except one qubit
        layer = [I] * nq
        if (t_idx % 3) == 0:
            layer[q] = ry(math.pi * p())
        elif (t_idx % 3) == 1:
            layer[q] = rz(math.pi * p())
        else:
            layer[q] = rx(math.pi * p())
        c.add_gates(layer)

    assert t_idx == tlen, f"Used {t_idx}, expected {tlen}"
    return c



import math


def staterp(spec, K=8):
    """
    RP-friendly VQC state (no sin/cos).
    Each rotation angle uses a learned linear combination of K input dims + bias,
    so the circuit actually "sees" many coordinates (important for random projection).

    Params:
      optional init mixer: 2*nq
      each block: nq * 3*(K+1)   (RY,RZ,RX each uses K weights + bias)
      leftovers: burned as single-qubit mixers
    """
    x = Xsym()
    t = Tsym()

    nq   = spec.nq if hasattr(spec, "nq") else 4   # 4+ recommended; 2 works but weaker
    xlen = spec.xlen
    tlen = spec.tlen

    c = Circuit(num_qubits=nq)

    # ---- param helper ----
    t_idx = 0
    def p():
        nonlocal t_idx
        if t_idx >= tlen:
            raise ValueError("Ran out of parameters vs spec.tlen")
        v = t[t_idx]
        t_idx += 1
        return v

    # ---- init ----
    c.add_gates([H] * nq)
    c.add_gates([X if (q % 2 == 0) else I for q in range(nq)])  # break symmetry

    # ---- small trainable mixer up front (if budget) ----
    if (tlen - t_idx) >= 2 * nq:
        c.add_gates([ry(math.pi * p()) for _ in range(nq)])
        c.add_gates([rz(math.pi * p()) for _ in range(nq)])

    def entangle(block):
        # ring, alternating direction
        if nq >= 2:
            if block % 2 == 0:
                for q in range(nq - 1):
                    c.add_full(CNOT(nq, q, q + 1))
                c.add_full(CNOT(nq, nq - 1, 0))
            else:
                for q in range(nq - 1, 0, -1):
                    c.add_full(CNOT(nq, q, q - 1))
                c.add_full(CNOT(nq, 0, nq - 1))

        # a cheap long-range link
        if nq >= 4:
            c.add_full(CNOT(nq, 0, nq // 2))

    def linmix_angle(block, q, gate_id):
        """
        angle = pi * ( sum_{r=0..K-1} a_r * x[idx_r] + b )
        idx_r chosen to cover many dims across blocks/qubits/gates.
        """
        expr = x[0] * 0.0

        base = (block * (nq * K * 3) + q * (K * 3) + gate_id * K) % xlen
        stride = 17 + 2 * gate_id  # 17, 19, 21 ... spreads indices for xlen=256

        for r in range(K):
            idx = (base + r * stride) % xlen
            expr = expr + p() * x[idx]

        expr = expr + p()  # bias
        return math.pi * expr

    # ---- main blocks ----
    per_block = nq * 3 * (K + 1)
    block = 0

    while (tlen - t_idx) >= per_block:
        gry, grz, grx = [], [], []

        for q in range(nq):
            gry.append(ry(linmix_angle(block, q, gate_id=0)))
            grz.append(rz(linmix_angle(block, q, gate_id=1)))
            grx.append(rx(linmix_angle(block, q, gate_id=2)))

        c.add_gates(gry); entangle(block)
        c.add_gates(grz); entangle(block + 1)
        c.add_gates(grx); entangle(block + 2)

        block += 1

    # ---- spend leftovers: keep it stable rather than wasting params ----
    while (tlen - t_idx) >= nq:
        layer = [I] * nq
        q = (t_idx // nq) % nq
        # cycle ry/rz/rx
        if (t_idx % 3) == 0:
            layer[q] = ry(math.pi * p())
        elif (t_idx % 3) == 1:
            layer[q] = rz(math.pi * p())
        else:
            layer[q] = rx(math.pi * p())
        c.add_gates(layer)

    # burn any last params (rare)
    while t_idx < tlen:
        _ = p()

    assert t_idx == tlen, f"Used {t_idx}, expected {tlen}"
    return c




__all__ = ['state2_big', 'state2_poly', 'state2cr', 'staterp']
