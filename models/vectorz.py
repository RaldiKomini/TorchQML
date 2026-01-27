import torch
import torch.nn as nn
from TorchQML.core.config import DEVICE, DTYPE
from TorchQML.encoding.amplitude import AmpEnc
import torch.nn.functional as F

def expval_Z_fast_batch(psi: torch.Tensor, tg: int, nq: int) -> torch.Tensor:
    p = (psi.conj() * psi).real

    bitpos = (nq - 1 - tg)  # adjust if your convention differs
    idx = torch.arange(p.shape[-1], device=psi.device)
    bit = (idx >> bitpos) & 1
    sign = (1.0 - 2.0 * bit).to(p.dtype)

    return (p * sign).sum(dim=-1)


def calcZvec(xb, theta, circ, spec, amp_enc):
    xb = xb.to(device=theta.device, dtype=theta.dtype)
    B = xb.shape[0]
    nq = spec.num_qubits

    Zmat = torch.empty(B, nq, device=xb.device, dtype=xb.real.dtype)

    # --- run circuit per sample ---
    psis = []

    for x in xb:
        if amp_enc is None:
            psi = circ.apply_to(x=x, theta=theta)
        else:
            psi0 = amp_enc(x)
            psi = circ.apply_to(state=psi0, x = x, theta=theta)

        psis.append(psi)

    psi = torch.stack(psis, dim=0)

    # --- measure Z ---
    for i in range(nq):
        Zmat[:, i] = expval_Z_fast_batch(psi, i, nq)

    return Zmat





class VectorZLinear(nn.Module):
    """
    Z-vector quantum model with trainable linear readout.
    Trainer-compatible: forward() returns ONLY score.
    """
    def __init__(self, circ, spec, amp_enc=False):
        super().__init__()

        self.circ = circ
        self.spec = spec

        self.theta = nn.Parameter(
            torch.randn(spec.tlen, device=DEVICE, dtype=DTYPE)
        )


        #trainable readout (4 -> 1)
        self.w = nn.Parameter(
            torch.randn(spec.num_qubits, device=DEVICE, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros((), device=DEVICE, dtype=torch.float32)
        )

        self.amp_enc = AmpEnc(spec.num_qubits) if amp_enc else None


    


    def forward(self, xb):
        Zout = calcZvec(xb, self.theta, self.circ, self.spec, self.amp_enc)

        Zout = Zout.real.to(dtype=self.w.dtype)

        score = Zout @ self.w + self.b

        score = score.real

        return score
    
__all__ = ["VectorZLinear"]
