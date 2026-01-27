import torch
from TorchQML.core.config import DEVICE, DTYPE


#Amplitude Encoding
class AmpEnc:
    def __init__(self, nq):
        self.num_qubits = nq
        self.dim = 1 << self.num_qubits #vector dimension

    def __call__(self, x):

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device = DEVICE)
        if x.dim() != 1:
            raise ValueError("X has wrong dim")
        
        if x.shape[0] >= self.dim: #encodes only the first self.dim features
            x = x[:self.dim]

        else:#if features are less it pads with 0s
            pad = torch.full((self.dim - x.shape[0], ), float(0.0), device= DEVICE, dtype=x.dtype)
            x = torch.cat([x, pad], dim = 0)

        if not torch.is_complex(x):
            x = x.to(torch.complex64)

        norm = torch.linalg.vector_norm(x)
        psi = x/(norm + 1e-12)
        return psi.to(dtype = DTYPE)
        

__all__ = ["AmpEnc"]