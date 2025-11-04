import torch
from torch.optim import Adam

from qml_lib.circuits.circuit import Circuit
from qml_lib.gates.gate import rx, ry, rz, CNOT
from qml_lib.wrappers.gatewrapper import VGate
from qml_lib.measurements.expectation_Z import expectation_Z
from qml_lib.config import DEVICE
from qml_lib.data_gen.moon import make_moons_torch

n = 2
c = Circuit(n)
c.add_gates([VGate(rx, ("x",0), 0), VGate(ry, ("x",1), 1)])
c.add_full(CNOT)
c.add_gates([VGate(rz, ("x",0), 0), VGate(rx, ("theta",0), 1)])
c.add_gates([VGate(ry, ("theta",1), 0), VGate(rz, ("theta",2), 1)])

theta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=DEVICE))
data  = make_moons_torch(n_samples=1000, noise=0.2, to_angle="pi")
Xtr, ytr = data["X_train"], data["y_train"]   # Xtr: [N, 2]

opt = Adam([theta], lr=0.01)
target = torch.tensor(0.6, device=DEVICE)

for step in range(200):
    opt.zero_grad()

    # pick one sample (or iterate a small batch with a for-loop)
    i = torch.randint(0, Xtr.shape[0], (1,)).item()
    x_sample = Xtr[i]             # shape [2] -> scalars at indices 0 and 1

    psi = c.apply_to(x=x_sample, theta=theta)        # <-- pass ONE sample
    out = expectation_Z(psi, index=0, n_qubits=n)    # scalar
    loss = (out - target).pow(2)
    loss.backward()
    opt.step()

    if step % 50 == 0:
        print(f"step {step:03d} out={float(out):+.4f} loss={float(loss):.6f}")

print("final out:", float(out))
print("theta:", theta.detach().cpu().numpy())
