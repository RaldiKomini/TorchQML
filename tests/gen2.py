import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogits

from qml_lib.circuits.circuit import Circuit
from qml_lib.gates.gate import rx, ry, rz, CNOT, H, I
from qml_lib.wrappers.gatewrapper import VGate
from qml_lib.measurements.expectation_Z import expectation_Z
from qml_lib.config import DEVICE, DTYPE
from qml_lib.data_gen.moon import make_moons_torch
from qml_lib.states.states1q import stateg

# ----- build circuit -----

# quantum params
theta = torch.nn.Parameter(torch.zeros(7, dtype=torch.float32, device=DEVICE))
# simple classical head: logit = w * <Z0> + b
head_w = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=DEVICE))
head_b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=DEVICE))

# data
data  = make_moons_torch(n_samples=1000, noise=0.2, to_angle="pi")
Xtr, ytr = data["X_train"], data["y_train"]   # Xtr: [N, 2], ytr: [N]

opt = Adam([theta, head_w, head_b], lr=0.01)
batch_size = 16

def evaluate_accuracy(theta, w, b, X, y, n_qubits: int) -> float:
    correct = 0
    N = X.shape[0]
    for i in range(N):
        x = X[i]
        c = stateg(x, theta)
        psi = c.apply_to()
        z0 = expectation_Z(psi, index=0, n_qubits=n_qubits)   # [-1,1]
        logit = w * z0 + b
        prob1 = torch.sigmoid(logit)
        pred = (prob1 > 0.5).long()
        if pred.item() == y[i].item():
            correct += 1
    return correct / N

# --- before training ---
#acc_before = evaluate_accuracy(c, theta, head_w, head_b, Xtr, ytr, n_qubits=n)
#print(f"accuracy before training: {acc_before:.3f}")
n = 1
# ----- training -----
steps = 1000
for step in range(steps):
    opt.zero_grad()
    loss = 0.0

    # mini-batch
    idx = torch.randint(0, Xtr.shape[0], (batch_size,), device=DEVICE)
    for i in idx:
        x = Xtr[i]
        y = ytr[i].float()

        c = stateg(x, theta)
        psi = c.apply_to()
        z0 = expectation_Z(psi, index=0, n_qubits=n)
        logit = head_w * z0 + head_b

        loss = loss + BCEWithLogits(logit, y)

    loss = loss / batch_size
    loss.backward()
    opt.step()

    if step % 100 == 0:
        acc = evaluate_accuracy(theta, head_w, head_b, Xtr, ytr, n_qubits=n)
        print(f"step {step:04d} loss={float(loss):.4f}  acc={acc:.3f}  w={float(head_w):+.3f} b={float(head_b):+.3f}")

# --- after training ---
#acc_after = evaluate_accuracy(c, theta, head_w, head_b, Xtr, ytr, n_qubits=n)
print("theta:", theta.detach().cpu().numpy())
print("head w,b:", float(head_w), float(head_b))
#print(f"accuracy after training: {acc_after:.3f}")
