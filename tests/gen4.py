import torch
from qml_lib.QVCC.qvc1 import QVCClassifier
from qml_lib.data_gen.moon import make_moons_torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogits
from qml_lib.config import DEVICE, DTYPE
from qml_lib.states.states1q import stateg
from sklearn.datasets import make_circles

Xtr, ytr = make_circles(n_samples=1000, noise=0.2, factor=0.4)
#data  = make_moons_torch(n_samples=1000, noise=0.2, to_angle="pi")
#Xtr, ytr = data["X_train"], data["y_train"]   # Xtr: [N, 2], ytr: [N]
Xtr = torch.from_numpy(Xtr).float()
ytr = torch.from_numpy(ytr).long()
t = torch.zeros(7)
c1 = stateg()

model = QVCClassifier(c1, n_qubits=1, n_thetas=7).to(DEVICE)

opt = Adam(model.parameters(), lr=0.01)

from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogits

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogits

Xtr = Xtr.to(DEVICE)
ytr = ytr.to(DEVICE).float()

dataset = TensorDataset(Xtr, ytr)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)  # try 16 or 32

num_epochs  = 20
global_step = 0

for epoch in range(num_epochs):
    for xb, yb in loader:           # xb: [B, 2], yb: [B]
        logits = model(xb)          # still loops over B inside forward
        logits = logits.view(-1)
        loss   = BCEWithLogits(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if global_step % 100 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                acc   = (preds == yb).float().mean().item()
            print(f"step {global_step:05d} loss={loss.item():.4f} acc={acc:.3f}")

        global_step += 1
