import torch
from torch.utils.data import DataLoader, TensorDataset

from TorchQML import Circuit, CircuitSpec, H, Tsym, Xsym, rx
from TorchQML.models import VectorZLinear
from TorchQML.training import TrainConfig, Trainer, batch_accuracy, softplus


def make_circuit():
    spec = CircuitSpec(num_qubits=2, xlen=2, tlen=2)
    x = Xsym()
    t = Tsym()

    circ = Circuit(num_qubits=2, specs=spec)
    circ.add_gates([H, H])
    circ.add_gates([rx(x[0]), rx(x[1])])
    circ.add_gates([rx(t[0]), rx(t[1])])
    return circ, spec


def main():
    torch.manual_seed(0)
    X = torch.tensor([[0.0, 0.1], [0.6, -0.2], [-0.3, 0.4], [0.9, 0.8]])
    y = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False)

    circ, spec = make_circuit()
    model = VectorZLinear(circ, spec)
    trainer = Trainer(model, softplus, batch_accuracy, TrainConfig(epochs=2, lr=0.01))
    history, best_state, best_val = trainer.fit(loader, loader)
    test_out = trainer.evaluate(loader, best_state)

    print({"best_val": best_val, "test_acc": test_out["acc"], "epochs": len(history["train_loss"])})


if __name__ == "__main__":
    main()
