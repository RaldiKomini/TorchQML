import torch
from torch.utils.data import DataLoader, TensorDataset

from TorchQML.distillation import fit_binary_kd


def main():
    torch.manual_seed(0)
    X = torch.tensor(
        [
            [-1.0, 0.0, 0.5],
            [-0.5, 0.3, 0.2],
            [0.6, -0.4, 0.1],
            [1.0, 0.2, -0.2],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    teacher_logits = torch.tensor([-2.0, -1.0, 1.5, 2.0], dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y, teacher_logits), batch_size=2, shuffle=False)

    student = torch.nn.Linear(3, 1)
    _, history = fit_binary_kd(
        student,
        loader,
        loader,
        torch.device("cpu"),
        epochs=2,
        lr=0.05,
        alpha=0.5,
        temperature=2.0,
    )
    print(history[-1])


if __name__ == "__main__":
    main()
