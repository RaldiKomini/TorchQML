import torch
from qml_lib.measurements.expectation_Z import expectation_Z

class QVCClassifier(torch.nn.Module):
    def __init__(self, circuit, n_qubits: int, n_thetas: int):
        super().__init__()
        self.circuit = circuit
        self.n_qubits = n_qubits
        self.theta = torch.nn.Parameter(torch.zeros(n_thetas))
        self.head_w = torch.nn.Parameter(torch.tensor(1.0))
        self.head_b = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):  # x: (B, 2)
        batch_size = x.shape[0]
        z_list = []
        for i in range(batch_size):
            psi_i = self.circuit.apply_to(x=x[i], theta=self.theta)
            z0_i = expectation_Z(psi_i, index=0, n_qubits=self.n_qubits)
            z_list.append(z0_i)

        z = torch.stack(z_list)           # shape (B,)
        logits = self.head_w * z + self.head_b  # broadcasts → (B,)
        return logits