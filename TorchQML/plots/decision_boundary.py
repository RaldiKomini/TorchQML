import numpy as np
import torch
import matplotlib.pyplot as plt
from TorchQML.core.config import DEVICE
from .bloch_sphere import states_per_batch, states_to_bloch



def class_means(x, y, theta, circ, measure_qubit: int = 0):
    """
    x: [B, 2^n_input_features]  (your classical inputs)
    y: [B] labels (0/1)
    theta: params
    circ: Circuit
    measure_qubit: which qubit Bloch vector is computed from (0..n-1)
    """
    states = states_per_batch(x, theta, circ)                  # [B, 2^n]
    vecs   = states_to_bloch(states, measure_qubit)            # [B, 3]

    y = y.long()
    mask0 = (y == 0)
    mask1 = (y == 1)

    # if one class is missing, avoid .mean() on empty
    if mask0.any():
        avg0 = vecs[mask0].mean(dim=0)
    else:
        avg0 = torch.zeros(3, device=x.device)

    if mask1.any():
        avg1 = vecs[mask1].mean(dim=0)
    else:
        avg1 = torch.zeros(3, device=x.device)

    return avg0, avg1

def classify_meansPerBatch(X, t, circ, avg0, avg1, measure_qubit: int = 0):
    """
    X: [B, d]
    t: params
    avg0, avg1: class mean Bloch vectors [3]
    """
    states = states_per_batch(X, t, circ)                      # [B, 2^n]
    vecs   = states_to_bloch(states, measure_qubit)            # [B, 3]

    # cosine-like similarity via dot product with means
    sim0 = (vecs * avg0).sum(dim=-1)   # [B]
    sim1 = (vecs * avg1).sum(dim=-1)   # [B]
    preds = (sim1 > sim0).long()
    return preds


def decision_boundary(X, y, theta, circ, title="Decision Boundary"):
    """
    X:      torch.Tensor [N, 2]  (data points)
    y:      torch.Tensor [N]     (labels 0/1)
    theta:  torch.Tensor [n_thetas] (trainable params)
    circ:   Circuit-like object with .apply_to(x=..., theta=theta)
    """

    X = X.to(DEVICE)
    y = y.to(DEVICE)

    # ---- 1. Compute class mean Bloch vectors ONCE (replaces ra, rb) ----
    avg0, avg1 = class_means(X, y, theta, circ)

    # ---- 2. Build mesh grid over the data range ----
    x_min = X[:, 0].min().item() - 0.2
    x_max = X[:, 0].max().item() + 0.2
    y_min = X[:, 1].min().item() - 0.2
    y_max = X[:, 1].max().item() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # ---- 3. Classify all grid points in one go ----
    grid_points = np.c_[xx.ravel(), yy.ravel()]              # [M, 2]
    grid_tensor = torch.from_numpy(grid_points).float().to(DEVICE)

    grid_preds = classify_meansPerBatch(grid_tensor, theta, circ, avg0, avg1)
    Z = grid_preds.view(xx.shape).detach().cpu().numpy()     # reshape to grid

    # ---- 4. Plot decision boundary + data points ----
    cmap = plt.cm.colors.ListedColormap(['red', 'blue'])

    plt.figure(figsize=(6, 6))
    contour = plt.contourf(
        xx, yy, Z,
        levels=[-0.5, 0.5, 1.5],
        alpha=0.4,
        cmap=cmap
    )

    # color bar
    cbar = plt.colorbar(contour)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Class 0', 'Class 1'])

    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    plt.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
                color='red', edgecolors='k', label='Class 0')
    plt.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
                color='blue', edgecolors='k', label='Class 1')

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.gca().set_aspect("equal", "box")
    plt.show()


def plot_data(X, y):
    X = X.cpu().numpy()
    y = y.cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='orange', label='class 0', alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='dodgerblue', label='class 1', alpha=0.7)

    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.title("Dataset")
    plt.show()
