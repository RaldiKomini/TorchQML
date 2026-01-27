import numpy as np
import torch
import plotly.graph_objects as go
from TorchQML.core.config import DEVICE
from TorchQML.gates.gate import I, X, Z, Y
import math

def embed_single_qubit_op(op_1q: torch.Tensor,
                          num_qubits: int,
                          target: int,
                          device=DEVICE) -> torch.Tensor:
    """
    Build an n-qubit operator that acts as `op_1q` on `target` qubit
    and as identity on all others.
    target: 0..num_qubits-1 (assume qubit 0 is 'leftmost' in kron order).
    """

    if not (0 <= target < num_qubits):
        raise ValueError("target out of range")

    op = None
    for q in reversed(range(num_qubits)):
        factor = op_1q if q == target else I.matrix.to(device)
        op = factor if op is None else torch.kron(op, factor)

    return op.to(device)

def states_per_batch(xb, theta, circ):
    states = []
    for x in xb:
        psi = circ.apply_to(x= x, theta = theta)
        states.append(psi)

    return torch.stack(states, dim = 0)


def states_to_bloch(states: torch.Tensor,
                  measure_qubit: int = 0) -> torch.Tensor:
    """
    states: [B, 2^n] complex amplitudes
    returns: [B, 3] Bloch vector for `measure_qubit`.
    """
    B, dim = states.shape
    # infer number of qubits from dimension
    num_qubits = int(math.log2(dim))
    assert 2 ** num_qubits == dim, "states last dim must be a power of 2"

    # build n-qubit X, Y, Z acting on `measure_qubit`
    Xn = embed_single_qubit_op(X.matrix, num_qubits, measure_qubit, device=states.device)
    Yn = embed_single_qubit_op(Y.matrix, num_qubits, measure_qubit, device=states.device)
    Zn = embed_single_qubit_op(Z.matrix, num_qubits, measure_qubit, device=states.device)

    conjs = torch.conj(states)

    # <ψ|O|ψ> = sum_i ψ*_i (Oψ)_i
    oX = states @ Xn.T
    eX = (conjs * oX).sum(dim=-1).real

    oY = states @ Yn.T
    eY = (conjs * oY).sum(dim=-1).real

    oZ = states @ Zn.T
    eZ = (conjs * oZ).sum(dim=-1).real

    vecs = torch.stack([eX, eY, eZ], dim=-1)
    return vecs


def plot_bloch_sphere(
    X,
    y,
    theta,
    circ,
    colors_list=("red", "blue"),
    measure_qubit=0,
    filename=None,
    state_params=None,
):
    """
    X:      torch.Tensor [N, ...]   input features
    y:      torch.Tensor [N]        labels in {0,1}
    theta:  torch.Tensor            trainable circuit parameters
    circ:   Circuit-like object with .apply_to(x=..., theta=...)
    """

    X = X.to(DEVICE)
    y = y.to(DEVICE)

    # ---- 1. Compute states and Bloch vectors ----
    with torch.no_grad():
        states = states_per_batch(X, theta, circ)          
        vecs   = states_to_bloch(states, measure_qubit)   

    vecs_np = vecs.cpu().numpy()
    y_np    = y.cpu().numpy()

    # ---- 2. Create sphere mesh ----
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale="Viridis",
        showscale=False,
        opacity=0.3,
    )

    # ---- 3. Plot Bloch vectors ----
    for cls_idx, color in zip([0, 1], colors_list):
        cls_vecs = vecs_np[y_np == cls_idx]
        for bx, by, bz in cls_vecs:
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, bx],
                    y=[0.0, by],
                    z=[0.0, bz],
                    mode="lines+markers",
                    marker=dict(size=4, color=color),
                    line=dict(color=color, width=3),
                    showlegend=False,
                )
            )

    # ---- 4. Mean Bloch vectors ----
    if (y_np == 0).any():
        v0 = vecs_np[y_np == 0].mean(axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, v0[0]],
                y=[0.0, v0[1]],
                z=[0.0, v0[2]],
                mode="lines+markers",
                marker=dict(size=8, color="green"),
                line=dict(color="green", width=8),
                name="Mean class 0",
            )
        )

    if (y_np == 1).any():
        v1 = vecs_np[y_np == 1].mean(axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, v1[0]],
                y=[0.0, v1[1]],
                z=[0.0, v1[2]],
                mode="lines+markers",
                marker=dict(size=8, color="yellow"),
                line=dict(color="yellow", width=8),
                name="Mean class 1",
            )
        )

    # ---- 5. Cosmetics ----
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5]),
            yaxis=dict(range=[-1.5, 1.5]),
            zaxis=dict(range=[-1.5, 1.5]),
            aspectmode="cube",
        )
    )

    if state_params is not None:
        fig.add_annotation(
            x=0, y=0, z=1.2,
            text=", ".join(f"{k}: {v}" for k, v in state_params.items()),
            showarrow=False,
        )

    if filename is not None:
        import plotly.offline as pyo
        pyo.plot(fig, filename=filename, auto_open=False)

    fig.show()
    return fig


__all__ = ["plot_bloch_sphere"]