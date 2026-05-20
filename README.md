# TorchQML

TorchQML is a PyTorch-based quantum machine learning library. It started as a
research workspace for variational quantum models, quantum kernels,
architecture search, knowledge distillation, and circuit synthesis. This public
version keeps the reusable parts clean and runnable, with the code split into
small modules instead of long experiment scripts.

The goal of the project is simple: make quantum ML experiments easy to read,
modify, test, and extend without hiding the mechanics behind a large framework.

This repository is the curated public version of a larger research codebase.
Additional unpublished ablations, high-budget searches, and early-stage ideas
are kept private until they are either published, rewritten, or no longer useful.

## What Is Included

- A state-vector circuit API built on PyTorch tensors.
- Fixed gates, symbolic gates, Hamiltonian-style gates, and amplitude encoding.
- Trainable model heads for Sum-Z, Vector-Z, Pauli readouts, and entanglement
  readouts.
- Training utilities for supervised binary experiments.
- Quantum fidelity kernels and precomputed-kernel QSVM helpers.
- Pattern-based quantum architecture search.
- Reusable knowledge-distillation pieces for teacher logits, FMNIST-style
  datasets, multiclass heads, search spaces, and multi-fidelity surrogate search.
- Circuit synthesis environments and search tools for Bell, GHZ, T, Toffoli,
  QFT, and related unitary targets.
- Unit tests and smoke tests that exercise the main code paths.
- Small examples in `examples/`.

## Package Layout

```text
TorchQML/
  core/          Circuit, CircuitSpec, symbols, runtime helpers, unitary simulator
  gates/         Fixed gates, Gate.CNOT, variable gates, Hamiltonian gates
  encoding/      Amplitude encoding
  models/        Sum-Z, Vector-Z, Pauli, Dirichlet/Hamiltonian, Spooky heads
  training/      Simple trainer, losses, metrics, run saving
  kernels/       Fidelity kernels, centered kernels, SVM helpers
  quantum_svm/   QSVM compatibility wrapper
  qas/           Pattern-based architecture search
  distillation/  Teacher logits, KD loops, search spaces, FMNIST utilities
  synthesis/     RL/search environments, synthesis baselines, unitary distillation
  states/        Reusable state and circuit templates
  datasets/      Dataset loading helpers
  plots/         Plotting helpers
tests/           Unit tests and smoke tests
examples/        Short runnable examples
```

## Installation

Recommended full install for this repository:

```powershell
pip install -e .[dev,experiments,remote-sensing]
```

If your shell treats brackets specially, quote the package spec:

```powershell
pip install -e ".[dev,experiments,remote-sensing]"
```

Minimal core install:

```powershell
pip install -e .
```

The minimal install is enough for the core circuit and model modules. The full
install is recommended for running the README examples, tests, datasets,
distillation utilities, synthesis scripts, and plotting code.

## Dependencies

TorchQML is developed with Python 3.10+.

Core dependencies:

- `torch`
- `numpy`

Development and tests:

- `pytest`

Experiment dependencies:

- `scikit-learn` for PCA, train/validation splits, Gaussian-process surrogate
  search, and precomputed-kernel SVMs.
- `pandas` for FashionMNIST CSV loading.
- `xgboost` for teacher models used in distillation experiments.
- `kagglehub` for locating cached FashionMNIST files.
- `gymnasium` for the circuit-synthesis environments.
- `stable-baselines3` for PPO synthesis experiments.
- `sb3-contrib` for MaskablePPO synthesis experiments.
- `torchvision` for image-folder datasets.
- `tifffile` for EuroSAT multispectral TIFF loading.
- `matplotlib` for experiment plotting.
- `plotly` for Bloch-sphere visualizations.

Basic circuits and model heads are built on PyTorch and NumPy. FMNIST
distillation, QSVMs, synthesis training, remote-sensing data helpers, and
plotting need the corresponding experiment dependencies above.

## Quick Start

```python
import torch

from TorchQML import Circuit, CircuitSpec, H, Tsym, Xsym, rx
from TorchQML.models import VectorZLinear

spec = CircuitSpec(num_qubits=2, xlen=2, tlen=2)
x = Xsym()
t = Tsym()

circ = Circuit(num_qubits=2, specs=spec)
circ.add_gates([H, H])
circ.add_gates([rx(x[0]), rx(x[1])])
circ.add_gates([rx(t[0]), rx(t[1])])

model = VectorZLinear(circ, spec)
scores = model(torch.randn(4, 2))
```

Run the same idea as a script:

```powershell
python examples/quickstart.py
```

## Model Families

`SumZModel` is the simplest baseline. It applies a circuit and sums the Z
expectations across qubits.

`VectorZLinear` keeps the vector of per-qubit Z expectations and learns a small
linear head. This is the main compact VQC model in the repo.

`VectorZDirHam` is for direct-rank and Hamiltonian-style experiments where a
model learns over slots or candidate operators.

`PauliHeadModel` and `PauliVectorModel` use X/Y/Z expectation values instead of
only Z. They are useful when Z-only readout is too restrictive.

`SpookyModel` uses per-qubit linear entropy as a learned entanglement readout.
It is included because it was a useful experimental direction, even when it is
not the default model.

The distillation package also includes multiclass Vector-Z and Pauli heads for
FMNIST-style student models.

## Experiments

The public repo keeps representative, reusable experiment code rather than raw
notebook dumps.

Architecture search lives in `TorchQML/qas/` and `TorchQML/distillation/`.
There are pattern searches, repeated-cell spaces, alternating layered spaces,
flexible reuploading spaces, broad amplitude-encoding spaces, and a
multi-fidelity Gaussian-process surrogate.

Knowledge distillation lives in `TorchQML/distillation/`. It includes binary and
multiclass KD losses, teacher-logit loaders, XGBoost teacher helpers,
FMNIST-style data builders, teacher-kernel alignment, and direct-rank circuit
builders.

Circuit synthesis lives in `TorchQML/synthesis/`. It includes random baselines,
PPO-compatible environments, behavior/probe-state rewards, meet-in-the-middle
search, reverse-guided search, Toffoli variants, and unitary-distillation tools.

Small examples:

```powershell
python examples/distillation_toy.py
python examples/qsvm_kernel.py
python examples/synthesis_smoke.py
```

These are smoke examples, not paper-quality benchmark runs.

## Optimizations

The code is written to stay readable, but several slow paths from the early
experiments were replaced:

- Batched circuit execution through `Circuit.apply_to`.
- Cached dense matrices for fixed circuit layers.
- Cached `|0...0>` initial states.
- Batch-aware Kronecker products for symbolic gates.
- Cached CNOT and unitary-simulator expansions.
- Cached Z-sign matrices for expectation values.
- Batched amplitude encoding.
- Blocked fidelity-kernel construction for train/test SVM matrices.
- Action masks, inverse-action blocking, and cycle pruning in synthesis.

The important point is that the speedups are visible in normal Python code.
There is no hidden backend that makes the library hard to inspect.

## Tests

```powershell
python -m pytest tests -q
```

The tests cover public imports, gates, CNOT conventions, core circuit execution,
model heads, training, kernels, QSVM helpers, distillation utilities, synthesis
environments, meet-in-the-middle search, and unitary-compilation smoke paths.

## How To Read The Code

A practical reading order:

1. `TorchQML/core/` and `TorchQML/gates/` for circuits and gates.
2. `TorchQML/encoding/` for amplitude encoding.
3. `TorchQML/models/` for the model heads.
4. `TorchQML/training/` for the small supervised trainer.
5. `TorchQML/kernels/` and `TorchQML/quantum_svm/` for kernel methods.
6. `TorchQML/qas/` for compact architecture search.
7. `TorchQML/distillation/` for teacher/student experiments and search spaces.
8. `TorchQML/synthesis/` for circuit synthesis and unitary distillation.
9. `tests/` for the fastest executable examples of supported behavior.

New experiment code should keep reusable logic inside the package and leave the
entry-point scripts short.

## Notes On Scope

This repository is meant to be useful and readable. It is not a dump of every
run, every plot, or every unpublished idea. The private workspace contains more
search results, failed directions, incomplete ablations, and unpublished
branches. The public code here is the part that is clean enough for other people
to inspect, run, and build on.
