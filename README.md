# TorchQML

**TorchQML** is a small, simple, and readable Quantum Machine Learning library built on top of **PyTorch**.  
It’s designed for experimentation, learning, and building variational quantum models without the weight and complexity of large frameworks.

The goal is to give you a clear, minimal codebase where you can understand every part of the pipeline:
quantum states → gates → circuits → training.

---

## ✨ What TorchQML offers

- **Clean and simple Circuit class** for building quantum circuits step-by-step  
- **Fixed gates** (H, X, Y, Z, CNOT…) and **parameterized gates**
- **Full PyTorch autograd support**  
- **Lazy evaluation** → circuits stay *symbolic* until you call `apply_to`  
  - no big matrices are built early  
  - no wasted memory  
  - everything stays lightweight until execution  
- Easy integration into **PyTorch training loops**
- **Small, readable codebase** that you can understand and extend quickly
- A simple **Variational Quantum Classifier (VQC)** for quick experiments

---

## 📌 Upcoming Features

These features are already prepared and will be integrated soon:

- Custom loss functions and classification metrics
- Decision boundary visualization tools
- Bloch sphere visualization for single-qubit states
- Cleaner training utilities for VQC models
- Additional example notebooks and demos

More improvements will be added over time as the library grows.

---




## Project Structure (Public API)

```text
TorchQML/
├── core/        # circuits, symbols, configuration
├── encoding/    # data encoding (e.g. amplitude encoding)
├── gates/       # quantum gates
├── models/      # variational quantum models
├── states/      # predefined circuit architectures
├── training/    # trainer, losses, metrics
├── datasets/    # EuroSAT utilities
├── plots/       # visualization tools
---

## Planned / Ongoing Work

The following features are actively explored but not yet fully exposed:

- Quantum SVM (QSVM)
- Quantum architecture search
- Additional variational circuit templates

---

## Installation

git clone https://github.com/RaldiKomini/TorchQML.git  
cd TorchQML

TorchQML depends primarily on PyTorch, NumPy, and standard scientific Python tooling.

---

## Notes

This library reflects an ongoing research and learning process.  
The focus is on clarity, correctness, and experimentation, rather than strict API stability.

The intended audience includes researchers, students, and engineers interested in quantum machine learning internals.

Feedback, discussion, and critical review are welcome.



## 🚀 Quick Example

```python
from TorchQML import Circuit,  H, Z

c = Circuit(num_qubits=1)
c.add_gates([H])
c.add_gates([Z])

state = c.apply_to()
print(state)

```


Installation:
git clone https://github.com/RaldiKomini/TorchQML.git
cd TorchQML
