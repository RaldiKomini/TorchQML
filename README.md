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
- A simple **Variational Quantum Classifier (QVC)** for quick experiments

---

## 🚀 Quick Example

```python
from TorchQML.circuit import Circuit
from TorchQML.gates.gate import H, Z

c = Circuit(num_qubits=1)
c.add_gates([H])
c.add_gates([Z])

state = c.apply_to()
print(state)

```


Installation:
git clone https://github.com/RaldiKomini/TorchQML.git
cd TorchQML