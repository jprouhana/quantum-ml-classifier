# Quantum Machine Learning Classifier

A variational quantum classifier (VQC) and quantum kernel SVM implementation using Qiskit Machine Learning. Compares quantum and classical approaches on standard ML datasets.

Built as part of independent study work on quantum-classical hybrid optimization.

## Background

**Variational Quantum Classification** uses parameterized quantum circuits as machine learning models. The idea is:

1. Map classical data into quantum states using a **feature map** (data encoding circuit)
2. Apply a **variational ansatz** (trainable circuit) with parameters optimized via classical gradient descent
3. Measure the output qubits to get class predictions

**Quantum Kernel Methods** take a different approach — instead of training a quantum circuit, they use the quantum feature map to compute kernel values (similarity measures) between data points, then feed these into a classical SVM.

The big question is whether quantum feature maps can capture structure in data that classical kernels can't. For the toy datasets used here, classical methods work just fine — but this serves as a proof of concept for the quantum pipeline.

## Project Structure

```
quantum-ml-classifier/
├── src/
│   ├── data_utils.py              # Dataset loading and preprocessing
│   ├── quantum_classifier.py      # VQC implementation
│   ├── quantum_kernel.py          # Quantum kernel SVM
│   └── classical_baseline.py      # Classical SVM baseline
├── notebooks/
│   └── quantum_classification.ipynb  # Full analysis notebook
├── results/                        # Saved figures
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/quantum-ml-classifier.git
cd quantum-ml-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.quantum_classifier import train_vqc
from src.data_utils import load_moons_dataset

X_train, X_test, y_train, y_test = load_moons_dataset(n_samples=200)
model, history = train_vqc(X_train, y_train, n_qubits=2, reps=2)
```

### Jupyter Notebook

The main analysis is in `notebooks/quantum_classification.ipynb`:

```bash
jupyter notebook notebooks/quantum_classification.ipynb
```

## Approach

### Feature Map

We use a `ZZFeatureMap` which encodes classical data $\mathbf{x}$ into quantum states using:

$$U_\Phi(\mathbf{x}) = \exp\left(i \sum_{j,k} \phi_{jk}(\mathbf{x}) Z_j Z_k\right) \exp\left(i \sum_j x_j Z_j\right) H^{\otimes n}$$

This creates entanglement between qubits based on the input features, which lets the quantum model capture nonlinear relationships.

### Ansatz

For the VQC, we use the `RealAmplitudes` ansatz — alternating layers of Ry rotations and CNOT entangling gates. The parameters are optimized using COBYLA.

### Quantum Kernel

For the kernel SVM, we compute the kernel matrix:

$$K_{ij} = |\langle \phi(\mathbf{x}_i) | \phi(\mathbf{x}_j) \rangle|^2$$

This measures the overlap between quantum states, which we then use as the kernel in a classical SVM (from sklearn).

## Results

| Method | Accuracy (make_moons) | Accuracy (Iris 2D) |
|--------|----------------------|---------------------|
| Classical SVM (RBF) | 0.96 | 0.97 |
| Quantum Kernel SVM | 0.94 | 0.95 |
| Variational Quantum Classifier | 0.91 | 0.93 |

The classical SVM slightly outperforms the quantum methods on these small datasets, which is expected — quantum advantage is theorized for high-dimensional, structured data that classical kernels can't easily handle.

## References

1. Havlicek, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567(7747), 209-212.
2. Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
3. Qiskit Machine Learning Documentation: [https://qiskit-community.github.io/qiskit-machine-learning/](https://qiskit-community.github.io/qiskit-machine-learning/)

## License

MIT License — see [LICENSE](LICENSE) for details.
