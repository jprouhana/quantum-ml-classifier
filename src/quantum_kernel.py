"""
Quantum Kernel SVM — uses a quantum feature map to compute kernel values,
then trains a classical SVM on top.
"""

import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

from .data_utils import create_meshgrid


def build_quantum_kernel(n_qubits=2, reps=2):
    """
    Build a quantum kernel using ZZFeatureMap.

    The kernel value between two data points x_i and x_j is the
    fidelity (state overlap squared) between their encoded quantum states:
        K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
    """
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement='linear'
    )

    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return kernel, feature_map


def train_quantum_kernel_svm(X_train, y_train, n_qubits=2, reps=2):
    """
    Train an SVM with a quantum kernel.

    Steps:
    1. Build the quantum kernel (feature map + fidelity computation)
    2. Compute the kernel matrix on training data
    3. Fit a classical SVM using this precomputed kernel
    """
    kernel, feature_map = build_quantum_kernel(n_qubits, reps)

    # compute the training kernel matrix
    # this is the expensive part — it runs a circuit for every pair
    print("Computing quantum kernel matrix (this takes a moment)...")
    kernel_matrix_train = kernel.evaluate(X_train)
    print(f"Kernel matrix shape: {kernel_matrix_train.shape}")

    # fit classical SVM on the quantum kernel
    svm = SVC(kernel='precomputed', probability=True)
    svm.fit(kernel_matrix_train, y_train)

    return svm, kernel, feature_map


def evaluate_quantum_kernel_svm(svm, kernel, X_train, X_test, y_test):
    """
    Evaluate the quantum kernel SVM on test data.
    Need to compute kernel between test and train points.
    """
    kernel_matrix_test = kernel.evaluate(X_test, X_train)
    accuracy = svm.score(kernel_matrix_test, y_test)
    predictions = svm.predict(kernel_matrix_test)
    return accuracy, predictions


def predict_grid_quantum_kernel(svm, kernel, X_train, X, resolution=30):
    """
    Predict on a grid for decision boundary plots.
    Using lower resolution since kernel evaluation is slow.
    """
    xx, yy, grid_points = create_meshgrid(X, resolution)

    # compute kernel between grid points and training data
    print("Computing kernel for grid (this may take a while)...")
    kernel_matrix_grid = kernel.evaluate(grid_points, X_train)
    predictions = svm.predict(kernel_matrix_grid)
    Z = predictions.reshape(xx.shape)

    return xx, yy, Z
