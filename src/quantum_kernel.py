import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

from .data_utils import create_meshgrid


def build_quantum_kernel(n_qubits=2, reps=2):
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement='linear'
    )
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return kernel, feature_map


def train_quantum_kernel_svm(X_train, y_train, n_qubits=2, reps=2):
    kernel, feature_map = build_quantum_kernel(n_qubits, reps)

    print("computing quantum kernel matrix (this takes a moment)...")
    kernel_matrix_train = kernel.evaluate(X_train)
    print(f"kernel matrix shape: {kernel_matrix_train.shape}")
    # print(f"kernel matrix sample:\n{kernel_matrix_train[:3,:3]}")

    svm = SVC(kernel='precomputed', probability=True)
    svm.fit(kernel_matrix_train, y_train)

    return svm, kernel, feature_map


def evaluate_quantum_kernel_svm(svm, kernel, X_train, X_test, y_test):
    kernel_matrix_test = kernel.evaluate(X_test, X_train)
    accuracy = svm.score(kernel_matrix_test, y_test)
    predictions = svm.predict(kernel_matrix_test)
    return accuracy, predictions


def predict_grid_quantum_kernel(svm, kernel, X_train, X, resolution=30):
    xx, yy, grid_points = create_meshgrid(X, resolution)

    print("computing kernel for grid (this may take a while)...")
    kernel_matrix_grid = kernel.evaluate(grid_points, X_train)
    predictions = svm.predict(kernel_matrix_grid)
    Z = predictions.reshape(xx.shape)

    return xx, yy, Z
