import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC

from .data_utils import create_meshgrid


def build_feature_map(n_qubits=2, reps=2):
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement='linear'
    )
    return feature_map


def build_ansatz(n_qubits=2, reps=3):
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=reps,
        entanglement='full'
    )
    return ansatz


def train_vqc(X_train, y_train, n_qubits=2, feature_reps=2, ansatz_reps=3,
              maxiter=100, seed=42):
    feature_map = build_feature_map(n_qubits, feature_reps)
    ansatz = build_ansatz(n_qubits, ansatz_reps)
    optimizer = COBYLA(maxiter=maxiter)

    # this took forever to get working lol
    obj_values = []
    def callback(weights, obj_value):
        obj_values.append(obj_value)
        # print(f"iter {len(obj_values)}: obj={obj_value:.4f}")

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    vqc.fit(X_train, y_train)
    print(f"done training, {len(obj_values)} iterations")

    return vqc, obj_values


def predict_grid(model, X, resolution=50):
    xx, yy, grid_points = create_meshgrid(X, resolution)
    predictions = model.predict(grid_points)
    Z = predictions.reshape(xx.shape)
    return xx, yy, Z
