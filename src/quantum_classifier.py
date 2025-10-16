"""
Variational Quantum Classifier (VQC) using Qiskit Machine Learning.
"""

import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC

from .data_utils import create_meshgrid


def build_feature_map(n_qubits=2, reps=2):
    """
    Build a ZZFeatureMap for encoding classical data into quantum states.

    The ZZFeatureMap applies H gates, then single-qubit Z rotations using
    the input features, followed by ZZ entangling interactions. The 'reps'
    parameter controls how many times this is repeated.
    """
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement='linear'
    )
    return feature_map


def build_ansatz(n_qubits=2, reps=3):
    """
    Build a RealAmplitudes ansatz — Ry rotations + CNOT entanglement.
    This is a pretty standard choice for variational classifiers.
    """
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=reps,
        entanglement='full'
    )
    return ansatz


def train_vqc(X_train, y_train, n_qubits=2, feature_reps=2, ansatz_reps=3,
              maxiter=100, seed=42):
    """
    Train a Variational Quantum Classifier.

    Returns the trained model and a dict with training info.
    """
    feature_map = build_feature_map(n_qubits, feature_reps)
    ansatz = build_ansatz(n_qubits, ansatz_reps)

    optimizer = COBYLA(maxiter=maxiter)

    # track the objective during training
    obj_values = []
    def callback(weights, obj_value):
        obj_values.append(obj_value)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    # fit the classifier
    vqc.fit(X_train, y_train)

    training_info = {
        'objective_values': obj_values,
        'feature_map': feature_map,
        'ansatz': ansatz,
        'n_params': ansatz.num_parameters,
    }

    return vqc, training_info


def predict_grid(model, X, resolution=50):
    """
    Predict class labels on a meshgrid for decision boundary visualization.
    """
    xx, yy, grid_points = create_meshgrid(X, resolution)
    predictions = model.predict(grid_points)
    Z = predictions.reshape(xx.shape)
    return xx, yy, Z
