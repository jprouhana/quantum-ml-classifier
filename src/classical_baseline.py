"""
Classical SVM baselines for comparison against quantum classifiers.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from .data_utils import create_meshgrid


def train_classical_svm(X_train, y_train, kernel='rbf', C=1.0):
    """
    Train a standard sklearn SVM. This is our baseline to beat (or match).
    """
    svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm


def evaluate_classical_svm(svm, X_test, y_test):
    """Evaluate and return accuracy + predictions."""
    predictions = svm.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions


def predict_grid_classical(model, X, resolution=50):
    """Predict on meshgrid for decision boundary visualization."""
    xx, yy, grid_points = create_meshgrid(X, resolution)
    predictions = model.predict(grid_points)
    Z = predictions.reshape(xx.shape)
    return xx, yy, Z
