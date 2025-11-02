import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from .data_utils import create_meshgrid


def train_classical_svm(X_train, y_train, kernel='rbf', C=1.0):
    svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm


def evaluate_classical_svm(svm, X_test, y_test):
    predictions = svm.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    # print(f"accuracy: {acc}")
    return acc, predictions


def predict_grid_classical(model, X, resolution=50):
    xx, yy, grid_points = create_meshgrid(X, resolution)
    predictions = model.predict(grid_points)
    Z = predictions.reshape(xx.shape)
    return xx, yy, Z
