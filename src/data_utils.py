"""
Dataset loading and preprocessing for quantum classification experiments.
"""

import numpy as np
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_moons_dataset(n_samples=200, noise=0.15, test_size=0.25, seed=42):
    """
    Generate the make_moons dataset — two interleaving half circles.
    Good for testing nonlinear classifiers.

    Features are scaled to [0, pi] for the quantum feature map.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # scale to [0, pi] — this is important for the quantum feature map
    # since we're using it as rotation angles
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_iris_2d(test_size=0.25, seed=42):
    """
    Load the Iris dataset, but only use 2 features and 2 classes
    to keep it simple for quantum circuits (2 qubits).

    Uses sepal length and sepal width, classes 0 and 1 only.
    """
    iris = load_iris()
    X = iris.data[:, :2]  # just the first two features
    y = iris.target

    # filter to binary classification (setosa vs versicolor)
    mask = y < 2
    X = X[mask]
    y = y[mask]

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_meshgrid(X, resolution=50):
    """
    Create a meshgrid for plotting decision boundaries.
    Returns the grid points as a 2D array suitable for prediction.
    """
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid_points
