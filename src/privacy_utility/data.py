"""Dataset loading. Uses scikit-learn's bundled breast cancer dataset (no
network access required) as a standard, reproducible binary classification
benchmark for comparing privacy mechanisms.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_split(test_size: float = 0.3, seed: int = 0):
    """Load and standardize the breast cancer dataset, split train/test.

    Returns (X_train, X_test, y_train, y_test) as numpy arrays. Features are
    standardized (mean 0, std 1) using statistics from the training split
    only.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def feature_names() -> list[str]:
    return list(load_breast_cancer().feature_names)
