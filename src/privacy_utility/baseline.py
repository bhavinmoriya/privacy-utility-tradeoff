"""Plaintext logistic regression baseline (no privacy mechanism), trained by
plain gradient descent so that dp.py's DP-SGD variant is a minimal, legible
diff away from the non-private version rather than a different codepath
entirely.
"""

from __future__ import annotations

import time

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 50,
    lr: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Plain (non-private) full-batch gradient descent logistic regression.

    Returns (weights, bias).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0
    for _ in range(n_epochs):
        z = X @ w + b
        p = sigmoid(z)
        grad_w = X.T @ (p - y) / n
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    preds = predict(X, w, b) >= 0.5
    return float(np.mean(preds == y))


def run_baseline(X_train, y_train, X_test, y_test, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    w, b = train_logreg(X_train, y_train, seed=seed)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = accuracy(X_test, y_test, w, b)
    n_test = X_test.shape[0]
    inference_time_per_sample_ms = (time.perf_counter() - t0) / n_test * 1000

    return {
        "mechanism": "plaintext",
        "epsilon": float("inf"),
        "poly_modulus_degree": None,
        "accuracy": acc,
        "train_time_s": train_time,
        "inference_ms_per_sample": inference_time_per_sample_ms,
        "protects_against": "nothing (baseline)",
    }
