"""Differentially private logistic regression via DP-SGD, implemented from
scratch: per-example gradient clipping + calibrated Gaussian noise, with a
simple (epsilon, delta)-DP accountant.

This is deliberately a *simplified* accountant (per-step Gaussian mechanism
+ basic composition), not a production Renyi-DP accountant like Opacus's
moments accountant, which would give a tighter epsilon for the same noise.
It is accurate enough to produce a correct, defensible qualitative
privacy-utility tradeoff curve, which is the point of this benchmark.
"""

from __future__ import annotations

import time

import numpy as np

from privacy_utility.baseline import accuracy, sigmoid


def gaussian_sigma_for_epsilon(
    epsilon_per_step: float, delta: float, clip_norm: float
) -> float:
    """Calibrate Gaussian noise std for a single step to satisfy
    (epsilon_per_step, delta)-DP, using the standard analytic Gaussian
    mechanism bound: sigma >= clip_norm * sqrt(2 * ln(1.25/delta)) / epsilon.
    """
    return clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon_per_step


def total_epsilon_basic_composition(epsilon_per_step: float, n_steps: int) -> float:
    """Basic (non-tight) composition: total epsilon = n_steps * epsilon_per_step.

    This is a conservative (loose) upper bound; a moments/RDP accountant
    would report a smaller total epsilon for the same noise level. Kept
    intentionally simple and clearly labeled for this benchmark.
    """
    return epsilon_per_step * n_steps


def train_dp_sgd(
    X: np.ndarray,
    y: np.ndarray,
    target_epsilon: float,
    delta: float = 1e-5,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.5,
    clip_norm: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Train logistic regression with per-example gradient clipping and
    Gaussian noise (DP-SGD), targeting a given total epsilon via basic
    composition over all minibatch steps.

    Returns (weights, bias, achieved_epsilon).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0

    n_batches_per_epoch = max(1, n // batch_size)
    n_steps = n_epochs * n_batches_per_epoch
    epsilon_per_step = target_epsilon / n_steps
    sigma = gaussian_sigma_for_epsilon(epsilon_per_step, delta, clip_norm)

    idx = np.arange(n)
    for _epoch in range(n_epochs):
        rng.shuffle(idx)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            bs = len(batch_idx)

            z = Xb @ w + b
            p = sigmoid(z)
            # Per-example gradients (bs, d) for the weight term
            per_example_grad_w = Xb * (p - yb)[:, None]
            per_example_grad_b = (p - yb)

            # Clip each per-example gradient (including bias) to clip_norm
            grad_norms = np.sqrt(
                np.sum(per_example_grad_w**2, axis=1) + per_example_grad_b**2
            )
            clip_factor = np.minimum(1.0, clip_norm / (grad_norms + 1e-12))
            clipped_grad_w = per_example_grad_w * clip_factor[:, None]
            clipped_grad_b = per_example_grad_b * clip_factor

            sum_grad_w = clipped_grad_w.sum(axis=0)
            sum_grad_b = clipped_grad_b.sum()

            # Add calibrated Gaussian noise, then average over the batch
            noisy_grad_w = (sum_grad_w + rng.normal(0, sigma, d)) / bs
            noisy_grad_b = (sum_grad_b + rng.normal(0, sigma)) / bs

            w -= lr * noisy_grad_w
            b -= lr * noisy_grad_b

    achieved_epsilon = total_epsilon_basic_composition(epsilon_per_step, n_steps)
    return w, b, achieved_epsilon


def run_dp(
    X_train, y_train, X_test, y_test, target_epsilon: float, seed: int = 0
) -> dict:
    t0 = time.perf_counter()
    w, b, achieved_eps = train_dp_sgd(
        X_train, y_train, target_epsilon=target_epsilon, seed=seed
    )
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = accuracy(X_test, y_test, w, b)
    n_test = X_test.shape[0]
    inference_time_per_sample_ms = (time.perf_counter() - t0) / n_test * 1000

    return {
        "mechanism": "differential_privacy",
        "epsilon": achieved_eps,
        "poly_modulus_degree": None,
        "accuracy": acc,
        "train_time_s": train_time,
        "inference_ms_per_sample": inference_time_per_sample_ms,
        "protects_against": "membership inference on training set",
    }
