"""Runs the plaintext baseline, DP-SGD at multiple privacy budgets, and CKKS
encrypted inference at multiple security parameters, on the same
train/test split, and collects everything into a single Polars DataFrame
for easy comparison and plotting.
"""

from __future__ import annotations

import polars as pl

from privacy_utility.baseline import run_baseline
from privacy_utility.data import load_split
from privacy_utility.dp import run_dp
from privacy_utility.fhe import run_fhe


def run_full_benchmark(
    epsilons: list[float] | None = None,
    poly_modulus_degrees: list[int] | None = None,
    n_fhe_eval_samples: int = 40,
    seed: int = 0,
) -> pl.DataFrame:
    """Run baseline + DP-SGD (per epsilon) + CKKS inference (per security
    parameter) on the same train/test split, and return a tidy Polars
    DataFrame with one row per configuration.
    """
    epsilons = epsilons if epsilons is not None else [0.5, 1.0, 2.0, 4.0, 8.0]
    poly_modulus_degrees = (
        poly_modulus_degrees if poly_modulus_degrees is not None else [4096, 8192, 16384, 32768]
    )

    X_train, X_test, y_train, y_test = load_split(seed=seed)

    rows: list[dict] = []
    rows.append(run_baseline(X_train, y_train, X_test, y_test, seed=seed))

    for eps in epsilons:
        rows.append(run_dp(X_train, y_train, X_test, y_test, target_epsilon=eps, seed=seed))

    for pmd in poly_modulus_degrees:
        rows.append(
            run_fhe(
                X_train, y_train, X_test, y_test,
                poly_modulus_degree=pmd,
                n_eval_samples=n_fhe_eval_samples,
                seed=seed,
            )
        )

    return pl.DataFrame(rows)


if __name__ == "__main__":
    results = run_full_benchmark()
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(results)
