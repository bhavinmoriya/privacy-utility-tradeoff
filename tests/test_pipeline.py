import numpy as np

from privacy_utility.baseline import accuracy, run_baseline, sigmoid, train_logreg
from privacy_utility.data import load_split
from privacy_utility.dp import (
    gaussian_sigma_for_epsilon,
    run_dp,
    total_epsilon_basic_composition,
    train_dp_sgd,
)
from privacy_utility.fhe import encrypted_linear_score, make_context, run_fhe


def test_load_split_shapes():
    X_train, X_test, y_train, y_test = load_split(seed=0)
    assert X_train.shape[1] == X_test.shape[1] == 30
    assert set(np.unique(y_train)) <= {0.0, 1.0}
    # standardized: training features should have ~zero mean, ~unit std
    assert abs(X_train.mean()) < 0.1
    assert abs(X_train.std() - 1.0) < 0.2


def test_sigmoid_range():
    z = np.array([-100, -1, 0, 1, 100])
    s = sigmoid(z)
    assert np.all(s >= 0) and np.all(s <= 1)
    assert s[2] == 0.5


def test_baseline_beats_chance():
    X_train, X_test, y_train, y_test = load_split(seed=0)
    result = run_baseline(X_train, y_train, X_test, y_test, seed=0)
    assert result["accuracy"] > 0.85  # breast cancer is fairly separable
    assert result["mechanism"] == "plaintext"


def test_dp_sigma_increases_as_epsilon_shrinks():
    sigma_tight = gaussian_sigma_for_epsilon(0.1, 1e-5, clip_norm=1.0)
    sigma_loose = gaussian_sigma_for_epsilon(5.0, 1e-5, clip_norm=1.0)
    assert sigma_tight > sigma_loose  # smaller epsilon -> more noise needed


def test_dp_composition_scales_with_steps():
    eps1 = total_epsilon_basic_composition(0.01, n_steps=10)
    eps2 = total_epsilon_basic_composition(0.01, n_steps=100)
    assert eps2 > eps1


def test_dp_accuracy_improves_with_looser_budget():
    X_train, X_test, y_train, y_test = load_split(seed=1)
    tight = run_dp(X_train, y_train, X_test, y_test, target_epsilon=0.3, seed=1)
    loose = run_dp(X_train, y_train, X_test, y_test, target_epsilon=20.0, seed=1)
    # Not a strict guarantee for one seed, but should hold on average /
    # with reasonable margin for this well-separated dataset.
    assert loose["accuracy"] >= tight["accuracy"] - 0.05


def test_ckks_context_and_encrypted_dot_product():
    context = make_context(poly_modulus_degree=8192)
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, 0.5, 0.5])
    score = encrypted_linear_score(context, x, w, b=0.0)
    assert abs(score - 3.0) < 0.01  # 0.5*1 + 0.5*2 + 0.5*3 = 3.0


def test_fhe_inference_runs_and_returns_expected_fields():
    X_train, X_test, y_train, y_test = load_split(seed=0)
    result = run_fhe(
        X_train, y_train, X_test, y_test,
        poly_modulus_degree=8192, n_eval_samples=5, seed=0,
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["inference_ms_per_sample"] > 0
    assert result["mechanism"] == "homomorphic_encryption"
