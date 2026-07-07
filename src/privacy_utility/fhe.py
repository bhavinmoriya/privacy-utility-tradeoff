"""Homomorphic encryption (CKKS) inference for the trained logistic
regression model, via TenSEAL (Python bindings for Microsoft SEAL).

Threat model: the model weights live on the server; the client wants a
prediction without revealing its raw feature vector to the server. The
client encrypts its input under CKKS, the server computes the encrypted dot
product (the linear layer) homomorphically without ever decrypting, and
returns the encrypted result. The client decrypts and applies sigmoid
locally, since sigmoid is not something CKKS can evaluate exactly (only
polynomial approximations are possible under CKKS) — this is the same
practical boundary you run into with any approximate-arithmetic FHE scheme,
mirroring the CKKS-vs-TFHE tradeoffs from the AnoMoB benchmarking work.
"""

from __future__ import annotations

import time

import numpy as np
import tenseal as ts

from privacy_utility.baseline import sigmoid


def make_context(poly_modulus_degree: int) -> ts.Context:
    """Create a CKKS context at a given security parameter. Larger
    poly_modulus_degree -> higher security level and more precision
    headroom, but slower encryption/computation/decryption.
    """
    # coeff_mod_bit_sizes must have length scaling with poly_modulus_degree
    # for SEAL's security estimates; we use a small set of standard depths.
    # Coefficient modulus bit sizes must stay within SEAL's 128-bit-security
    # budget for each poly_modulus_degree (2048 -> 54 bits, 4096 -> 109,
    # 8192 -> 218, 16384 -> 438 total). Smaller poly_modulus_degree therefore
    # means both lower security AND less precision headroom (fewer/smaller
    # moduli), which is exactly the security-vs-cost tradeoff this benchmark
    # is measuring.
    depth_profiles = {
        4096: [40, 20, 40],
        8192: [60, 40, 40, 60],
        16384: [60, 40, 40, 40, 40, 60],
        32768: [60, 50, 50, 50, 50, 50, 50, 60],
    }
    scale_profiles = {
        4096: 2 ** 20,
        8192: 2 ** 40,
        16384: 2 ** 40,
        32768: 2 ** 50,
    }
    coeff_mod_bit_sizes = depth_profiles.get(poly_modulus_degree, [60, 40, 40, 60])
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = scale_profiles.get(poly_modulus_degree, 2 ** 40)
    context.generate_galois_keys()
    return context


def encrypted_linear_score(
    context: ts.Context, x: np.ndarray, w: np.ndarray, b: float
) -> float:
    """Client encrypts x; server computes the encrypted dot product with
    plaintext weights w (+ bias b) without seeing x; result is decrypted
    (by whoever holds the secret key, i.e. the client) to get the linear
    score, which sigmoid is then applied to in plaintext.
    """
    enc_x = ts.ckks_vector(context, x.tolist())
    enc_score = enc_x.dot(w.tolist()) + b
    return enc_score.decrypt()[0]


def run_fhe(
    X_train,
    y_train,
    X_test,
    y_test,
    poly_modulus_degree: int,
    n_eval_samples: int = 40,
    seed: int = 0,
) -> dict:
    from privacy_utility.baseline import train_logreg

    # Train the model in plaintext (server-side); only inference is encrypted.
    w, b = train_logreg(X_train, y_train, seed=seed)

    context = make_context(poly_modulus_degree)

    rng = np.random.default_rng(seed)
    n_test = X_test.shape[0]
    eval_idx = rng.choice(n_test, size=min(n_eval_samples, n_test), replace=False)

    correct = 0
    latencies = []
    for i in eval_idx:
        t0 = time.perf_counter()
        score = encrypted_linear_score(context, X_test[i], w, b)
        latencies.append(time.perf_counter() - t0)
        pred = sigmoid(np.array([score]))[0] >= 0.5
        correct += int(pred == y_test[i])

    acc = correct / len(eval_idx)
    avg_latency_ms = float(np.mean(latencies)) * 1000

    return {
        "mechanism": "homomorphic_encryption",
        "epsilon": None,
        "poly_modulus_degree": poly_modulus_degree,
        "accuracy": acc,
        "train_time_s": None,
        "inference_ms_per_sample": avg_latency_ms,
        "protects_against": "server seeing the client's raw input",
    }
