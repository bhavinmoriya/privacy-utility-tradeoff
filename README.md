# privacy-utility-tradeoff

A Polars-native benchmarking toolkit that quantifies the **statistical
utility cost of two different privacy mechanisms** — differential privacy
(DP-SGD, from scratch) and homomorphic encryption (CKKS, via TenSEAL /
Microsoft SEAL) — on the same logistic regression task, and exposes the
result as an interactive Gradio dashboard.

## Why this project

Privacy-preserving ML has (at least) two dominant paradigms with very
different guarantees and very different costs:

- **Differential privacy** perturbs training (or output) with calibrated
  noise. It protects against *membership inference on the training set*,
  degrades gracefully with a tunable budget (epsilon), but the noise is
  irreversible — utility loss is baked into the trained model.
- **Homomorphic encryption** lets a server compute on a client's *encrypted*
  input without ever seeing the plaintext. It protects against
  *the server learning the input*, is (with CKKS) only approximately exact,
  and its cost shows up as **latency and parameter-size overhead**, not
  necessarily as accuracy loss.

These two mechanisms answer different threat models and get compared far
less often than they should be, because doing so requires implementing both
properly rather than citing them abstractly. This project runs the same
classification task under: a plaintext baseline, DP-SGD logistic regression
at a range of privacy budgets, and CKKS-encrypted inference at a range of
security parameters — and reports accuracy, latency, and the effective
threat model side by side.

This directly extends the applied FHE work from the AnoMoB project (custom
comparison/non-linear function evaluation under OpenFHE/TFHE-rs) into the
statistical-learning-theory question of *how much utility a learning
algorithm retains once its inputs, training process, or outputs are
constrained by a formal privacy guarantee* — the research direction
outlined for the Simula UiB Department of Information Theory application.

## Project layout

```
src/privacy_utility/
    data.py        # dataset loading (breast cancer, built into scikit-learn)
    baseline.py     # plaintext logistic regression
    dp.py            # DP-SGD logistic regression + simple privacy accountant
    fhe.py            # CKKS-encrypted inference via TenSEAL
    benchmark.py      # runs all three, collects results into a Polars DataFrame
    app.py            # Gradio dashboard: privacy-utility tradeoff curves
tests/
    test_pipeline.py
```

## Setup

```bash
uv sync
```

## Run the dashboard

```bash
uv run python -m privacy_utility.app
```

## Run the benchmark from Python

```python
from privacy_utility.benchmark import run_full_benchmark

results = run_full_benchmark(
    epsilons=[0.5, 1.0, 2.0, 4.0, 8.0, float("inf")],
    poly_modulus_degrees=[2048, 4096, 8192, 16384],
)
print(results)
```

## Tests

```bash
uv run pytest
```

## Notes on scope

The DP accountant here is a simplified Gaussian-mechanism / basic-composition
accountant for illustration, not a production Rényi-DP accountant (e.g.
Opacus's). The CKKS inference protects only the linear layer of the model
(the standard practical pattern: encrypted dot product server-side, sigmoid
applied client-side after decryption, since sigmoid is not a polynomial CKKS
can evaluate exactly) — this mirrors the real constraint you run into with
approximate-arithmetic FHE schemes, which is exactly the CKKS vs TFHE
trade-off from the AnoMoB benchmarking work.
