"""Interactive Gradio dashboard for the privacy-utility tradeoff benchmark.

Run with:
    uv run python -m privacy_utility.app
"""

from __future__ import annotations

import gradio as gr
import plotly.graph_objects as go

from privacy_utility.benchmark import run_full_benchmark

ACCENT = "#4a7fa5"
BASE = "#1a2e4a"


def _dp_curve(df):
    dp = df.filter(df["mechanism"] == "differential_privacy").sort("epsilon")
    baseline_acc = df.filter(df["mechanism"] == "plaintext")["accuracy"][0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dp["epsilon"].to_list(), y=dp["accuracy"].to_list(),
        mode="lines+markers", name="DP-SGD", line=dict(color=ACCENT, width=3),
        marker=dict(size=9),
    ))
    fig.add_hline(y=baseline_acc, line_dash="dash", line_color=BASE,
                   annotation_text="plaintext baseline")
    fig.update_layout(
        title="Utility cost of differential privacy",
        xaxis_title="Privacy budget epsilon (lower = more private, basic-composition bound)",
        yaxis_title="Test accuracy",
        template="plotly_white",
    )
    return fig


def _fhe_latency(df):
    fhe = df.filter(df["mechanism"] == "homomorphic_encryption").sort("poly_modulus_degree")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(v) for v in fhe["poly_modulus_degree"].to_list()],
        y=fhe["inference_ms_per_sample"].to_list(),
        marker_color=ACCENT, name="CKKS inference latency",
    ))
    fig.update_layout(
        title="Cost of homomorphic encryption: latency vs security parameter",
        xaxis_title="poly_modulus_degree (higher = more secure)",
        yaxis_title="Encrypted inference latency (ms/sample)",
        template="plotly_white",
    )
    return fig


def _fhe_accuracy(df):
    fhe = df.filter(df["mechanism"] == "homomorphic_encryption").sort("poly_modulus_degree")
    baseline_acc = df.filter(df["mechanism"] == "plaintext")["accuracy"][0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[str(v) for v in fhe["poly_modulus_degree"].to_list()],
        y=fhe["accuracy"].to_list(),
        mode="lines+markers", name="CKKS inference accuracy",
        line=dict(color="#c0392b", width=3), marker=dict(size=9),
    ))
    fig.add_hline(y=baseline_acc, line_dash="dash", line_color=BASE,
                   annotation_text="plaintext baseline")
    fig.update_layout(
        title="CKKS approximation error: accuracy vs security parameter",
        xaxis_title="poly_modulus_degree",
        yaxis_title="Test accuracy (evaluated on encrypted-inference subset)",
        template="plotly_white",
    )
    return fig


def run(epsilons_str, poly_degrees_str, n_fhe_samples, progress=gr.Progress()):
    epsilons = [float(x.strip()) for x in epsilons_str.split(",") if x.strip()]
    poly_degrees = [int(x.strip()) for x in poly_degrees_str.split(",") if x.strip()]

    progress(0.05, desc="Training plaintext baseline")
    progress(0.2, desc="Running DP-SGD across privacy budgets")
    progress(0.5, desc="Running CKKS encrypted inference across security levels")
    df = run_full_benchmark(
        epsilons=epsilons, poly_modulus_degrees=poly_degrees, n_fhe_eval_samples=int(n_fhe_samples)
    )
    progress(0.95, desc="Building plots")

    table = df.to_pandas()
    summary = (
        f"**Baseline accuracy (no privacy):** {df.filter(df['mechanism']=='plaintext')['accuracy'][0]:.3f}\n\n"
        f"**DP-SGD range:** accuracy {df.filter(df['mechanism']=='differential_privacy')['accuracy'].min():.3f}"
        f" – {df.filter(df['mechanism']=='differential_privacy')['accuracy'].max():.3f} "
        f"across epsilon in {epsilons}\n\n"
        f"**CKKS inference latency range:** "
        f"{df.filter(df['mechanism']=='homomorphic_encryption')['inference_ms_per_sample'].min():.2f} – "
        f"{df.filter(df['mechanism']=='homomorphic_encryption')['inference_ms_per_sample'].max():.2f} "
        f"ms/sample across poly_modulus_degree in {poly_degrees}"
    )

    return summary, _dp_curve(df), _fhe_latency(df), _fhe_accuracy(df), table


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Privacy-Utility Tradeoff Explorer") as demo:
        gr.Markdown(
            "# Privacy-Utility Tradeoff Explorer\n"
            "Trains a logistic regression classifier (breast cancer diagnosis dataset) under three "
            "regimes: a **plaintext baseline**, **DP-SGD** at a range of privacy budgets (epsilon, "
            "via per-example gradient clipping and calibrated Gaussian noise), and **CKKS-encrypted "
            "inference** (via TenSEAL / Microsoft SEAL) at a range of security parameters. Differential "
            "privacy trades *accuracy* for a membership-inference guarantee; homomorphic encryption "
            "trades *latency* for a guarantee that the server never sees the client's raw input. "
            "These are different threat models, and this tool makes the actual numeric cost of each "
            "one directly comparable."
        )

        with gr.Row():
            with gr.Column(scale=1):
                epsilons_str = gr.Textbox(
                    label="DP privacy budgets (epsilon, comma-separated)",
                    value="0.5, 1.0, 2.0, 4.0, 8.0",
                )
                poly_degrees_str = gr.Textbox(
                    label="CKKS poly_modulus_degree values (comma-separated)",
                    value="4096, 8192, 16384",
                )
                n_fhe_samples = gr.Slider(10, 100, value=30, step=10,
                                           label="Test samples evaluated under encryption (slow — kept small)")
                run_btn = gr.Button("Run benchmark", variant="primary")
                gr.Markdown(
                    "*Note: encrypted inference runs real CKKS ciphertext operations per sample, so "
                    "it is evaluated on a subset of the test set rather than all of it, to keep run "
                    "time reasonable in this dashboard.*"
                )

            with gr.Column(scale=2):
                summary_md = gr.Markdown()
                dp_plot = gr.Plot(label="DP privacy-utility curve")
                with gr.Row():
                    fhe_latency_plot = gr.Plot(label="FHE latency vs security")
                    fhe_accuracy_plot = gr.Plot(label="FHE accuracy vs security")
                results_table = gr.Dataframe(label="Full results")

        run_btn.click(
            fn=run,
            inputs=[epsilons_str, poly_degrees_str, n_fhe_samples],
            outputs=[summary_md, dp_plot, fhe_latency_plot, fhe_accuracy_plot, results_table],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
