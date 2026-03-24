#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Provenance: see src/provenance/experiments.rs — WDM_SQW_PROVENANCE
"""
nW-03: S(q,ω) Peak Predictor — LSTM on MD density fluctuation time series.

In molecular dynamics (MD), the dynamic structure factor S(q,ω) is
computed from the time autocorrelation of density fluctuations:

  S(q,ω) = |FT[δρ(q,t)]|²

where δρ(q,t) is a damped oscillation at the plasma frequency ω_p
with damping rate γ (Landau damping). The LSTM processes the time
series δρ(q,t) directly and predicts (ω_p, γ) without explicit
Fourier analysis.

For each (ρ, T) condition:
  δρ(t) = exp(-γt) · cos(ω_p·t + φ) + noise

where:
  ω_p ~ ρ^(1/2)  (plasma frequency)
  γ ~ T^(1/2) / ρ^(1/3)  (thermal damping)

Architecture: LSTM reservoir (fixed random weights, spectral radius
scaling) + ridge regression readout. Validates LSTM cell inference
for Rust port without requiring BPTT.

Reference: Hansen & McDonald, "Theory of Simple Liquids" (2013)
           Gregori et al., PRE 67, 026412 (2003)
           Jaeger, "The echo state approach" (2001)
License: AGPL-3.0-or-later

Provenance:
  Baseline commit: f9ad0268917a335dce2b1175ea0d77add271b25b
  Baseline date:   2026-02-16
  Command:         python3 control/wdm/sqw_peak_predictor.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
N_SAMPLES = 800
N_TIMESTEPS = 64
HIDDEN_SIZE = 32
TEST_FRACTION = 0.2
RIDGE_ALPHA = 1e-3
INPUT_SCALE = 0.5
SPECTRAL_RADIUS = 0.9
FORGET_BIAS = 1.0
WASHOUT = 4

R2_PEAK_MIN = 0.80
R2_WIDTH_MIN = 0.65
RMSE_MAX = 0.4


def generate_density_fluctuation(omega_p, gamma, n_steps, dt, seed_offset):
    """Generate a damped oscillation mimicking MD density fluctuation.

    δρ(t) = exp(-γ_red · t) · cos(ω_red · t + φ) + noise

    We work in reduced time units where the oscillation frequency is
    mapped to [0.05, 0.45] cycles/step (well within Nyquist) and the
    damping rate gives 2-8 e-folding times across the window.
    """
    rng = np.random.RandomState(SEED + seed_offset)

    t = np.arange(n_steps, dtype=np.float64) * dt

    signal = np.exp(-gamma * t) * np.cos(omega_p * t)

    noise_level = 0.03
    signal += noise_level * rng.randn(n_steps)

    return signal


def generate_dataset(n_samples, seed=SEED):
    """Generate (time_series, reduced_omega, reduced_gamma) dataset.

    Maps physical parameters to reduced units where ω ∈ [0.3, 2.8]
    rad/step and γ ∈ [0.02, 0.20] 1/step, ensuring the oscillations
    are well-resolved in the 64-step window.
    """
    rng = np.random.RandomState(seed)

    log_rho = rng.uniform(-0.5, 1.5, n_samples)
    log_T = rng.uniform(4.0, 7.5, n_samples)

    omega_reduced = 0.3 + 2.5 * (log_rho - (-0.5)) / 2.0
    gamma_reduced = 0.02 + 0.18 * (log_T - 4.0) / 3.5

    dt = 1.0
    series = np.zeros((n_samples, N_TIMESTEPS))
    for i in range(n_samples):
        series[i] = generate_density_fluctuation(
            omega_reduced[i], gamma_reduced[i], N_TIMESTEPS, dt, i
        )

    return series, omega_reduced, gamma_reduced, log_rho, log_T


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def lstm_cell(x_val, h_prev, c_prev, W_i, W_h, b_i, b_h, hs):
    """Single LSTM cell step. x_val is a scalar."""
    x = np.array([[x_val]])
    gates = (W_i @ x.T).ravel() + (W_h @ h_prev.reshape(-1, 1)).ravel() + b_i + b_h

    f = sigmoid(gates[:hs])
    i = sigmoid(gates[hs:2*hs])
    g = np.tanh(gates[2*hs:3*hs])
    o = sigmoid(gates[3*hs:])

    c_new = f * c_prev + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def lstm_forward(time_series, W_i, W_h, b_i, b_h, hs):
    """Process time series through LSTM, return final hidden state."""
    h = np.zeros(hs)
    c = np.zeros(hs)
    for val in time_series:
        h, c = lstm_cell(val, h, c, W_i, W_h, b_i, b_h, hs)
    return h


def lstm_all_hidden(time_series, W_i, W_h, b_i, b_h, hs):
    """Process time series, return ALL hidden states (n_steps, hs)."""
    h = np.zeros(hs)
    c = np.zeros(hs)
    all_h = []
    for val in time_series:
        h, c = lstm_cell(val, h, c, W_i, W_h, b_i, b_h, hs)
        all_h.append(h.copy())
    return np.array(all_h)


def get_hidden_representations(data, W_i, W_h, b_i, b_h, hs, washout=WASHOUT):
    """Get pooled LSTM representation for each time series.

    Collects all hidden states after washout, then computes
    [mean, std, last] to form the readout feature vector.
    This captures both average dynamics and temporal evolution.
    """
    n = len(data)
    feat_dim = 3 * hs
    H = np.zeros((n, feat_dim))
    for i in range(n):
        all_h = lstm_all_hidden(data[i], W_i, W_h, b_i, b_h, hs)
        valid_h = all_h[washout:]
        h_mean = valid_h.mean(axis=0)
        h_std = valid_h.std(axis=0)
        h_last = valid_h[-1]
        H[i] = np.concatenate([h_mean, h_std, h_last])
    return H


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def main():
    np.random.seed(SEED)
    t0 = time.time()

    print("=== nW-03: S(q,ω) Peak Predictor (LSTM on MD Time Series) ===")
    print(f"Samples: {N_SAMPLES}, Timesteps: {N_TIMESTEPS}")
    print(f"LSTM hidden: {HIDDEN_SIZE}, ridge α: {RIDGE_ALPHA}")
    print()

    series, omega_red, gamma_red, log_rho, log_T = generate_dataset(N_SAMPLES)

    s_mean = series.mean()
    s_std = series.std()
    s_std = max(s_std, 1e-12)
    series_norm = (series - s_mean) / s_std

    y = np.column_stack([omega_red, gamma_red])
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    y_norm = (y - y_mean) / y_std

    n = len(series)
    n_test = max(1, int(n * TEST_FRACTION))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    train_idx = perm[n_test:]
    test_idx = perm[:n_test]

    x_train = series_norm[train_idx]
    x_test = series_norm[test_idx]
    y_train = y_norm[train_idx]
    y_test = y_norm[test_idx]

    print(f"  Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"  ω_reduced range: [{omega_red.min():.3f}, {omega_red.max():.3f}] rad/step")
    print(f"  γ_reduced range: [{gamma_red.min():.3f}, {gamma_red.max():.3f}] 1/step")

    rng_w = np.random.RandomState(SEED)
    hs = HIDDEN_SIZE

    W_i = rng_w.randn(4 * hs, 1) * INPUT_SCALE

    W_h_raw = rng_w.randn(4 * hs, hs) * 0.1
    h_block = W_h_raw[:hs, :]
    eig_vals = np.linalg.eigvals(h_block)
    rho_max = np.max(np.abs(eig_vals))
    if rho_max > 1e-10:
        W_h_raw *= (SPECTRAL_RADIUS / rho_max)
    W_h = W_h_raw

    b_i = np.zeros(4 * hs)
    b_i[:hs] = FORGET_BIAS
    b_h = np.zeros(4 * hs)

    print("\n  Computing LSTM hidden representations...")
    t1 = time.time()
    H_train = get_hidden_representations(x_train, W_i, W_h, b_i, b_h, hs)
    H_test = get_hidden_representations(x_test, W_i, W_h, b_i, b_h, hs)
    t_repr = time.time() - t1
    print(f"  Representations computed in {t_repr:.1f}s")

    h_active = np.mean(np.abs(H_train) > 0.01)
    print(f"  Hidden unit activity: {100*h_active:.0f}%")

    H_aug = np.column_stack([H_train, np.ones(len(H_train))])
    reg = RIDGE_ALPHA * np.eye(H_aug.shape[1])
    reg[-1, -1] = 0.0
    W_out_aug = np.linalg.solve(H_aug.T @ H_aug + reg, H_aug.T @ y_train)

    W_out = W_out_aug[:-1]
    b_out = W_out_aug[-1]

    pred_train = H_train @ W_out + b_out
    pred_test = H_test @ W_out + b_out

    r2_peak_test = r2_score(y_test[:, 0], pred_test[:, 0])
    r2_width_test = r2_score(y_test[:, 1], pred_test[:, 1])
    rmse_test = np.sqrt(np.mean((pred_test - y_test) ** 2))

    r2_peak_train = r2_score(y_train[:, 0], pred_train[:, 0])
    r2_width_train = r2_score(y_train[:, 1], pred_train[:, 1])

    print(f"\n  Train: R²(ω)={r2_peak_train:.4f}, R²(γ)={r2_width_train:.4f}")
    print(f"  Test:  R²(ω)={r2_peak_test:.4f}, R²(γ)={r2_width_test:.4f}")
    print(f"  Test RMSE: {rmse_test:.4f}")

    ref_preds = []
    ref_points = [
        (0.5, 5.0), (1.0, 6.0), (0.0, 5.5), (-0.3, 4.5), (1.2, 7.0),
    ]
    for lr_val, lt_val in ref_points:
        omega_r = 0.3 + 2.5 * (lr_val - (-0.5)) / 2.0
        gamma_r = 0.02 + 0.18 * (lt_val - 4.0) / 3.5
        ts = generate_density_fluctuation(omega_r, gamma_r, N_TIMESTEPS, 1.0, 9999)
        ts_n = (ts - s_mean) / s_std
        all_h = lstm_all_hidden(ts_n, W_i, W_h, b_i, b_h, hs)
        valid_h = all_h[WASHOUT:]
        feat = np.concatenate([valid_h.mean(0), valid_h.std(0), valid_h[-1]])
        pred_n = feat @ W_out + b_out
        pred_orig = pred_n * y_std + y_mean
        ref_preds.append({
            "log_rho": lr_val,
            "log_T": lt_val,
            "pred_omega": float(pred_orig[0]),
            "pred_gamma": float(pred_orig[1]),
            "true_omega": float(omega_r),
            "true_gamma": float(gamma_r),
        })

    print()
    checks = [
        (f"R²(ω) > {R2_PEAK_MIN}", r2_peak_test > R2_PEAK_MIN),
        (f"R²(γ) > {R2_WIDTH_MIN}", r2_width_test > R2_WIDTH_MIN),
        (f"RMSE < {RMSE_MAX}", rmse_test < RMSE_MAX),
        ("predictions finite", bool(np.isfinite(pred_test).all())),
        ("peak R² train ≥ test - 0.05", r2_peak_train >= r2_peak_test - 0.05),
    ]

    n_pass = 0
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        print(f"  {status}: {name}")

    elapsed = time.time() - t0
    print(f"\n  {n_pass}/{len(checks)} checks PASS ({elapsed:.1f}s)")

    output = {
        "_source": "neuralSpring nW-03 — S(q,ω) Peak Predictor",
        "_citation": "Hansen & McDonald (2013), Gregori et al. PRE 67 (2003)",
        "_method": "LSTM reservoir + ridge regression readout on MD time series",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_timesteps": N_TIMESTEPS,
        "lstm_config": {
            "hidden_size": HIDDEN_SIZE,
            "input_size": 1,
            "output_size": 2,
            "ridge_alpha": RIDGE_ALPHA,
            "input_scale": INPUT_SCALE,
            "spectral_radius": SPECTRAL_RADIUS,
        },
        "normalization": {
            "series_mean": float(s_mean),
            "series_std": float(s_std),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        },
        "weights": {
            "hidden_size": HIDDEN_SIZE,
            "W_i": W_i.flatten().tolist(),
            "W_h": W_h.flatten().tolist(),
            "b_i": b_i.tolist(),
            "b_h": b_h.tolist(),
            "W_out": W_out.flatten().tolist(),
            "b_out": b_out.tolist(),
        },
        "r2_omega": float(r2_peak_test),
        "r2_gamma": float(r2_width_test),
        "rmse": float(rmse_test),
        "reference_predictions": ref_preds,
        "result": f"{n_pass}/{len(checks)} PASS",
        "_provenance": {
            "date": time.strftime("%Y-%m-%d"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "sqw_peak_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Baseline: {out_path}")

    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
