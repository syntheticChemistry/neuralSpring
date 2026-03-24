#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Provenance: see src/provenance/experiments.rs — WDM_ESN_PROVENANCE
"""
nW-05: ESN Classifier for WDM Regime Detection.

Echo State Network (ESN) classifier that predicts the WDM regime from
(ρ, T) conditions. Three regimes:

  0 = Classical plasma (Γ < 1, weakly coupled)
  1 = Warm Dense Matter  (1 ≤ Γ ≤ 10, partially degenerate)
  2 = Degenerate plasma  (Γ > 10, strongly coupled/quantum)

where Γ = Z²e² / (a_ws · k_B · T) is the Coulomb coupling parameter
and a_ws = (3/4πn_i)^(1/3) is the Wigner-Seitz radius.

Architecture: ESN with tanh reservoir (fixed random weights, spectral
radius scaling) + ridge regression readout with one-hot targets.
Validates reservoir computing pattern for Rust port.

Reference: Jaeger, "The echo state approach" (2001)
           Ichimaru, "Statistical Plasma Physics" (1994)
License: AGPL-3.0-or-later

Provenance:
  Baseline commit: f9ad0268917a335dce2b1175ea0d77add271b25b
  Baseline date:   2026-02-16
  Command:         python3 control/wdm/esn_regime_classifier.py
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
N_SAMPLES = 1000
RESERVOIR_SIZE = 64
INPUT_DIM = 2
N_CLASSES = 3
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 0.5
RIDGE_ALPHA = 1e-3
TEST_FRACTION = 0.2

ACCURACY_MIN = 0.85


def coupling_parameter(rho, T, Z_star=1.0, A=1.0):
    """Compute Coulomb coupling Γ for given density and temperature."""
    m_p = 1.673e-24
    n_i = rho / (A * m_p)
    a_ws = (3.0 / (4.0 * np.pi * n_i)) ** (1.0 / 3.0)
    k_B_eV = 8.617e-5
    e_cgs = 4.803e-10
    coulomb_energy = Z_star ** 2 * e_cgs ** 2 / a_ws
    thermal_energy = k_B_eV * T * 1.602e-12
    return coulomb_energy / (thermal_energy + 1e-30)


def classify_regime(gamma):
    """Map coupling parameter to regime label."""
    if gamma < 1.0:
        return 0  # Classical
    elif gamma <= 10.0:
        return 1  # WDM
    else:
        return 2  # Degenerate


def generate_dataset(n_samples, seed=SEED):
    """Generate (log_rho, log_T) → regime_label dataset.

    Samples uniformly in log-space to cover all three regimes.
    """
    rng = np.random.RandomState(seed)

    log_rho = rng.uniform(-1.0, 2.5, n_samples)
    log_T = rng.uniform(3.5, 8.5, n_samples)

    rho = 10.0 ** log_rho
    T = 10.0 ** log_T

    gammas = np.array([coupling_parameter(r, t) for r, t in zip(rho, T)])
    labels = np.array([classify_regime(g) for g in gammas])

    return log_rho, log_T, gammas, labels


def esn_transform(x, W_in, W_res, b_res, reservoir_size):
    """Transform input through ESN reservoir.

    For a static input (not a time series), we do a single-step
    nonlinear transformation: h = tanh(W_in · x + W_res · 0 + b)
    followed by a second "self-recurrence" step to add nonlinearity:
    h2 = tanh(W_in · x + W_res · h + b).

    Two steps give the reservoir enough nonlinear capacity for the
    classification boundary without needing a full time series.
    """
    h = np.tanh(W_in @ x + b_res)
    h = np.tanh(W_in @ x + W_res @ h + b_res)
    return h


def main():
    np.random.seed(SEED)
    t0 = time.time()

    print("=== nW-05: ESN WDM Regime Classifier ===")
    print(f"Samples: {N_SAMPLES}, Reservoir: {RESERVOIR_SIZE}")
    print(f"Classes: Classical(Γ<1), WDM(1≤Γ≤10), Degenerate(Γ>10)")
    print()

    log_rho, log_T, gammas, labels = generate_dataset(N_SAMPLES)

    class_counts = [np.sum(labels == c) for c in range(N_CLASSES)]
    print(f"  Class distribution: Classical={class_counts[0]}, "
          f"WDM={class_counts[1]}, Degenerate={class_counts[2]}")

    x_raw = np.column_stack([log_rho, log_T])
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    x_norm = (x_raw - x_mean) / x_std

    y_onehot = np.zeros((N_SAMPLES, N_CLASSES))
    for i, lab in enumerate(labels):
        y_onehot[i, lab] = 1.0

    n = len(x_norm)
    n_test = max(1, int(n * TEST_FRACTION))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    train_idx = perm[n_test:]
    test_idx = perm[:n_test]

    x_train = x_norm[train_idx]
    x_test = x_norm[test_idx]
    y_train = y_onehot[train_idx]
    y_test_oh = y_onehot[test_idx]
    labels_test = labels[test_idx]

    rng_w = np.random.RandomState(SEED)
    W_in = rng_w.randn(RESERVOIR_SIZE, INPUT_DIM) * INPUT_SCALE

    W_res_raw = rng_w.randn(RESERVOIR_SIZE, RESERVOIR_SIZE)
    eig_vals = np.linalg.eigvals(W_res_raw)
    rho_max = np.max(np.abs(eig_vals))
    W_res = W_res_raw * (SPECTRAL_RADIUS / max(rho_max, 1e-10))

    b_res = rng_w.randn(RESERVOIR_SIZE) * 0.1

    print("\n  Computing reservoir representations...")
    H_train = np.array([esn_transform(x, W_in, W_res, b_res, RESERVOIR_SIZE)
                        for x in x_train])
    H_test = np.array([esn_transform(x, W_in, W_res, b_res, RESERVOIR_SIZE)
                       for x in x_test])

    H_aug = np.column_stack([H_train, np.ones(len(H_train))])
    reg = RIDGE_ALPHA * np.eye(H_aug.shape[1])
    reg[-1, -1] = 0.0
    W_out_aug = np.linalg.solve(H_aug.T @ H_aug + reg, H_aug.T @ y_train)
    W_out = W_out_aug[:-1]
    b_out = W_out_aug[-1]

    pred_train_raw = H_train @ W_out + b_out
    pred_test_raw = H_test @ W_out + b_out

    pred_train_labels = pred_train_raw.argmax(axis=1)
    pred_test_labels = pred_test_raw.argmax(axis=1)

    train_acc = np.mean(pred_train_labels == labels[train_idx])
    test_acc = np.mean(pred_test_labels == labels_test)

    per_class_acc = []
    for c in range(N_CLASSES):
        mask = labels_test == c
        if mask.sum() > 0:
            acc = np.mean(pred_test_labels[mask] == c)
            per_class_acc.append(float(acc))
        else:
            per_class_acc.append(0.0)

    print(f"\n  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Per-class test: Classical={per_class_acc[0]:.3f}, "
          f"WDM={per_class_acc[1]:.3f}, Degenerate={per_class_acc[2]:.3f}")

    ref_preds = []
    ref_points = [
        (-0.5, 7.0, "Classical (hot, low density)"),
        (2.0, 4.0, "Degenerate (cold, high density)"),
        (0.5, 5.5, "WDM boundary region"),
        (1.0, 6.0, "Moderate coupling"),
        (1.5, 5.0, "Strong coupling"),
    ]
    print()
    for lr_val, lt_val, desc in ref_points:
        x_ref = np.array([(lr_val - x_mean[0]) / x_std[0],
                          (lt_val - x_mean[1]) / x_std[1]])
        h_ref = esn_transform(x_ref, W_in, W_res, b_res, RESERVOIR_SIZE)
        pred_ref_raw = h_ref @ W_out + b_out
        pred_label = int(pred_ref_raw.argmax())
        gamma_ref = coupling_parameter(10**lr_val, 10**lt_val)
        true_label = classify_regime(gamma_ref)
        label_names = ["Classical", "WDM", "Degenerate"]
        ref_preds.append({
            "log_rho": lr_val,
            "log_T": lt_val,
            "pred_label": pred_label,
            "true_label": true_label,
            "pred_name": label_names[pred_label],
            "true_name": label_names[true_label],
            "gamma": float(gamma_ref),
            "scores": pred_ref_raw.tolist(),
        })
        match = "✓" if pred_label == true_label else "✗"
        print(f"  {match} ({lr_val:.1f},{lt_val:.1f}) Γ={gamma_ref:.2e}: "
              f"pred={label_names[pred_label]}, true={label_names[true_label]} — {desc}")

    print()
    checks = [
        (f"test accuracy > {ACCURACY_MIN}", test_acc > ACCURACY_MIN),
        ("Classical accuracy > 0.80", per_class_acc[0] > 0.80),
        ("WDM accuracy > 0.70", per_class_acc[1] > 0.70),
        ("Degenerate accuracy > 0.80", per_class_acc[2] > 0.80),
        ("predictions finite", bool(np.isfinite(pred_test_raw).all())),
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
        "_source": "neuralSpring nW-05 — ESN WDM Regime Classifier",
        "_citation": "Jaeger (2001), Ichimaru (1994)",
        "_method": "Echo State Network + ridge regression readout",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_classes": N_CLASSES,
        "class_names": ["Classical", "WDM", "Degenerate"],
        "esn_config": {
            "reservoir_size": RESERVOIR_SIZE,
            "input_dim": INPUT_DIM,
            "spectral_radius": SPECTRAL_RADIUS,
            "input_scale": INPUT_SCALE,
            "ridge_alpha": RIDGE_ALPHA,
        },
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
        },
        "weights": {
            "reservoir_size": RESERVOIR_SIZE,
            "input_dim": INPUT_DIM,
            "n_classes": N_CLASSES,
            "W_in": W_in.flatten().tolist(),
            "W_res": W_res.flatten().tolist(),
            "b_res": b_res.tolist(),
            "W_out": W_out.flatten().tolist(),
            "b_out": b_out.tolist(),
        },
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "per_class_accuracy": per_class_acc,
        "reference_predictions": ref_preds,
        "result": f"{n_pass}/{len(checks)} PASS",
        "_provenance": {
            "date": time.strftime("%Y-%m-%d"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "esn_regime_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Baseline: {out_path}")

    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
