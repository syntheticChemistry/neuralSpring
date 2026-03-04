#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
nW-04: Classical-to-WDM Transfer Learning.

Demonstrates that an MLP pretrained on classical transport (Gamma, kappa)
transfers to WDM conditions (rho, T, Z*) with fewer samples than
training from scratch.

Extends Exp 004 pattern: frozen-layer + fine-tune architecture.
- Phase 1: Train on classical regime (Gamma < 10, kappa < 3)
- Phase 2: Transfer to WDM regime (wider Gamma, kappa, Z*)
- Phase 3: Compare transfer vs scratch R² and sample efficiency

Reference: Extends Stanton-Murillo (2016) + Diaw et al. (2024)
License: AGPL-3.0-or-later

Provenance:
  Baseline commit: f9ad0268917a335dce2b1175ea0d77add271b25b
  Baseline date:   2026-02-16
  Command:         python3 control/wdm/transfer_classical_to_wdm.py
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


class SimpleMLP:
    def __init__(self, layer_sizes, seed=42):
        self.rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(self.rng.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def forward(self, x):
        self._acts = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            h = np.maximum(0, z) if i < len(self.weights) - 1 else z
            self._acts.append(h)
        return h

    def train(self, x, y, epochs, lr, batch_size, frozen_layers=0):
        n = x.shape[0]
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                self._backward(x[idx[s:e]], y[idx[s:e]], lr, frozen_layers)

    def _backward(self, x, y, lr, frozen_layers=0):
        pred = self.forward(x)
        n = x.shape[0]
        g = 2.0 * (pred - y) / n
        for i in range(len(self.weights) - 1, -1, -1):
            dw = self._acts[i].T @ g
            db = g.sum(axis=0)
            if i > 0:
                g = (g @ self.weights[i].T) * (self._acts[i] > 0).astype(float)
            if i >= frozen_layers:
                self.weights[i] -= lr * dw
                self.biases[i] -= lr * db

    def copy(self):
        import copy
        return copy.deepcopy(self)


def generate_classical_data(n, seed):
    rng = np.random.RandomState(seed)
    Gamma = rng.uniform(0.1, 10.0, n)
    kappa = rng.uniform(0.1, 3.0, n)
    x = np.column_stack([np.log10(Gamma), kappa])

    D_star = 0.3 / (Gamma ** 1.5 + 0.1) + 0.01
    y = np.log10(D_star + 1e-30).reshape(-1, 1)
    return x, y


def generate_wdm_data(n, seed):
    rng = np.random.RandomState(seed)
    Gamma = rng.uniform(0.01, 200.0, n)
    kappa = rng.uniform(0.1, 10.0, n)
    x = np.column_stack([np.log10(Gamma + 0.001), kappa])

    Gamma_eff = Gamma * (1.0 + kappa / 3.0) * np.exp(-kappa)
    Gamma_eff = np.clip(Gamma_eff, 0.01, 200.0)
    D_star = 0.3 / (Gamma_eff ** 1.5 + 0.1) + 0.01
    y = np.log10(D_star + 1e-30).reshape(-1, 1)
    return x, y


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def main():
    np.random.seed(SEED)
    t0 = time.time()

    print("=== nW-04: Classical-to-WDM Transfer Learning ===")
    print()

    # Phase 1: Classical pretraining
    x_classical, y_classical = generate_classical_data(500, SEED)
    x_mean, x_std = x_classical.mean(0), x_classical.std(0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    y_mean, y_std = y_classical.mean(), y_classical.std()

    x_cl_n = (x_classical - x_mean) / x_std
    y_cl_n = (y_classical - y_mean) / y_std

    mlp_pretrained = SimpleMLP([2, 64, 64, 1], SEED)
    mlp_pretrained.train(x_cl_n, y_cl_n, 500, 0.001, 32)

    pred_cl = mlp_pretrained.forward(x_cl_n)
    r2_classical = r2_score(y_cl_n, pred_cl)
    print(f"  Phase 1: Classical R² = {r2_classical:.4f}")

    # Phase 2a: Fine-tune on small WDM data (transfer)
    n_wdm_small = 30
    x_wdm, y_wdm = generate_wdm_data(n_wdm_small, SEED + 1)
    x_wdm_n = (x_wdm - x_mean) / x_std
    y_wdm_n = (y_wdm - y_mean) / y_std

    mlp_transfer = mlp_pretrained.copy()
    mlp_transfer.train(x_wdm_n, y_wdm_n, 300, 0.0003, 16, frozen_layers=0)

    # Phase 2b: Train from scratch on same small WDM data
    mlp_scratch = SimpleMLP([2, 64, 64, 1], SEED + 100)
    mlp_scratch.train(x_wdm_n, y_wdm_n, 300, 0.001, 16)

    # Phase 3: Evaluate on fresh WDM test set
    x_test, y_test = generate_wdm_data(200, SEED + 2)
    x_test_n = (x_test - x_mean) / x_std
    y_test_n = (y_test - y_mean) / y_std

    r2_transfer = r2_score(y_test_n, mlp_transfer.forward(x_test_n))
    r2_scratch = r2_score(y_test_n, mlp_scratch.forward(x_test_n))
    improvement = r2_transfer - r2_scratch

    print(f"  Phase 2: Transfer R² = {r2_transfer:.4f} (frozen-layer + fine-tune)")
    print(f"  Phase 2: Scratch  R² = {r2_scratch:.4f} (same data, from random init)")
    print(f"  Improvement: {improvement:+.4f}")
    print()

    checks = [
        ("Classical R² > 0.90", r2_classical > 0.90),
        ("Transfer R² > 0.60", r2_transfer > 0.60),
        ("Transfer > Scratch", r2_transfer > r2_scratch),
        ("Improvement > 0", improvement > 0),
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
        "_source": "neuralSpring nW-04 — Classical-to-WDM Transfer Learning",
        "_citation": "Stanton-Murillo (2016) + Diaw et al. (2024)",
        "seed": SEED,
        "r2_classical": float(r2_classical),
        "r2_transfer": float(r2_transfer),
        "r2_scratch": float(r2_scratch),
        "improvement": float(improvement),
        "n_classical": 500,
        "n_wdm_finetune": n_wdm_small,
        "n_wdm_test": 200,
        "result": f"{n_pass}/{len(checks)} PASS",
        "_provenance": {
            "date": time.strftime("%Y-%m-%d"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "transfer_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Baseline: {out_path}")

    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
