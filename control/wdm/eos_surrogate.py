#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
nW-02: EOS Surrogate Validation — MLP surrogate for FPEOS tables.

Trains an MLP to predict pressure P(rho, T) and energy E(rho, T) for
H, He, and C from the Militzer FPEOS database. Validates that the
surrogate reproduces the first-principles data within documented tolerance.

Citation: Militzer et al., PRE 103, 013203 (2021)
Source:   https://militzer.berkeley.edu/FPEOS/

Author: ecoPrimals
License: AGPL-3.0-or-later

Provenance:
  Baseline commit: f9ad0268917a335dce2b1175ea0d77add271b25b
  Baseline date:   2026-02-16
  Command:         python3 control/wdm/eos_surrogate.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "fpeos_data")

ELEMENTS = ["H", "He", "C"]
TABLE_FILES = {
    "H": "H_EOS_09-18-20.txt",
    "He": "He_EOS_09-18-20.txt",
    "C": "C_EOS_09-18-20.txt",
}

MLP_HIDDEN = [128, 128]
EPOCHS = 1000
LR = 0.001
BATCH_SIZE = 64
SEED = 42
TEST_FRACTION = 0.2

R2_PRESSURE_MIN = 0.95
R2_ENERGY_MIN = 0.70
RMSE_MAX = 1.0


def parse_fpeos_table(filepath):
    """Parse an FPEOS table file into structured arrays.

    Returns dict with keys: rho, T, P, P_err, E, E_err (all numpy arrays).
    """
    rho, temp, pres, pres_err, energy, energy_err = [], [], [], [], [], []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Format: f= X N= 1 rho[g/cc]= ... T[K]= ... P[GPa]= ... err E[Ha]= ... err
            try:
                rho_idx = parts.index("rho[g/cc]=") + 1
                t_idx = parts.index("T[K]=") + 1
                p_idx = parts.index("P[GPa]=") + 1
                e_idx = parts.index("E[Ha]=") + 1

                rho.append(float(parts[rho_idx]))
                temp.append(float(parts[t_idx]))
                pres.append(float(parts[p_idx]))
                pres_err.append(float(parts[p_idx + 1]))
                energy.append(float(parts[e_idx]))
                energy_err.append(float(parts[e_idx + 1]))
            except (ValueError, IndexError):
                continue

    return {
        "rho": np.array(rho),
        "T": np.array(temp),
        "P": np.array(pres),
        "P_err": np.array(pres_err),
        "E": np.array(energy),
        "E_err": np.array(energy_err),
    }


def z_normalize(x, mean=None, std=None):
    """Z-score normalization. Returns (normalized, mean, std)."""
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


class SimpleMLP:
    """Numpy-only MLP for reproducibility (no PyTorch dependency)."""

    def __init__(self, layer_sizes, seed=42):
        self.rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = self.rng.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        self._activations = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(0, z)  # ReLU
            else:
                h = z
            self._activations.append(h)
        return h

    def train(self, x_train, y_train, epochs, lr, batch_size):
        n = x_train.shape[0]
        for epoch in range(epochs):
            indices = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]
                xb, yb = x_train[idx], y_train[idx]
                self._backward(xb, yb, lr)

    def _backward(self, x, y, lr):
        pred = self.forward(x)
        n = x.shape[0]
        grad = 2.0 * (pred - y) / n

        for i in range(len(self.weights) - 1, -1, -1):
            dw = self._activations[i].T @ grad
            db = grad.sum(axis=0)
            if i > 0:
                grad = grad @ self.weights[i].T
                grad = grad * (self._activations[i] > 0).astype(float)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def export_weights(self):
        """Export weights/biases as flat lists for Rust port."""
        result = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            result.append({
                "layer": i,
                "weights": w.flatten().tolist(),
                "bias": b.flatten().tolist(),
                "in_features": w.shape[0],
                "out_features": w.shape[1],
            })
        return result


def train_eos_surrogate(element, data, seed=SEED):
    """Train MLP surrogate P(rho,T), E(rho,T) for one element."""
    rng = np.random.RandomState(seed)

    log_rho = np.log10(data["rho"] + 1e-30)
    log_t = np.log10(data["T"] + 1e-30)
    x_raw = np.column_stack([log_rho, log_t])

    log_p = np.sign(data["P"]) * np.log10(np.abs(data["P"]) + 1e-30)
    log_e = np.sign(data["E"]) * np.log10(np.abs(data["E"]) + 1e-30)
    y_raw = np.column_stack([log_p, log_e])

    n = len(x_raw)
    n_test = max(1, int(n * TEST_FRACTION))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    x_train_raw, x_test_raw = x_raw[train_idx], x_raw[test_idx]
    y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

    x_train, x_mean, x_std = z_normalize(x_train_raw)
    x_test, _, _ = z_normalize(x_test_raw, x_mean, x_std)
    y_train, y_mean, y_std = z_normalize(y_train_raw)
    y_test, _, _ = z_normalize(y_test_raw, y_mean, y_std)

    layers = [2] + MLP_HIDDEN + [2]
    mlp = SimpleMLP(layers, seed=seed)
    mlp.train(x_train, y_train, EPOCHS, LR, BATCH_SIZE)

    pred_test = mlp.forward(x_test)
    mse = np.mean((pred_test - y_test) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_test - pred_test) ** 2, axis=0)
    ss_tot = np.sum((y_test - y_test.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot < 1e-30, 1.0, ss_tot)

    return {
        "element": element,
        "n_train": len(train_idx),
        "n_test": n_test,
        "rmse": float(rmse),
        "r2_P": float(r2[0]),
        "r2_E": float(r2[1]),
        "mlp_layers": layers,
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        },
        "weights": mlp.export_weights(),
    }


def main():
    np.random.seed(SEED)
    t0 = time.time()
    results = {}
    n_pass = 0
    n_total = 0

    print("=== nW-02: EOS Surrogate Validation ===")
    print(f"Source: Militzer FPEOS (PRE 103, 013203)")
    print(f"Elements: {', '.join(ELEMENTS)}")
    print(f"MLP: {MLP_HIDDEN}, epochs={EPOCHS}, lr={LR}")
    print()

    for element in ELEMENTS:
        filepath = os.path.join(DATA_DIR, TABLE_FILES[element])
        if not os.path.exists(filepath):
            print(f"  SKIP {element}: {filepath} not found")
            continue

        data = parse_fpeos_table(filepath)
        n_points = len(data["rho"])
        print(f"  {element}: {n_points} data points, rho=[{data['rho'].min():.4f}, {data['rho'].max():.2f}] g/cc, "
              f"T=[{data['T'].min():.0f}, {data['T'].max():.0f}] K")

        result = train_eos_surrogate(element, data)
        results[element] = result

        n_total += 3
        for check, passed in [
            (f"R²(P) > {R2_PRESSURE_MIN}", result["r2_P"] > R2_PRESSURE_MIN),
            (f"R²(E) > {R2_ENERGY_MIN}", result["r2_E"] > R2_ENERGY_MIN),
            (f"RMSE < {RMSE_MAX}", result["rmse"] < RMSE_MAX),
        ]:
            status = "PASS" if passed else "FAIL"
            if passed:
                n_pass += 1
            print(f"    {status}: {element} {check} "
                  f"(R²_P={result['r2_P']:.4f}, R²_E={result['r2_E']:.4f}, RMSE={result['rmse']:.4f})")

    elapsed = time.time() - t0
    print(f"\n  {n_pass}/{n_total} checks PASS ({elapsed:.1f}s)")

    output = {
        "_source": "neuralSpring nW-02 — EOS Surrogate Validation",
        "_citation": "Militzer et al., PRE 103, 013203 (2021)",
        "_data": "https://militzer.berkeley.edu/FPEOS/",
        "seed": SEED,
        "mlp_config": {"hidden": MLP_HIDDEN, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE},
        "elements": results,
        "result": f"{n_pass}/{n_total} PASS",
        "_provenance": {
            "date": time.strftime("%Y-%m-%d"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "command": f"python3 {os.path.basename(__file__)}",
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "eos_surrogate_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Baseline saved: {out_path}")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
