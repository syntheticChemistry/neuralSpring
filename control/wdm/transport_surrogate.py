#!/usr/bin/env python3
"""
nW-01: WDM Transport Surrogate — MLP for Stanton-Murillo coefficients.

Extends hotSpring Paper 3 (Diaw surrogate) from classical (Gamma, kappa)
to WDM (rho, T, Z*) parameter space. Trains an MLP to predict diffusion
coefficient D*, viscosity eta*, and thermal conductivity lambda*.

The Stanton-Murillo effective potential model provides transport
coefficients for partially ionized plasmas via:
  D* = D / (a^2 * omega_p)  [reduced diffusion]
  eta* = eta / (n * m * a^2 * omega_p)  [reduced viscosity]
  lambda* = lambda / (n * k_B * a^2 * omega_p)  [reduced conductivity]

where a = (3/4*pi*n)^(1/3), omega_p = sqrt(Z*^2 * e^2 / (m * a^3)).

Reference: Stanton & Murillo, PRE 93, 043203 (2016)
License: AGPL-3.0-or-later

Provenance:
  Baseline commit: f9ad0268917a335dce2b1175ea0d77add271b25b
  Baseline date:   2026-02-16
  Command:         python3 control/wdm/transport_surrogate.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MLP_HIDDEN = [64, 64]
EPOCHS = 800
LR = 0.001
BATCH_SIZE = 32
SEED = 42
N_SAMPLES = 1000
TEST_FRACTION = 0.2


def generate_stanton_murillo_data(n_samples, seed=42):
    """Generate synthetic WDM transport data via SM effective potential.

    Parameters:
      rho: density (g/cc) in [0.1, 50]
      T: temperature (K) in [1e4, 1e8]
      Z_star: effective ionization in [1, 13] (H..Al)

    Outputs:
      D_star: reduced diffusion coefficient
      eta_star: reduced viscosity
      lambda_star: reduced thermal conductivity
    """
    rng = np.random.RandomState(seed)

    log_rho = rng.uniform(-1.0, 1.7, n_samples)  # 0.1 to 50 g/cc
    log_T = rng.uniform(4.0, 8.0, n_samples)      # 1e4 to 1e8 K
    Z_star = rng.uniform(1.0, 13.0, n_samples)

    rho = 10.0 ** log_rho
    T = 10.0 ** log_T

    k_B_eV = 8.617e-5  # eV/K
    a0 = 5.292e-9       # Bohr radius in cm
    m_p = 1.673e-24     # proton mass in g

    n_i = rho / (Z_star * m_p)
    a_ws = (3.0 / (4.0 * np.pi * n_i)) ** (1.0 / 3.0)

    Gamma = Z_star ** 2 * 14.4e-8 / (a_ws * k_B_eV * T)
    kappa = a_ws / (np.sqrt(k_B_eV * T / (4 * np.pi * n_i * (Z_star * 4.803e-10)**2 + 1e-30)) + 1e-30)

    Gamma_eff = Gamma * (1.0 + kappa / 3.0) * np.exp(-kappa)
    Gamma_eff = np.clip(Gamma_eff, 0.01, 200.0)

    # SM transport scaling (reduced units)
    D_star = 0.3 / (Gamma_eff ** 1.5 + 0.1) + 0.01
    eta_star = 0.2 * Gamma_eff ** 0.5 / (1.0 + 0.05 * Gamma_eff ** 2) + 0.005
    lambda_star = 1.5 / (Gamma_eff ** 0.8 + 0.2) + 0.02

    noise_scale = 0.02
    D_star *= (1.0 + noise_scale * rng.randn(n_samples))
    eta_star *= (1.0 + noise_scale * rng.randn(n_samples))
    lambda_star *= (1.0 + noise_scale * rng.randn(n_samples))

    return {
        "log_rho": log_rho,
        "log_T": log_T,
        "Z_star": Z_star,
        "D_star": D_star,
        "eta_star": eta_star,
        "lambda_star": lambda_star,
        "Gamma_eff": Gamma_eff,
    }


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

    def train(self, x, y, epochs, lr, batch_size):
        n = x.shape[0]
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                self._backward(x[idx[s:e]], y[idx[s:e]], lr)

    def _backward(self, x, y, lr):
        pred = self.forward(x)
        n = x.shape[0]
        g = 2.0 * (pred - y) / n
        for i in range(len(self.weights) - 1, -1, -1):
            dw = self._acts[i].T @ g
            db = g.sum(axis=0)
            if i > 0:
                g = (g @ self.weights[i].T) * (self._acts[i] > 0).astype(float)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def export_weights(self):
        return [{"layer": i, "weights": w.flatten().tolist(), "bias": b.flatten().tolist(),
                 "in_features": w.shape[0], "out_features": w.shape[1]}
                for i, (w, b) in enumerate(zip(self.weights, self.biases))]


def main():
    np.random.seed(SEED)
    t0 = time.time()

    print("=== nW-01: WDM Transport Surrogate ===")
    print("Model: Stanton-Murillo effective potential")
    print(f"Samples: {N_SAMPLES}, MLP: {MLP_HIDDEN}, epochs={EPOCHS}")
    print()

    data = generate_stanton_murillo_data(N_SAMPLES, SEED)
    x_raw = np.column_stack([data["log_rho"], data["log_T"], data["Z_star"]])

    log_d = np.log10(np.abs(data["D_star"]) + 1e-30)
    log_eta = np.log10(np.abs(data["eta_star"]) + 1e-30)
    log_lam = np.log10(np.abs(data["lambda_star"]) + 1e-30)
    y_raw = np.column_stack([log_d, log_eta, log_lam])

    n = len(x_raw)
    n_test = max(1, int(n * TEST_FRACTION))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    x_train, x_test = x_raw[perm[n_test:]], x_raw[perm[:n_test]]
    y_train, y_test = y_raw[perm[n_test:]], y_raw[perm[:n_test]]

    x_mean, x_std = x_train.mean(0), x_train.std(0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)

    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    layers = [3] + MLP_HIDDEN + [3]
    mlp = SimpleMLP(layers, SEED)
    mlp.train(x_train_n, y_train_n, EPOCHS, LR, BATCH_SIZE)

    pred = mlp.forward(x_test_n)
    mse = np.mean((pred - y_test_n) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_test_n - pred) ** 2, axis=0)
    ss_tot = np.sum((y_test_n - y_test_n.mean(0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot < 1e-30, 1.0, ss_tot)

    checks = [
        ("R²(D*) > 0.85", r2[0] > 0.85),
        ("R²(η*) > 0.85", r2[1] > 0.85),
        ("R²(λ*) > 0.85", r2[2] > 0.85),
        ("RMSE < 0.5", rmse < 0.5),
    ]

    n_pass = sum(1 for _, p in checks if p)
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name} (R²=[{r2[0]:.4f}, {r2[1]:.4f}, {r2[2]:.4f}], RMSE={rmse:.4f})")

    elapsed = time.time() - t0
    print(f"\n  {n_pass}/{len(checks)} checks PASS ({elapsed:.1f}s)")

    output = {
        "_source": "neuralSpring nW-01 — WDM Transport Surrogate",
        "_citation": "Stanton & Murillo, PRE 93, 043203 (2016)",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "mlp_config": {"hidden": MLP_HIDDEN, "epochs": EPOCHS, "lr": LR},
        "normalization": {
            "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        },
        "weights": mlp.export_weights(),
        "r2_D": float(r2[0]), "r2_eta": float(r2[1]), "r2_lambda": float(r2[2]),
        "rmse": float(rmse),
        "result": f"{n_pass}/{len(checks)} PASS",
        "_provenance": {
            "date": time.strftime("%Y-%m-%d"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "transport_surrogate_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Baseline: {out_path}")

    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
