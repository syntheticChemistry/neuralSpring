#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 096: Digester Community–Performance Coupling via Anderson-ESN.

Novel composition: Paper 027 (ESN digestion prediction, Wang/Liao 2020)
× Paper 023 (Anderson localization, Bourgain & Kachkovskiy 2018).

Scientific question:
  Does microbial community diversity (modeled as Anderson disorder W)
  predict ESN yield prediction quality? Communities with high disorder W
  have fragmented QS signaling → less stable metabolic coordination →
  noisier biogas production → harder for the ESN to predict.

Design:
  1. Generate 12 communities with varying evenness (Dirichlet α)
  2. Map evenness → Anderson disorder W = W_max * (1 - evenness)
  3. Compute Anderson localization length ξ(W) via 1D eigensolve
  4. Generate digester data with community-dependent noise (∝ 1/ξ)
  5. Train ONE ESN on pooled data from all communities
  6. Measure per-community test R² on held-out data
  7. Test: Pearson r(W, R²) should be significantly negative

Components composed:
  - digestion_prediction (Paper 027): ESN, process model, biogas_yield
  - anderson_localization (Paper 023): Anderson Hamiltonian, IPR, eigensolve
  - barracuda::stats: R², Pearson correlation, variance

Provenance:
  Baseline commit: (first run)
  Baseline date:   2026-03-10
  Command:         python3 control/digester_anderson/digester_anderson.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""
import json
import sys
import numpy as np

SEED = 42
N_COMMUNITIES = 12
LATTICE_SIZE = 32
HOPPING = 1.0
N_SAMPLES_PER_COMMUNITY = 500
RESERVOIR_SIZE = 512
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 0.3
LEAK_RATE = 0.8
RIDGE_ALPHA = 0.01
RECURRENCE_STEPS = 2
INPUT_DIM = 5
TRAIN_FRAC = 0.7
W_MAX = 20.0


# ═════════════════════════════════════════════════════════════════════
# Anderson localization (from Paper 023)
# ═════════════════════════════════════════════════════════════════════

def anderson_hamiltonian(n, t, w, rng):
    """1D Anderson Hamiltonian with random diagonal disorder."""
    h = np.zeros((n, n))
    for i in range(n):
        h[i, i] = rng.uniform(-w / 2, w / 2)
    for i in range(n - 1):
        h[i, i + 1] = -t
        h[i + 1, i] = -t
    return h


def inverse_participation_ratio(psi):
    """IPR = sum(|ψ|^4). Extended: ~1/N, Localized: >> 1/N."""
    return np.sum(psi**4)


def mean_ipr_for_disorder(w, n, t, rng, n_realizations=5):
    """Mean IPR over realizations for given disorder strength."""
    iprs = []
    for _ in range(n_realizations):
        h = anderson_hamiltonian(n, t, w, rng)
        _, evecs = np.linalg.eigh(h)
        mipr = np.mean([inverse_participation_ratio(evecs[:, k]) for k in range(n)])
        iprs.append(mipr)
    return np.mean(iprs)


def localization_length(mipr, n):
    """Estimate ξ from mean IPR. Extended: ξ ~ N, Localized: ξ ~ 1."""
    if mipr < 1e-12:
        return float(n)
    return 1.0 / (n * mipr)


# ═════════════════════════════════════════════════════════════════════
# Digestion process model (from Paper 027)
# ═════════════════════════════════════════════════════════════════════

Y_BASE = 150.0
WT = 60.0
WPH = 40.0
WOLR = 50.0
WHRT = 60.0
WVS = 30.0
WT_OLR = 25.0

MESO_CENTER, MESO_SIGMA = 35.0, 6.0
THERMO_CENTER, THERMO_SIGMA = 55.0, 6.0
PH_CENTER, PH_SIGMA = 7.2, 1.0
K_OLR = 2.0
OLR_INHIBITION = 0.15
TAU_HRT = 10.0


def temperature_response(t):
    meso = 0.7 * np.exp(-0.5 * ((t - MESO_CENTER) / MESO_SIGMA) ** 2)
    thermo = 0.3 * np.exp(-0.5 * ((t - THERMO_CENTER) / THERMO_SIGMA) ** 2)
    return meso + thermo


def ph_response(ph):
    return np.exp(-0.5 * ((ph - PH_CENTER) / PH_SIGMA) ** 2)


def olr_response(olr):
    return olr / (K_OLR + olr) * np.exp(-OLR_INHIBITION * olr)


def hrt_response(hrt):
    return 1.0 - np.exp(-hrt / TAU_HRT)


def biogas_yield(t, ph, olr, hrt, vs_ts):
    f_t = temperature_response(t)
    f_ph = ph_response(ph)
    f_olr = olr_response(olr)
    f_hrt = hrt_response(hrt)
    f_vs = vs_ts / 100.0
    return (Y_BASE + WT * f_t + WPH * f_ph + WOLR * f_olr
            + WHRT * f_hrt + WVS * f_vs + WT_OLR * f_t * f_olr)


# ═════════════════════════════════════════════════════════════════════
# Community generation + noise model
# ═════════════════════════════════════════════════════════════════════

def generate_community(n_species, alpha, rng):
    """Generate abundance distribution via Dirichlet(α).

    Low α → dominated community (uneven). High α → even community.
    """
    abundances = rng.dirichlet(np.ones(n_species) * alpha)
    h_prime = -np.sum(abundances * np.log(abundances + 1e-30))
    h_max = np.log(n_species)
    evenness = h_prime / h_max if h_max > 1e-12 else 0.0
    return abundances, h_prime, evenness


def evenness_to_disorder(evenness):
    """Map evenness → Anderson disorder W. Even → low W, uneven → high W."""
    return W_MAX * (1.0 - evenness)


def noise_from_xi(xi, base=2.0, scale=2.0, cap=15.0):
    """Community noise: base + scale/ξ, capped. Localized → noisy.

    Calibrated so best communities (ξ≈0.6) get noise≈5 (matching Paper 027)
    while worst communities (ξ≈0.05) get noise≈15 (substantially degraded).
    """
    return min(base + scale / max(xi, 0.01), cap)


def generate_community_data(n_samples, noise_std, rng):
    """Generate digester data with community-specific noise."""
    records = []
    for _ in range(n_samples):
        t = rng.uniform(20.0, 60.0)
        ph = rng.uniform(5.5, 8.5)
        olr = rng.uniform(0.5, 8.0)
        hrt = rng.uniform(5.0, 40.0)
        vs_ts = rng.uniform(50.0, 90.0)
        y_true = biogas_yield(t, ph, olr, hrt, vs_ts)
        noise = rng.normal(0.0, noise_std)
        y_obs = max(y_true + noise, 0.0)
        records.append((t, ph, olr, hrt, vs_ts, y_true, y_obs))
    return records


# ═════════════════════════════════════════════════════════════════════
# ESN (from Paper 027 architecture)
# ═════════════════════════════════════════════════════════════════════

def esn_reservoir_drive(x_norm, w_in, w_res, b_res):
    """Drive reservoir for one sample, return final hidden state."""
    h = np.tanh(w_in @ x_norm + b_res)
    for _ in range(RECURRENCE_STEPS - 1):
        h = np.tanh(w_in @ x_norm + w_res @ h + b_res)
    return h


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


# ═════════════════════════════════════════════════════════════════════
# Main experiment
# ═════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.RandomState(SEED)

    # sweep from uneven (α=0.1) to even (α=10) Dirichlet concentration
    alpha_values = np.geomspace(0.1, 10.0, N_COMMUNITIES)
    n_species_base = 20

    # Phase 1: generate communities and Anderson properties
    communities = []
    for ic, alpha in enumerate(alpha_values):
        comm_rng = np.random.RandomState(SEED + ic * 1000)
        abundances, h_prime, evenness = generate_community(n_species_base, alpha, comm_rng)
        w_disorder = evenness_to_disorder(evenness)

        anderson_rng = np.random.RandomState(SEED + ic * 2000)
        mipr = mean_ipr_for_disorder(w_disorder, LATTICE_SIZE, HOPPING, anderson_rng)
        xi = localization_length(mipr, LATTICE_SIZE)
        noise_std = noise_from_xi(xi)

        communities.append({
            "id": ic,
            "alpha": float(alpha),
            "n_species": n_species_base,
            "shannon_h": float(h_prime),
            "evenness": float(evenness),
            "disorder_w": float(w_disorder),
            "mean_ipr": float(mipr),
            "loc_length_xi": float(xi),
            "noise_std": float(noise_std),
        })

    # Phase 2: generate data for all communities
    all_x = []
    all_y = []
    community_indices = []

    for comm in communities:
        data_rng = np.random.RandomState(SEED + comm["id"] * 3000)
        data = generate_community_data(N_SAMPLES_PER_COMMUNITY, comm["noise_std"], data_rng)
        for rec in data:
            all_x.append(rec[:5])
            all_y.append(rec[6])
            community_indices.append(comm["id"])

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    community_indices = np.array(community_indices)

    # Normalize globally
    x_mean = all_x.mean(axis=0)
    x_std = all_x.std(axis=0) + 1e-8
    y_mean = all_y.mean()
    y_std = all_y.std() + 1e-8
    x_norm = (all_x - x_mean) / x_std
    y_norm = (all_y - y_mean) / y_std

    # Phase 3: split train/test per community, train ONE pooled ESN
    train_mask = np.zeros(len(all_y), dtype=bool)
    for ic in range(N_COMMUNITIES):
        comm_idx = np.where(community_indices == ic)[0]
        n_train = int(len(comm_idx) * TRAIN_FRAC)
        train_mask[comm_idx[:n_train]] = True

    # Initialize ESN weights
    esn_rng = np.random.RandomState(SEED + 99999)
    rs = RESERVOIR_SIZE
    w_in = esn_rng.standard_normal((rs, INPUT_DIM)) * INPUT_SCALE
    w_res_raw = esn_rng.standard_normal((rs, rs)) / np.sqrt(rs)
    evals = np.linalg.eigvalsh(w_res_raw)
    sr = max(abs(evals.max()), abs(evals.min()))
    if sr > 1e-12:
        w_res = w_res_raw * (SPECTRAL_RADIUS / sr)
    else:
        w_res = w_res_raw
    b_res = esn_rng.standard_normal(rs) * 0.1

    # Drive reservoir on training data
    train_x = x_norm[train_mask]
    train_y = y_norm[train_mask]
    n_train_total = train_x.shape[0]

    H_train = np.zeros((n_train_total, rs))
    for i in range(n_train_total):
        H_train[i] = esn_reservoir_drive(train_x[i], w_in, w_res, b_res)

    # Ridge regression readout
    reg = H_train.T @ H_train + RIDGE_ALPHA * np.eye(rs)
    w_out = np.linalg.solve(reg, H_train.T @ train_y)

    # Phase 4: evaluate per-community test R²
    results = []
    all_w = []
    all_r2 = []
    all_xi = []
    all_ipr = []

    for comm in communities:
        ic = comm["id"]
        comm_idx = np.where(community_indices == ic)[0]
        test_idx = comm_idx[~train_mask[comm_idx]]

        if len(test_idx) == 0:
            continue

        test_h = np.zeros((len(test_idx), rs))
        for j, idx in enumerate(test_idx):
            test_h[j] = esn_reservoir_drive(x_norm[idx], w_in, w_res, b_res)

        y_pred_norm = test_h @ w_out
        y_pred = y_pred_norm * y_std + y_mean
        y_true = all_y[test_idx]

        r2 = r2_score(y_true, y_pred)
        rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))

        comm["r2_test"] = float(r2)
        comm["rmse_test"] = float(rmse_val)
        comm["n_test"] = int(len(test_idx))
        results.append(comm)

        all_w.append(comm["disorder_w"])
        all_r2.append(r2)
        all_xi.append(comm["loc_length_xi"])
        all_ipr.append(comm["mean_ipr"])

    all_w = np.array(all_w)
    all_r2 = np.array(all_r2)
    all_xi = np.array(all_xi)
    all_ipr = np.array(all_ipr)

    # Coupling metrics
    pearson_w_r2 = np.corrcoef(all_w, all_r2)[0, 1]
    pearson_xi_r2 = np.corrcoef(all_xi, all_r2)[0, 1]
    pearson_ipr_r2 = np.corrcoef(all_ipr, all_r2)[0, 1]

    # Pooled ESN stats
    all_test_idx = np.where(~train_mask)[0]
    H_all_test = np.zeros((len(all_test_idx), rs))
    for j, idx in enumerate(all_test_idx):
        H_all_test[j] = esn_reservoir_drive(x_norm[idx], w_in, w_res, b_res)
    y_all_pred = H_all_test @ w_out * y_std + y_mean
    pooled_r2 = r2_score(all_y[all_test_idx], y_all_pred)

    coupling = {
        "pearson_w_r2": float(pearson_w_r2),
        "pearson_xi_r2": float(pearson_xi_r2),
        "pearson_ipr_r2": float(pearson_ipr_r2),
        "pooled_r2_test": float(pooled_r2),
        "n_communities": N_COMMUNITIES,
        "w_range": [float(all_w.min()), float(all_w.max())],
        "r2_range": [float(all_r2.min()), float(all_r2.max())],
        "xi_range": [float(all_xi.min()), float(all_xi.max())],
    }

    # ESN weights for Rust validation
    esn_params = {
        "w_in": w_in.tolist(),
        "w_res": w_res.tolist(),
        "b_res": b_res.tolist(),
        "w_out": w_out.tolist(),
        "reservoir_size": RESERVOIR_SIZE,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }

    # Reference predictions for Rust parity
    ref_inputs = [
        [35.0, 7.2, 3.0, 20.0, 70.0],
        [55.0, 7.0, 2.0, 30.0, 65.0],
        [25.0, 5.8, 7.0, 8.0, 55.0],
        [45.0, 7.5, 1.0, 25.0, 80.0],
        [40.0, 6.5, 5.0, 15.0, 60.0],
        [30.0, 8.0, 0.8, 35.0, 85.0],
    ]
    ref_predictions = []
    norm_ref = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    for inp in ref_inputs:
        x_n = (np.array(inp) - norm_ref["x_mean"]) / norm_ref["x_std"]
        hr = esn_reservoir_drive(x_n, w_in, w_res, b_res)
        y_pred = float((w_out @ hr) * y_std + y_mean)
        y_analytical = biogas_yield(*inp)
        ref_predictions.append({
            "input": inp,
            "esn_yield": float(y_pred),
            "analytical_yield": float(y_analytical),
        })

    baseline = {
        "experiment": "096_digester_anderson_coupling",
        "seed": SEED,
        "lattice_size": LATTICE_SIZE,
        "reservoir_size": RESERVOIR_SIZE,
        "n_samples_per_community": N_SAMPLES_PER_COMMUNITY,
        "coupling": coupling,
        "communities": results,
        "esn": esn_params,
        "reference_predictions": ref_predictions,
    }

    json_path = "control/digester_anderson/digester_anderson_baseline.json"
    with open(json_path, "w") as f:
        json.dump(baseline, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # Validation checks
    # ═══════════════════════════════════════════════════════════════

    checks_pass = 0
    checks_total = 0

    def check(name, condition):
        nonlocal checks_pass, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_pass += 1
        print(f"  [{status}] {name}")

    print(f"\n{'='*60}")
    print("Experiment 096: Digester-Anderson Coupling")
    print(f"{'='*60}")
    print(f"  Communities: {N_COMMUNITIES}, species={n_species_base}")
    print(f"  Dirichlet α: {alpha_values[0]:.2f}–{alpha_values[-1]:.2f}")
    print(f"  W range: {all_w.min():.2f}–{all_w.max():.2f}")
    print(f"  ξ range: {all_xi.min():.4f}–{all_xi.max():.4f}")
    print(f"  R² range: {all_r2.min():.4f}–{all_r2.max():.4f}")
    print(f"  Pooled R²(test): {pooled_r2:.4f}")
    print(f"  Pearson r(W, R²) = {pearson_w_r2:.4f}")
    print(f"  Pearson r(ξ, R²) = {pearson_xi_r2:.4f}")
    print(f"  Pearson r(IPR, R²) = {pearson_ipr_r2:.4f}")
    print()

    # Anderson physics checks
    check("Anderson: high W → high mean IPR",
          all_ipr[all_w > np.median(all_w)].mean() > all_ipr[all_w <= np.median(all_w)].mean())

    check("Anderson: high W → low ξ",
          all_xi[all_w > np.median(all_w)].mean() < all_xi[all_w <= np.median(all_w)].mean())

    # Coupling checks
    check("Coupling: Pearson r(W, R²) < 0 (disorder hurts prediction)",
          pearson_w_r2 < 0)

    check("Coupling: Pearson r(ξ, R²) > 0 (loc length helps prediction)",
          pearson_xi_r2 > 0)

    check("Coupling: |r(W, R²)| > 0.3 (meaningful correlation)",
          abs(pearson_w_r2) > 0.3)

    # ESN quality checks
    check("Pooled ESN: R²(test) > 0.5",
          pooled_r2 > 0.5)

    check("Low-disorder community R²(test) > 0.5",
          all_r2[np.argmin(all_w)] > 0.5)

    check("High-W community R² < low-W community R²",
          all_r2[np.argmax(all_w)] < all_r2[np.argmin(all_w)])

    # Diversity-disorder mapping
    check("Evenness sweep: W monotonically decreases with α",
          all(results[i]["disorder_w"] >= results[i + 1]["disorder_w"] - 0.5
              for i in range(len(results) - 1)))

    # Determinism
    rng2 = np.random.RandomState(SEED)
    _, h2, _ = generate_community(n_species_base, alpha_values[0], rng2)
    check("Deterministic: seed=42 reproduces H'(community 0)",
          abs(h2 - results[0]["shannon_h"]) < 1e-10)

    # JSON roundtrip
    with open(json_path) as f:
        loaded = json.load(f)
    check("JSON roundtrip: communities + ESN preserved",
          len(loaded["communities"]) == N_COMMUNITIES and "esn" in loaded)

    print(f"\n=== digester_anderson: {checks_pass}/{checks_total} checks "
          f"{'PASS' if checks_pass == checks_total else 'FAIL'} ===")

    sys.exit(0 if checks_pass == checks_total else 1)


if __name__ == "__main__":
    main()
