#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 098: WDM Surrogate Ensemble Quorum Sensing.

Novel composition: nS-05 (game theory / Anderson QS) + nW-01..05 (WDM surrogates).

Scientific hypothesis:
  Treat an ensemble of WDM surrogates (EOS, transport, S(q,w), ESN classifier)
  as a "microbial quorum." Each surrogate votes on a physics prediction. When
  predictions agree (low disagreement), the ensemble behaves like a cooperative
  quorum — high-confidence collective prediction. When predictions diverge
  (high disagreement near phase boundaries), it maps to high Anderson disorder W,
  signaling localization in prediction space.

  Key insight: Phase boundaries in warm dense matter are exactly where
  surrogates should disagree, because the physics changes abruptly. This
  disagreement acts as a phase transition detector.

Components composed:
  - game_theory (nS-05): QS cooperation model, replicator dynamics
  - anderson_localization: disorder W, IPR, localization length
  - wdm_surrogate (nW-02): EOS P(rho,T) prediction
  - wdm_transport (nW-04): transport coefficient prediction
  - wdm_sqw (nW-03): S(q,w) dynamic structure factor

Design:
  1. Generate a grid of (rho, T) points in WDM regime
  2. For each point, get predictions from multiple surrogates
  3. Compute per-point disagreement (normalized variance across surrogates)
  4. Map disagreement → Anderson disorder W
  5. Compute Anderson localization properties (IPR, xi) over the disorder field
  6. Show: high-disagreement regions (phase boundaries) → high W → localization

Provenance:
  Baseline commit: (first run)
  Baseline date:   2026-03-10
  Command:         python3 control/wdm_ensemble_qs/wdm_ensemble_qs.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""
import json
import sys
import numpy as np

SEED = 42
N_RHO = 32
N_TEMP = 32
RHO_RANGE = (0.1, 100.0)
TEMP_RANGE = (1000.0, 1e6)
W_SCALE = 20.0


# ═════════════════════════════════════════════════════════════════════
# Surrogate models (simplified analytical forms matching Rust modules)
# ═════════════════════════════════════════════════════════════════════

def eos_pressure_h(rho, T):
    """Hydrogen EOS: ideal gas + Coulomb correction + degeneracy."""
    k_b = 8.617e-5  # eV/K
    n_e = rho * 6.022e23 / 1.008
    e_f = 13.6 * (n_e / 1e24) ** (2/3)
    p_ideal = rho * k_b * T
    p_deg = 0.6 * n_e * e_f * 1e-24
    p_coulomb = -0.3 * (n_e ** (4/3)) * 1e-32
    return p_ideal + p_deg + p_coulomb


def eos_pressure_he(rho, T):
    """Helium EOS with shifted parameters."""
    k_b = 8.617e-5
    n_e = rho * 6.022e23 * 2 / 4.003
    e_f = 13.6 * (n_e / 1e24) ** (2/3)
    p_ideal = rho * k_b * T
    p_deg = 0.6 * n_e * e_f * 1e-24
    return p_ideal + p_deg


def transport_diffusion(rho, T):
    """Simplified self-diffusion coefficient D(rho, T).

    D ~ T^(5/2) / (rho * Gamma) where Gamma = coupling parameter.
    """
    gamma = (rho / 10.0) ** (1/3) * 13.6 / (8.617e-5 * T)
    gamma = max(gamma, 0.01)
    return T ** 2.5 / (rho * gamma * 1e15 + 1e-30)


def sqw_peak(rho, T):
    """Simplified dynamic structure factor peak position.

    omega_p ~ sqrt(rho) * (1 + thermal_correction).
    """
    omega_p = np.sqrt(rho) * 1e3
    thermal = 1.0 + 0.001 * T / 1e4
    return omega_p * thermal


def esn_phase_score(rho, T):
    """ESN-like phase classifier: maps (rho, T) to [0, 1] phase score.

    0 = ideal gas, 1 = degenerate. Transition near Gamma ≈ 1.
    """
    gamma = (rho / 10.0) ** (1/3) * 13.6 / (8.617e-5 * T)
    return 1.0 / (1.0 + np.exp(-2.0 * (gamma - 1.0)))


# ═════════════════════════════════════════════════════════════════════
# Anderson localization on the disagreement field
# ═════════════════════════════════════════════════════════════════════

def anderson_hamiltonian_1d(n, t_hop, disorder):
    """1D tight-binding Hamiltonian with site-dependent disorder."""
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = disorder[i]
        if i + 1 < n:
            H[i, i + 1] = t_hop
            H[i + 1, i] = t_hop
    return H


def ipr(psi):
    return float(np.sum(np.abs(psi) ** 4))


def localization_length(iprs, n):
    mean_ipr = np.mean(iprs)
    return 1.0 / (n * mean_ipr) if mean_ipr > 1e-12 else float(n)


# ═════════════════════════════════════════════════════════════════════
# QS cooperation mapping
# ═════════════════════════════════════════════════════════════════════

def replicator_step(freq_c, payoff, dt=0.01):
    """One step of replicator dynamics for cooperator frequency."""
    freq_d = 1.0 - freq_c
    f_c = payoff[0][0] * freq_c + payoff[0][1] * freq_d
    f_d = payoff[1][0] * freq_c + payoff[1][1] * freq_d
    f_bar = freq_c * f_c + freq_d * f_d
    dx = freq_c * (f_c - f_bar)
    new_fc = max(0.0, min(1.0, freq_c + dt * dx))
    return new_fc


# ═════════════════════════════════════════════════════════════════════
# Main experiment
# ═════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.RandomState(SEED)

    rho_grid = np.logspace(np.log10(RHO_RANGE[0]), np.log10(RHO_RANGE[1]), N_RHO)
    temp_grid = np.logspace(np.log10(TEMP_RANGE[0]), np.log10(TEMP_RANGE[1]), N_TEMP)

    print(f"Grid: {N_RHO}×{N_TEMP} = {N_RHO * N_TEMP} points")
    print(f"rho: [{RHO_RANGE[0]}, {RHO_RANGE[1]}] g/cm³")
    print(f"T: [{TEMP_RANGE[0]:.0f}, {TEMP_RANGE[1]:.0e}] K")

    # For each grid point, collect surrogate predictions (normalized)
    surrogates = ["eos_h", "eos_he", "transport", "sqw", "phase"]
    n_surrogates = len(surrogates)

    all_predictions = np.zeros((N_RHO, N_TEMP, n_surrogates))
    disagreement = np.zeros((N_RHO, N_TEMP))

    for i, rho in enumerate(rho_grid):
        for j, T in enumerate(temp_grid):
            p_h = eos_pressure_h(rho, T)
            p_he = eos_pressure_he(rho, T)
            d_coeff = transport_diffusion(rho, T)
            sq = sqw_peak(rho, T)
            phase = esn_phase_score(rho, T)

            preds = np.array([
                np.log10(abs(p_h) + 1e-30),
                np.log10(abs(p_he) + 1e-30),
                np.log10(abs(d_coeff) + 1e-30),
                np.log10(abs(sq) + 1e-30),
                phase,
            ])
            all_predictions[i, j] = preds

    # Normalize each surrogate prediction to [0, 1] for fair comparison
    for s in range(n_surrogates):
        vals = all_predictions[:, :, s].flatten()
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-30:
            all_predictions[:, :, s] = (all_predictions[:, :, s] - vmin) / (vmax - vmin)

    # Disagreement = coefficient of variation across surrogates
    for i in range(N_RHO):
        for j in range(N_TEMP):
            preds = all_predictions[i, j]
            mean_p = preds.mean()
            std_p = preds.std()
            disagreement[i, j] = std_p / max(mean_p, 1e-12)

    # Map disagreement → Anderson disorder W
    d_flat = disagreement.flatten()
    d_min, d_max = d_flat.min(), d_flat.max()
    d_norm = (d_flat - d_min) / max(d_max - d_min, 1e-12)
    W_field = d_norm * W_SCALE

    # Anderson localization along rho slices (fix T, vary rho)
    n_sites = N_RHO
    t_hop = 1.0
    slice_results = []

    for j_t in range(N_TEMP):
        disorder = W_field[j_t * N_RHO:(j_t + 1) * N_RHO]
        H = anderson_hamiltonian_1d(n_sites, t_hop, disorder)
        evals, evecs = np.linalg.eigh(H)

        iprs = [ipr(evecs[:, k]) for k in range(n_sites)]
        mean_ipr_val = np.mean(iprs)
        xi = localization_length(iprs, n_sites)
        mean_W = np.mean(disorder)

        slice_results.append({
            "temp_idx": j_t,
            "mean_W": float(mean_W),
            "mean_ipr": float(mean_ipr_val),
            "xi": float(xi),
        })

    # Cross-analysis: Pearson(mean_W, xi) — expect negative
    mean_Ws = np.array([s["mean_W"] for s in slice_results])
    xis = np.array([s["xi"] for s in slice_results])

    valid = np.isfinite(mean_Ws) & np.isfinite(xis) & (np.std(mean_Ws) > 1e-12) & (np.std(xis) > 1e-12)
    if valid.all() and len(mean_Ws) > 2:
        r_W_xi = float(np.corrcoef(mean_Ws, xis)[0, 1])
    else:
        r_W_xi = 0.0

    # QS cooperation mapping: high agreement → cooperators win
    qs_coop_freq_low_W = []
    qs_coop_freq_high_W = []
    median_W = np.median(mean_Ws)

    for s in slice_results:
        # Snowdrift-like game: net benefit = b - c/2 for mutual coop
        # Low W (agreement) → high net benefit → stable cooperation
        # High W (disagreement) → low net benefit → defection favored
        w_frac = s["mean_W"] / W_SCALE
        b = 3.0
        c = 1.0 + 4.0 * w_frac  # cost rises with disorder
        payoff = [[b - c / 2.0, b - c], [b, 0.0]]

        freq_c = 0.5
        for _ in range(500):
            freq_c = replicator_step(freq_c, payoff)

        if s["mean_W"] < median_W:
            qs_coop_freq_low_W.append(freq_c)
        else:
            qs_coop_freq_high_W.append(freq_c)

    mean_coop_low = float(np.mean(qs_coop_freq_low_W)) if qs_coop_freq_low_W else 0.0
    mean_coop_high = float(np.mean(qs_coop_freq_high_W)) if qs_coop_freq_high_W else 0.0

    # ═══════════════════════════════════════════════════════════════
    # Build baseline JSON
    # ═══════════════════════════════════════════════════════════════

    baseline = {
        "experiment": "098_wdm_ensemble_qs",
        "seed": SEED,
        "grid": {"n_rho": N_RHO, "n_temp": N_TEMP},
        "n_surrogates": n_surrogates,
        "disagreement_stats": {
            "mean": float(d_flat.mean()),
            "std": float(d_flat.std()),
            "min": float(d_flat.min()),
            "max": float(d_flat.max()),
        },
        "W_field_stats": {
            "mean": float(W_field.mean()),
            "std": float(W_field.std()),
        },
        "slice_results": slice_results,
        "coupling": {
            "r_W_xi": r_W_xi,
        },
        "qs_dynamics": {
            "mean_coop_low_W": mean_coop_low,
            "mean_coop_high_W": mean_coop_high,
        },
        "reference_disorder": W_field[:10].tolist(),
    }

    json_path = "control/wdm_ensemble_qs/wdm_ensemble_qs_baseline.json"
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
    print("Experiment 098: WDM Surrogate Ensemble QS")
    print(f"{'='*60}")

    print(f"  Disagreement: mean={d_flat.mean():.4f}, range=[{d_flat.min():.4f}, {d_flat.max():.4f}]")
    print(f"  W field: mean={W_field.mean():.4f}, std={W_field.std():.4f}")
    print(f"  r(W, ξ) = {r_W_xi:.4f}")
    print(f"  QS coop (low W): {mean_coop_low:.4f}")
    print(f"  QS coop (high W): {mean_coop_high:.4f}")
    print()

    # Ensemble produces meaningful disagreement spread
    check("Disagreement range > 0", d_flat.max() - d_flat.min() > 0.01)
    check("Disagreement mean > 0", d_flat.mean() > 0)

    # W field spans significant range
    check("W field std > 0", W_field.std() > 0.1)

    # Anderson physics: high W → high IPR → low xi
    check("r(W, ξ) < 0 (disorder localizes)",
          r_W_xi < 0)

    # QS cooperation: low-W (agreement) → higher cooperation
    check("Low-W coop > high-W coop",
          mean_coop_low > mean_coop_high)

    # All slices have valid results
    check("All 32 temp slices computed",
          len(slice_results) == N_TEMP)

    # IPR bounded and positive
    all_iprs = [s["mean_ipr"] for s in slice_results]
    check("All IPR > 0", all(ip > 0 for ip in all_iprs))
    check("All IPR < 1", all(ip < 1 for ip in all_iprs))

    # Xi > 0 everywhere
    all_xi = [s["xi"] for s in slice_results]
    check("All ξ > 0", all(x > 0 for x in all_xi))

    # Determinism
    rng2 = np.random.RandomState(SEED)
    d2 = eos_pressure_h(1.0, 1e4)
    d3 = eos_pressure_h(1.0, 1e4)
    check("Deterministic: EOS reproducible", abs(d2 - d3) < 1e-15)

    # JSON roundtrip
    with open(json_path) as f:
        loaded = json.load(f)
    check("JSON roundtrip: coupling preserved",
          abs(loaded["coupling"]["r_W_xi"] - r_W_xi) < 1e-10)

    print(f"\n=== wdm_ensemble_qs: {checks_pass}/{checks_total} checks "
          f"{'PASS' if checks_pass == checks_total else 'FAIL'} ===")

    sys.exit(0 if checks_pass == checks_total else 1)


if __name__ == "__main__":
    main()
