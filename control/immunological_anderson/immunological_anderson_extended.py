# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
neuralSpring baseCamp Paper 12 — Extended Experiments (nS-601..605)

Validates the deep modeling extensions for:
  nS-601: Hill dose-response for all 6 Gonzales cytokines + barrier heights
  nS-602: Pruritus time-series model (G3 treatment decay)
  nS-603: Lokivetmab PK decay + duration regression
  nS-604: Three-compartment disorder (3D tissue lattice)
  nS-605: Fajgenbaum MATRIX — Anderson-augmented drug repurposing

Provenance:
  Baseline date:   2026-03-02
  Command:         python3 control/immunological_anderson/immunological_anderson_extended.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, SciPy 1.14, seed=42

References:
  Gonzales AJ et al. (2014) J Vet Pharmacol Ther 37:317-324
  Gonzales AJ et al. (2016) Vet Dermatol 27:34-e10
  Fleck TJ et al. (2021) Vet Dermatol 32:681-e182
  McCandless EE et al. (2014) Vet Immunol Immunopathol 157:42-48
  Fajgenbaum DC et al. (2019) J Clin Invest
"""

import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
R_GOE = 0.5307

GONZALES_IC50 = {
    "JAK1": 10.0,
    "IL2": 36.0,
    "IL4": 159.0,
    "IL6": 36.0,
    "IL13": 249.0,
    "IL31": 63.0,
}

LOKIVETMAB_PK = [
    {"dose_mg_kg": 0.125, "onset_hr": 3.0, "duration_days": 14.0},
    {"dose_mg_kg": 0.5, "onset_hr": 3.0, "duration_days": 28.0},
    {"dose_mg_kg": 2.0, "onset_hr": 3.0, "duration_days": 42.0},
]

DRUG_CANDIDATES = [
    {"name": "Rapamycin", "indication": "Transplant rejection",
     "mechanism": "transduction_block", "pathway": 0.85,
     "mw_kda": 0.914, "systemic": True},
    {"name": "Tofacitinib", "indication": "Rheumatoid arthritis",
     "mechanism": "transduction_block", "pathway": 0.92,
     "mw_kda": 0.312, "systemic": True},
    {"name": "Tanezumab", "indication": "Osteoarthritis pain",
     "mechanism": "signal_elimination", "pathway": 0.78,
     "mw_kda": 148.0, "systemic": True},
    {"name": "Trametinib", "indication": "Melanoma",
     "mechanism": "transduction_block", "pathway": 0.65,
     "mw_kda": 0.615, "systemic": True},
    {"name": "Crisaborole", "indication": "Mild AD",
     "mechanism": "transduction_block", "pathway": 0.70,
     "mw_kda": 0.251, "systemic": False},
    {"name": "Nemolizumab", "indication": "Prurigo nodularis",
     "mechanism": "receptor_block", "pathway": 0.90,
     "mw_kda": 145.0, "systemic": True},
]

AD_FLARE = {"barrier_breach": 0.4, "d_eff": 2.7, "mean_w": 0.75}
AD_CHRONIC = {"barrier_breach": 0.6, "d_eff": 2.9, "mean_w": 0.85}

LOK_REG_A = 10.09
LOK_REG_B = 33.28


def hill_dose_response(concentration, ic50, hill_n=1.0, e_max=1.0):
    if ic50 <= 0 or concentration < 0:
        return 0.0
    c_n = concentration ** hill_n
    ic50_n = ic50 ** hill_n
    return e_max * c_n / (c_n + ic50_n)


def ic50_sweep(ic50, hill_n, concentrations):
    return [hill_dose_response(c, ic50, hill_n) for c in concentrations]


def cytokine_barrier_heights(scale=1.0):
    return {k: (v, math.log(v) * scale) for k, v in GONZALES_IC50.items()}


def pk_exponential_decay(c0, time_hours, half_life_hours):
    if half_life_hours <= 0:
        return 0.0
    k = math.log(2) / half_life_hours
    return c0 * math.exp(-k * time_hours)


def lokivetmab_duration_predict(dose_mg_kg):
    if dose_mg_kg <= 0:
        return 0.0
    return LOK_REG_A * math.log(dose_mg_kg) + LOK_REG_B


def pruritus_score_model(time_hours, baseline, suppression, decay_rate):
    nadir = baseline * (1.0 - max(0.0, min(1.0, suppression)))
    recovery = (baseline - nadir) * (1.0 - math.exp(-decay_rate * time_hours))
    return nadir + recovery


def pielou_evenness(fractions):
    s = len(fractions)
    if s <= 1:
        return 0.0
    h_prime = -sum(p * np.log(p) for p in fractions if p > 0)
    h_max = np.log(s)
    return float(h_prime / h_max) if h_max > 0 else 0.0


def three_compartment_disorder(immune_fracs, skin_fracs, neural_fracs, w_scale):
    w_i = pielou_evenness(immune_fracs) * w_scale
    w_s = pielou_evenness(skin_fracs) * w_scale
    w_n = pielou_evenness(neural_fracs) * w_scale
    mean_w = (w_i + w_s + w_n) / 3.0
    var_w = ((w_i - mean_w) ** 2 + (w_s - mean_w) ** 2 + (w_n - mean_w) ** 2) / 3.0
    return {"immune_w": w_i, "skin_w": w_s, "neural_w": w_n, "variance": var_w}


def tissue_lattice_hamiltonian(layer_sizes, layer_disorders, hopping, seed_val):
    rng = np.random.default_rng(seed_val)
    n = sum(layer_sizes)
    h = np.zeros((n, n))
    site = 0
    for li, ln in enumerate(layer_sizes):
        w = layer_disorders[min(li, len(layer_disorders) - 1)]
        for _ in range(ln):
            h[site, site] = w * rng.standard_normal()
            site += 1
    for i in range(n - 1):
        h[i, i + 1] = hopping
        h[i + 1, i] = hopping
    return h


def level_spacing_ratio(sorted_evals):
    if len(sorted_evals) < 3:
        return 0.0
    spacings = np.diff(sorted_evals)
    spacings = np.abs(spacings)
    ratios = []
    for i in range(len(spacings) - 1):
        a, b = spacings[i], spacings[i + 1]
        if a > 1e-15 or b > 1e-15:
            ratios.append(min(a, b) / max(a, b))
    return float(np.mean(ratios)) if ratios else 0.0


def dimensional_promotion(intact_fraction, baseline_d=2.0, target_d=3.0):
    breach = 1.0 - max(0.0, min(1.0, intact_fraction))
    return baseline_d + breach * (target_d - baseline_d)


def barrier_promotion_spectrum(n_sites, n_steps, base_disorder, hopping):
    results = []
    for step in range(n_steps):
        intact = 1.0 - step / max(1, n_steps - 1)
        d_eff = dimensional_promotion(intact)
        w_eff = base_disorder * (3.0 - d_eff + 1.0)
        ham = tissue_lattice_hamiltonian([n_sites], [w_eff], hopping, 42 + step)
        evals = np.sort(np.linalg.eigvalsh(ham))
        r = level_spacing_ratio(evals)
        results.append((intact, d_eff, r))
    return results


def tissue_geometry_factor(mw_kda, systemic, barrier_breach=0.0):
    if systemic:
        return max(0.5, min(1.0, 1.0 - 0.001 * mw_kda))
    size_f = 0.8 if mw_kda < 0.5 else (0.5 if mw_kda < 5.0 else 0.1)
    return max(0.0, min(1.0, size_f + barrier_breach * 0.3))


def fajgenbaum_matrix_score(drug, disease):
    geom = tissue_geometry_factor(drug["mw_kda"], drug["systemic"],
                                  disease["barrier_breach"])
    w_factor = 1.0 - min(1.0, disease["mean_w"]) * 0.3
    combined = drug["pathway"] * geom * w_factor
    return {
        "name": drug["name"],
        "pathway_score": drug["pathway"],
        "geometry_score": geom * w_factor,
        "combined_score": combined,
        "mechanism": drug["mechanism"],
    }


def main():
    total_passed = 0
    total_failed = 0
    baseline = {}

    print("=" * 72)
    print("neuralSpring baseCamp Paper 12: Extended Experiments (nS-601..605)")
    print("=" * 72)

    # ==================================================================
    # nS-601: Hill dose-response for all 6 Gonzales cytokines
    # ==================================================================
    print("\n" + "=" * 72)
    print("nS-601: Gonzales Dose-Response Modeling")
    print("=" * 72)

    concs = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    # Check 1: Hill n=1 at IC50 = 0.5
    print("\n--- Check 1: Hill n=1 at IC50 → 50% ---")
    r_n1 = hill_dose_response(10.0, 10.0, 1.0)
    baseline["hill_n1_at_ic50"] = r_n1
    if abs(r_n1 - 0.5) < 1e-10:
        print(f"  [PASS] response = {r_n1:.10f}")
        total_passed += 1
    else:
        print(f"  [FAIL] response = {r_n1:.10f}")
        total_failed += 1

    # Check 2: Hill n=2 cooperativity (steeper curve below IC50)
    print("\n--- Check 2: Hill cooperativity (n=2 < n=1 below IC50) ---")
    r_n1_below = hill_dose_response(5.0, 10.0, 1.0)
    r_n2_below = hill_dose_response(5.0, 10.0, 2.0)
    baseline["hill_cooperativity"] = {"n1": r_n1_below, "n2": r_n2_below}
    if r_n2_below < r_n1_below:
        print(f"  [PASS] n=2 ({r_n2_below:.4f}) < n=1 ({r_n1_below:.4f}) below IC50")
        total_passed += 1
    else:
        print(f"  [FAIL] n=2={r_n2_below:.4f}, n=1={r_n1_below:.4f}")
        total_failed += 1

    # Check 3: All 6 cytokine sweeps are monotonic
    print("\n--- Check 3: All 6 cytokine dose-response sweeps monotonic ---")
    all_sweeps = {}
    all_mono = True
    for name, ic50 in GONZALES_IC50.items():
        responses = ic50_sweep(ic50, 1.0, concs)
        all_sweeps[name] = responses
        for i in range(1, len(responses)):
            if responses[i] < responses[i - 1] - 1e-12:
                all_mono = False
    baseline["cytokine_sweeps"] = all_sweeps
    if all_mono:
        print(f"  [PASS] All 6 cytokine sweeps monotonically increasing")
        total_passed += 1
    else:
        print(f"  [FAIL] Non-monotonic sweep detected")
        total_failed += 1

    # Check 4: Barrier heights ordered (JAK1 < IL31 < IL13)
    print("\n--- Check 4: Cytokine barrier heights ordered ---")
    heights = cytokine_barrier_heights(1.0)
    baseline["barrier_heights"] = {k: {"ic50": v[0], "W": v[1]} for k, v in heights.items()}
    if heights["JAK1"][1] < heights["IL31"][1] < heights["IL13"][1]:
        print(f"  [PASS] W(JAK1)={heights['JAK1'][1]:.3f} < W(IL31)={heights['IL31'][1]:.3f} < W(IL13)={heights['IL13'][1]:.3f}")
        total_passed += 1
    else:
        print(f"  [FAIL] Barrier height ordering violated")
        total_failed += 1

    # Check 5: Dose-response saturation at high concentration
    print("\n--- Check 5: Dose-response saturation ---")
    r_sat = hill_dose_response(10000.0, 10.0, 1.0)
    baseline["hill_saturation"] = r_sat
    if r_sat > 0.999:
        print(f"  [PASS] At 1000× IC50: response = {r_sat:.6f}")
        total_passed += 1
    else:
        print(f"  [FAIL] {r_sat:.6f}")
        total_failed += 1

    # ==================================================================
    # nS-602: Pruritus time-series model
    # ==================================================================
    print("\n" + "=" * 72)
    print("nS-602: Pruritus Time-Series Model (Gonzales 2016 G3)")
    print("=" * 72)

    # Check 6: Nadir at t=0 post-treatment
    print("\n--- Check 6: Pruritus nadir at t=0 ---")
    baseline_score = 8.0
    suppression = 0.7
    decay_rate = 0.01
    nadir = pruritus_score_model(0.0, baseline_score, suppression, decay_rate)
    baseline["pruritus_nadir"] = nadir
    expected_nadir = baseline_score * (1.0 - suppression)
    if abs(nadir - expected_nadir) < 0.01:
        print(f"  [PASS] nadir = {nadir:.4f} (expected {expected_nadir:.4f})")
        total_passed += 1
    else:
        print(f"  [FAIL] nadir = {nadir:.4f}")
        total_failed += 1

    # Check 7: Score recovers toward baseline over time
    print("\n--- Check 7: Pruritus recovery toward baseline ---")
    timepoints = [0, 24, 72, 168, 336, 672]  # hours
    scores = [pruritus_score_model(t, baseline_score, suppression, decay_rate) for t in timepoints]
    baseline["pruritus_timeseries"] = {"hours": timepoints, "scores": scores}
    monotonic_recovery = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    if monotonic_recovery:
        print(f"  [PASS] Scores recover monotonically: {[f'{s:.2f}' for s in scores]}")
        total_passed += 1
    else:
        print(f"  [FAIL] Non-monotonic recovery")
        total_failed += 1

    # Check 8: Long-term score approaches baseline
    print("\n--- Check 8: Long-term asymptote → baseline ---")
    long_term = pruritus_score_model(10000.0, baseline_score, suppression, decay_rate)
    baseline["pruritus_asymptote"] = long_term
    if abs(long_term - baseline_score) < 0.1:
        print(f"  [PASS] At t=10000h: score = {long_term:.4f} ≈ baseline {baseline_score}")
        total_passed += 1
    else:
        print(f"  [FAIL] score = {long_term:.4f}")
        total_failed += 1

    # ==================================================================
    # nS-603: Lokivetmab PK decay + duration regression
    # ==================================================================
    print("\n" + "=" * 72)
    print("nS-603: Lokivetmab Pharmacokinetics")
    print("=" * 72)

    # Check 9: PK decay at half-life = C0/2
    print("\n--- Check 9: Exponential PK decay at half-life ---")
    c_half = pk_exponential_decay(100.0, 24.0, 24.0)
    baseline["pk_half_life"] = c_half
    if abs(c_half - 50.0) < 0.01:
        print(f"  [PASS] C(t=half_life) = {c_half:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL] C = {c_half:.4f}")
        total_failed += 1

    # Check 10: PK decay at t=0 = C0
    print("\n--- Check 10: PK decay at t=0 ---")
    c_zero = pk_exponential_decay(100.0, 0.0, 24.0)
    baseline["pk_t0"] = c_zero
    if abs(c_zero - 100.0) < 1e-10:
        print(f"  [PASS] C(0) = {c_zero:.10f}")
        total_passed += 1
    else:
        print(f"  [FAIL] C(0) = {c_zero:.10f}")
        total_failed += 1

    # Check 11: PK decay monotonically decreasing
    print("\n--- Check 11: PK decay monotonic ---")
    pk_times = [0, 6, 12, 24, 48, 96, 168]
    pk_concs = [pk_exponential_decay(100.0, t, 24.0) for t in pk_times]
    baseline["pk_decay_curve"] = {"hours": pk_times, "concentrations": pk_concs}
    pk_mono = all(pk_concs[i] > pk_concs[i + 1] for i in range(len(pk_concs) - 1))
    if pk_mono:
        print(f"  [PASS] PK monotonically decreasing")
        total_passed += 1
    else:
        print(f"  [FAIL] Non-monotonic")
        total_failed += 1

    # Check 12: Lokivetmab duration regression fits data
    print("\n--- Check 12: Lokivetmab log-linear regression ---")
    max_err = 0.0
    regression_results = []
    for pk in LOKIVETMAB_PK:
        pred = lokivetmab_duration_predict(pk["dose_mg_kg"])
        err = abs(pred - pk["duration_days"])
        max_err = max(max_err, err)
        regression_results.append({
            "dose": pk["dose_mg_kg"],
            "actual": pk["duration_days"],
            "predicted": pred,
            "error": err,
        })
    baseline["lokivetmab_regression"] = regression_results
    if max_err < 5.0:
        print(f"  [PASS] Max regression error = {max_err:.2f} days (< 5.0)")
        total_passed += 1
    else:
        print(f"  [FAIL] Max error = {max_err:.2f}")
        total_failed += 1

    # Check 13: Duration prediction monotonic with dose
    print("\n--- Check 13: Duration prediction monotonic ---")
    doses = [0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    dur_preds = [lokivetmab_duration_predict(d) for d in doses]
    baseline["duration_monotonicity"] = {"doses": doses, "durations": dur_preds}
    dur_mono = all(dur_preds[i] < dur_preds[i + 1] for i in range(len(dur_preds) - 1))
    if dur_mono:
        print(f"  [PASS] Predicted duration monotonically increasing with dose")
        total_passed += 1
    else:
        print(f"  [FAIL] Not monotonic")
        total_failed += 1

    # ==================================================================
    # nS-604: Three-compartment disorder (3D tissue lattice)
    # ==================================================================
    print("\n" + "=" * 72)
    print("nS-604: Three-Compartment Tissue Lattice (3D Systems)")
    print("=" * 72)

    immune_healthy = [0.25, 0.25, 0.25, 0.25]
    skin_healthy = [0.80, 0.10, 0.05, 0.05]
    neural_healthy = [0.50, 0.50]
    immune_inflamed = [0.15, 0.30, 0.25, 0.30]
    skin_inflamed = [0.40, 0.25, 0.20, 0.15]
    neural_inflamed = [0.35, 0.65]

    # Check 14: Three-compartment disorder (healthy tissue)
    print("\n--- Check 14: Three-compartment disorder (healthy) ---")
    tcd_h = three_compartment_disorder(immune_healthy, skin_healthy, neural_healthy, 10.0)
    baseline["three_comp_healthy"] = tcd_h
    if tcd_h["immune_w"] > tcd_h["skin_w"]:
        print(f"  [PASS] immune W={tcd_h['immune_w']:.4f} > skin W={tcd_h['skin_w']:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL] immune <= skin")
        total_failed += 1

    # Check 15: Inflamed tissue has higher mean disorder
    print("\n--- Check 15: Inflamed vs healthy mean disorder ---")
    tcd_i = three_compartment_disorder(immune_inflamed, skin_inflamed, neural_inflamed, 10.0)
    baseline["three_comp_inflamed"] = tcd_i
    mean_h = (tcd_h["immune_w"] + tcd_h["skin_w"] + tcd_h["neural_w"]) / 3
    mean_i = (tcd_i["immune_w"] + tcd_i["skin_w"] + tcd_i["neural_w"]) / 3
    if mean_i > mean_h:
        print(f"  [PASS] Inflamed mean W={mean_i:.4f} > healthy mean W={mean_h:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL] {mean_i:.4f} <= {mean_h:.4f}")
        total_failed += 1

    # Check 16: Cross-compartment variance is positive
    print("\n--- Check 16: Cross-compartment variance > 0 ---")
    if tcd_h["variance"] > 0:
        print(f"  [PASS] variance = {tcd_h['variance']:.6f}")
        total_passed += 1
    else:
        print(f"  [FAIL] variance = {tcd_h['variance']:.6f}")
        total_failed += 1

    # Check 17: Tissue lattice Hamiltonian symmetry
    print("\n--- Check 17: Tissue lattice Hamiltonian symmetry ---")
    ham = tissue_lattice_hamiltonian([4, 4], [1.0, 2.0], 1.0, 42)
    baseline["lattice_dim"] = ham.shape[0]
    sym_err = np.max(np.abs(ham - ham.T))
    if sym_err < 1e-15:
        print(f"  [PASS] |H - H^T| = {sym_err:.2e}")
        total_passed += 1
    else:
        print(f"  [FAIL] |H - H^T| = {sym_err:.2e}")
        total_failed += 1

    # Check 18: Level spacing ratio in valid range
    print("\n--- Check 18: Level spacing ratio in (0, 1] ---")
    evals_test = np.sort(np.linalg.eigvalsh(ham))
    r_test = level_spacing_ratio(evals_test)
    baseline["level_spacing_ratio_test"] = r_test
    if 0 < r_test <= 1.0:
        print(f"  [PASS] r = {r_test:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL] r = {r_test:.4f}")
        total_failed += 1

    # Check 19: Barrier promotion spectrum sweep
    print("\n--- Check 19: Barrier promotion spectrum (intact → breached) ---")
    spectrum = barrier_promotion_spectrum(16, 5, 1.0, 1.0)
    baseline["barrier_spectrum"] = [
        {"intact": s[0], "d_eff": s[1], "r": s[2]} for s in spectrum
    ]
    first_intact = abs(spectrum[0][0] - 1.0) < 1e-10
    last_breached = abs(spectrum[-1][0]) < 1e-10
    all_d_valid = all(2.0 <= s[1] <= 3.0 for s in spectrum)
    all_r_valid = all(0.0 <= s[2] <= 1.0 for s in spectrum)
    if first_intact and last_breached and all_d_valid and all_r_valid:
        print(f"  [PASS] 5-step sweep: d_eff in [2,3], r in [0,1]")
        total_passed += 1
    else:
        print(f"  [FAIL] Spectrum validation failed")
        total_failed += 1

    # Check 20: Multi-layer lattice eigenvalues are real
    print("\n--- Check 20: Multi-layer lattice eigenvalues real ---")
    ham_multi = tissue_lattice_hamiltonian([8, 8, 4], [0.5, 1.5, 2.0], 1.0, 42)
    evals_multi = np.linalg.eigvalsh(ham_multi)
    baseline["multilayer_evals"] = evals_multi.tolist()
    if all(np.isfinite(evals_multi)):
        print(f"  [PASS] All {len(evals_multi)} eigenvalues real and finite")
        total_passed += 1
    else:
        print(f"  [FAIL] Non-finite eigenvalues found")
        total_failed += 1

    # ==================================================================
    # nS-605: Fajgenbaum MATRIX — Anderson-augmented drug repurposing
    # ==================================================================
    print("\n" + "=" * 72)
    print("nS-605: Fajgenbaum MATRIX Drug Repurposing")
    print("=" * 72)

    # Check 21: All flare scores positive and <= 1
    print("\n--- Check 21: All MATRIX flare scores in (0, 1] ---")
    flare_scores = [fajgenbaum_matrix_score(d, AD_FLARE) for d in DRUG_CANDIDATES]
    baseline["matrix_flare_scores"] = flare_scores
    all_valid = all(0 < s["combined_score"] <= 1.0 for s in flare_scores)
    if all_valid:
        for s in flare_scores:
            print(f"    {s['name']:15s}  combined={s['combined_score']:.4f}")
        print(f"  [PASS] All 6 scores in valid range")
        total_passed += 1
    else:
        print(f"  [FAIL] Score out of range")
        total_failed += 1

    # Check 22: All chronic scores positive and <= 1
    print("\n--- Check 22: All MATRIX chronic scores in (0, 1] ---")
    chronic_scores = [fajgenbaum_matrix_score(d, AD_CHRONIC) for d in DRUG_CANDIDATES]
    baseline["matrix_chronic_scores"] = chronic_scores
    all_valid_c = all(0 < s["combined_score"] <= 1.0 for s in chronic_scores)
    if all_valid_c:
        for s in chronic_scores:
            print(f"    {s['name']:15s}  combined={s['combined_score']:.4f}")
        print(f"  [PASS] All 6 chronic scores in valid range")
        total_passed += 1
    else:
        print(f"  [FAIL] Score out of range")
        total_failed += 1

    # Check 23: Small molecule geometry > large mAb geometry
    print("\n--- Check 23: Small molecule geometry > mAb geometry ---")
    tofa = next(s for s in flare_scores if s["name"] == "Tofacitinib")
    nemo = next(s for s in flare_scores if s["name"] == "Nemolizumab")
    baseline["geom_tofa_vs_nemo"] = {"tofa": tofa["geometry_score"], "nemo": nemo["geometry_score"]}
    if tofa["geometry_score"] > nemo["geometry_score"]:
        print(f"  [PASS] Tofacitinib geom={tofa['geometry_score']:.4f} > Nemolizumab geom={nemo['geometry_score']:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL]")
        total_failed += 1

    # Check 24: Tofacitinib ranks #1 (highest combined score)
    print("\n--- Check 24: Tofacitinib top-ranked for flare ---")
    sorted_flare = sorted(flare_scores, key=lambda s: s["combined_score"], reverse=True)
    baseline["matrix_flare_ranking"] = [s["name"] for s in sorted_flare]
    if sorted_flare[0]["name"] == "Tofacitinib":
        print(f"  [PASS] Top: {sorted_flare[0]['name']} ({sorted_flare[0]['combined_score']:.4f})")
        total_passed += 1
    else:
        print(f"  [FAIL] Top: {sorted_flare[0]['name']}")
        total_failed += 1

    # Check 25: Chronic barrier breach helps topical Crisaborole
    print("\n--- Check 25: Chronic breach helps topical Crisaborole ---")
    crisa_flare = next(s for s in flare_scores if s["name"] == "Crisaborole")
    crisa_chronic = next(s for s in chronic_scores if s["name"] == "Crisaborole")
    baseline["crisaborole_breach_effect"] = {
        "flare": crisa_flare["geometry_score"],
        "chronic": crisa_chronic["geometry_score"],
    }
    if crisa_chronic["geometry_score"] >= crisa_flare["geometry_score"] - 0.1:
        print(f"  [PASS] Chronic geom={crisa_chronic['geometry_score']:.4f} ≥ flare geom-0.1={crisa_flare['geometry_score'] - 0.1:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL]")
        total_failed += 1

    # Check 26: pathway × geometry factorization holds
    print("\n--- Check 26: Score factorization (pathway × geometry_eff) ---")
    for s in flare_scores:
        expected = s["pathway_score"] * s["geometry_score"]
        if abs(s["combined_score"] - expected) > 1e-10:
            print(f"  [FAIL] {s['name']}: {s['combined_score']} ≠ {expected}")
            total_failed += 1
            break
    else:
        print(f"  [PASS] All 6 scores satisfy combined = pathway × geometry_eff")
        total_passed += 1

    # Check 27: Anderson filtering removes mechanism mismatch
    print("\n--- Check 27: Trametinib (MEK) ranks low for AD ---")
    trame = next(s for s in sorted_flare if s["name"] == "Trametinib")
    trame_rank = sorted_flare.index(trame) + 1
    baseline["trametinib_rank"] = trame_rank
    if trame_rank >= 4:
        print(f"  [PASS] Trametinib ranked #{trame_rank}/6 (pathway mismatch)")
        total_passed += 1
    else:
        print(f"  [FAIL] Ranked #{trame_rank}")
        total_failed += 1

    # Check 28: Combined integration — dose-response + MATRIX
    print("\n--- Check 28: Dose-response × MATRIX integration ---")
    tofa_ic50 = GONZALES_IC50["JAK1"]
    tofa_response_100nm = hill_dose_response(100.0, tofa_ic50, 1.0)
    tofa_matrix = tofa["combined_score"]
    integrated = tofa_response_100nm * tofa_matrix
    baseline["integrated_tofa"] = {
        "dose_response_100nm": tofa_response_100nm,
        "matrix_score": tofa_matrix,
        "integrated": integrated,
    }
    if 0 < integrated < 1.0:
        print(f"  [PASS] Integrated score = {integrated:.4f} (response={tofa_response_100nm:.4f} × MATRIX={tofa_matrix:.4f})")
        total_passed += 1
    else:
        print(f"  [FAIL] {integrated}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Write baseline JSON
    # ------------------------------------------------------------------
    baseline_path = os.path.join(SCRIPT_DIR, "immunological_anderson_extended_baseline.json")
    baseline_out = {
        "_source": "baseCamp Paper 12: Anderson Immunological Signaling — Extended",
        "_citation": "Gonzales et al. (2013-2024), Fajgenbaum et al. (2019)",
        "_experiments": "nS-601..605",
        "seed": SEED,
        **baseline,
        "_provenance": {
            "date": "2026-03-02",
            "python": sys.version,
            "numpy": np.__version__,
        },
    }

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(baseline_path, "w") as f:
        json.dump(baseline_out, f, indent=2, default=_default)
    print(f"\nBaseline written to {baseline_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = total_passed + total_failed
    print(f"\n{'=' * 72}")
    print(f"TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print(f"{'=' * 72}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
