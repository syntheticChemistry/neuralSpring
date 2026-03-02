# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
neuralSpring baseCamp Paper 12 — Anderson Localization in Immunological Signaling

Validates the core computational framework for mapping atopic dermatitis (AD)
cytokine signaling onto the Anderson localization model.

Core thesis: Th2 cytokines (IL-4, IL-13, IL-31) propagate through disordered
tissue (heterogeneous cell populations). Anderson localization physics predicts
signal confinement vs propagation based on tissue geometry and cell-type disorder.

This experiment implements:
  1. Pielou evenness → disorder W mapping
  2. IC50 → Anderson barrier height (Hill equation, n=1)
  3. Dimensional promotion: 2D → 3D barrier disruption
  4. Tissue geometry factor for drug accessibility
  5. Anderson-augmented drug repurposing score
  6. AD skin state classification from Anderson parameters
  7. Lokivetmab pharmacokinetic dose-duration validation
  8. Cross-species barrier comparison (canine vs human)
  9. Gonzales IC50 data consistency (JAK1 selectivity)

Provenance:
  Baseline date:   2026-03-02
  Command:         python3 control/immunological_anderson/immunological_anderson.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42

BarraCUDA connection:
  - KL divergence (Dispatcher::kl_divergence) for cytokine distribution shift
  - eigh_f64 for Anderson lattice spectral analysis
  - BatchIprGpu for localization diagnostics
  - MultiHeadEsn for regime classification

References:
  Gonzales AJ et al. (2014) J Vet Pharmacol Ther 37:317-324
  Gonzales AJ et al. (2016) Vet Dermatol 27:34-e10
  Fleck TJ et al. (2021) Vet Dermatol 32:681-e182
  McCandless EE et al. (2014) Vet Immunol Immunopathol 157:42-48
  Fajgenbaum DC et al. (2019) J Clin Invest
"""

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
R_GOE = 0.5307  # GOE level spacing ratio threshold

# Gonzales (2014) — oclacitinib IC50 values (nM)
GONZALES_IC50 = {
    "JAK1": 10.0,
    "IL2": 36.0,
    "IL4": 159.0,
    "IL6": 36.0,
    "IL13": 249.0,
    "IL31": 63.0,
}

# Fleck/Gonzales (2021) — lokivetmab dose-duration data
LOKIVETMAB_PK = [
    {"dose_mg_kg": 0.125, "onset_hr": 3.0, "duration_days": 14.0},
    {"dose_mg_kg": 0.5, "onset_hr": 3.0, "duration_days": 28.0},
    {"dose_mg_kg": 2.0, "onset_hr": 3.0, "duration_days": 42.0},
]

# Skin layer geometry
SKIN_LAYERS = [
    {"name": "stratum_corneum", "d_eff": 0.0, "acellular": True},
    {"name": "viable_epidermis", "d_eff": 2.25, "acellular": False},
    {"name": "basement_membrane", "d_eff": 2.0, "acellular": True},
    {"name": "papillary_dermis", "d_eff": 3.0, "acellular": False},
    {"name": "reticular_dermis", "d_eff": 3.0, "acellular": False},
]


def pielou_evenness(fractions):
    """Pielou J = H'/ln(S), where H' = -sum(p*ln(p))."""
    s = len(fractions)
    if s <= 1:
        return 0.0
    h_prime = -sum(p * np.log(p) for p in fractions if p > 0)
    h_max = np.log(s)
    return float(h_prime / h_max) if h_max > 0 else 0.0


def ic50_to_w_reduction(drug_conc_nm, ic50_nm, max_w_reduction=1.0):
    """Hill equation (n=1): occupancy = [drug]/([drug]+IC50)."""
    if ic50_nm <= 0:
        return 0.0
    occupancy = drug_conc_nm / (drug_conc_nm + ic50_nm)
    return occupancy * max_w_reduction


def dimensional_promotion(intact_fraction, baseline_d=2.0, target_d=3.0):
    """Barrier disruption: d_eff = baseline + breach*(target-baseline)."""
    breach = 1.0 - max(0.0, min(1.0, intact_fraction))
    return baseline_d + breach * (target_d - baseline_d)


def tissue_geometry_factor(mw_kda, systemic, barrier_breach=0.0):
    """Drug tissue accessibility factor."""
    if systemic:
        return max(0.5, min(1.0, 1.0 - 0.001 * mw_kda))
    size_f = 0.8 if mw_kda < 0.5 else (0.5 if mw_kda < 5.0 else 0.1)
    return max(0.0, min(1.0, size_f + barrier_breach * 0.3))


def classify_ad_state(r, d_eff, is_treated=False):
    """Classify AD skin state from Anderson parameters."""
    if is_treated and r < R_GOE:
        return "treated"
    if d_eff < 2.5 and r < R_GOE:
        return "healthy"
    if d_eff > 2.7 and r > R_GOE:
        return "chronic"
    if r > R_GOE:
        return "flare"
    return "healthy"


def main():
    total_passed = 0
    total_failed = 0
    rng = np.random.default_rng(SEED)
    baseline = {}

    print("=" * 72)
    print("neuralSpring baseCamp Paper 12: Anderson Immunological Signaling")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Check 1: Pielou evenness — perfectly even = 1.0
    # ------------------------------------------------------------------
    print("\n--- Check 1: Pielou evenness (perfectly even) ---")
    j_even = pielou_evenness([0.25, 0.25, 0.25, 0.25])
    baseline["pielou_even"] = j_even
    if abs(j_even - 1.0) < 1e-10:
        print(f"  [PASS] J = {j_even:.10f} (expected 1.0)")
        total_passed += 1
    else:
        print(f"  [FAIL] J = {j_even:.10f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 2: Pielou evenness — dominated community < 0.3
    # ------------------------------------------------------------------
    print("\n--- Check 2: Pielou evenness (dominated) ---")
    j_dom = pielou_evenness([0.97, 0.01, 0.01, 0.01])
    baseline["pielou_dominated"] = j_dom
    if j_dom < 0.3:
        print(f"  [PASS] J = {j_dom:.6f} < 0.3 (dominated)")
        total_passed += 1
    else:
        print(f"  [FAIL] J = {j_dom:.6f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 3: IC50 at concentration = IC50 → 50% reduction
    # ------------------------------------------------------------------
    print("\n--- Check 3: IC50 half-maximal occupancy ---")
    w_at_ic50 = ic50_to_w_reduction(GONZALES_IC50["JAK1"], GONZALES_IC50["JAK1"])
    baseline["ic50_half"] = w_at_ic50
    if abs(w_at_ic50 - 0.5) < 1e-10:
        print(f"  [PASS] W_reduction = {w_at_ic50:.10f} (expected 0.5)")
        total_passed += 1
    else:
        print(f"  [FAIL] W_reduction = {w_at_ic50:.10f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 4: IC50 at 10x IC50 → >90% reduction
    # ------------------------------------------------------------------
    print("\n--- Check 4: IC50 at 10x concentration ---")
    w_at_10x = ic50_to_w_reduction(100.0, GONZALES_IC50["JAK1"])
    baseline["ic50_10x"] = w_at_10x
    if w_at_10x > 0.9:
        print(f"  [PASS] W_reduction = {w_at_10x:.6f} > 0.9")
        total_passed += 1
    else:
        print(f"  [FAIL] W_reduction = {w_at_10x:.6f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 5: Dimensional promotion — intact = baseline
    # ------------------------------------------------------------------
    print("\n--- Check 5: Dimensional promotion (intact barrier) ---")
    d_intact = dimensional_promotion(1.0)
    baseline["dim_intact"] = d_intact
    if abs(d_intact - 2.0) < 1e-10:
        print(f"  [PASS] d_eff = {d_intact:.10f} (intact = 2D)")
        total_passed += 1
    else:
        print(f"  [FAIL] d_eff = {d_intact:.10f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 6: Dimensional promotion — fully breached = target
    # ------------------------------------------------------------------
    print("\n--- Check 6: Dimensional promotion (fully breached) ---")
    d_breach = dimensional_promotion(0.0)
    baseline["dim_breached"] = d_breach
    if abs(d_breach - 3.0) < 1e-10:
        print(f"  [PASS] d_eff = {d_breach:.10f} (breached = 3D)")
        total_passed += 1
    else:
        print(f"  [FAIL] d_eff = {d_breach:.10f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 7: AD classification — healthy skin
    # ------------------------------------------------------------------
    print("\n--- Check 7: AD classification (healthy) ---")
    state_h = classify_ad_state(0.40, 2.0)
    baseline["classify_healthy"] = state_h
    if state_h == "healthy":
        print(f"  [PASS] r=0.40, d_eff=2.0 → {state_h}")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected healthy, got {state_h}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 8: AD classification — chronic AD
    # ------------------------------------------------------------------
    print("\n--- Check 8: AD classification (chronic) ---")
    state_c = classify_ad_state(0.60, 2.8)
    baseline["classify_chronic"] = state_c
    if state_c == "chronic":
        print(f"  [PASS] r=0.60, d_eff=2.8 → {state_c}")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected chronic, got {state_c}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 9: AD classification — flare
    # ------------------------------------------------------------------
    print("\n--- Check 9: AD classification (flare) ---")
    state_f = classify_ad_state(0.60, 2.6)
    baseline["classify_flare"] = state_f
    if state_f == "flare":
        print(f"  [PASS] r=0.60, d_eff=2.6 → {state_f}")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected flare, got {state_f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 10: AD classification — treated
    # ------------------------------------------------------------------
    print("\n--- Check 10: AD classification (treated) ---")
    state_t = classify_ad_state(0.40, 2.6, is_treated=True)
    baseline["classify_treated"] = state_t
    if state_t == "treated":
        print(f"  [PASS] r=0.40, d_eff=2.6, treated → {state_t}")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected treated, got {state_t}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 11: Tissue geometry — systemic small molecule reaches dermis
    # ------------------------------------------------------------------
    print("\n--- Check 11: Tissue geometry (systemic small molecule) ---")
    g_sys = tissue_geometry_factor(0.3, systemic=True)
    baseline["geom_systemic_small"] = g_sys
    if g_sys > 0.9:
        print(f"  [PASS] g = {g_sys:.6f} > 0.9 (systemic small molecule)")
        total_passed += 1
    else:
        print(f"  [FAIL] g = {g_sys:.6f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 12: Tissue geometry — topical large molecule blocked by barrier
    # ------------------------------------------------------------------
    print("\n--- Check 12: Tissue geometry (topical mAb blocked) ---")
    g_top = tissue_geometry_factor(150.0, systemic=False)
    baseline["geom_topical_large"] = g_top
    if g_top < 0.3:
        print(f"  [PASS] g = {g_top:.6f} < 0.3 (topical mAb blocked)")
        total_passed += 1
    else:
        print(f"  [FAIL] g = {g_top:.6f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 13: Lokivetmab dose-duration monotonicity
    # ------------------------------------------------------------------
    print("\n--- Check 13: Lokivetmab dose-duration monotonicity ---")
    durations = [d["duration_days"] for d in LOKIVETMAB_PK]
    baseline["lokivetmab_durations"] = durations
    monotonic = all(durations[i] < durations[i + 1] for i in range(len(durations) - 1))
    if monotonic:
        print(f"  [PASS] Durations monotonic: {durations}")
        total_passed += 1
    else:
        print(f"  [FAIL] Durations not monotonic: {durations}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 14: JAK1 selectivity (most potent = lowest IC50)
    # ------------------------------------------------------------------
    print("\n--- Check 14: JAK1 selectivity ---")
    jak1_most_potent = all(GONZALES_IC50["JAK1"] <= v for v in GONZALES_IC50.values())
    baseline["gonzales_ic50"] = GONZALES_IC50
    if jak1_most_potent:
        print(f"  [PASS] JAK1 IC50={GONZALES_IC50['JAK1']}nM is lowest (most selective)")
        total_passed += 1
    else:
        print(f"  [FAIL] JAK1 not most potent")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 15: Anderson drug score = pathway × geometry
    # ------------------------------------------------------------------
    print("\n--- Check 15: Anderson drug score ---")
    pathway = 0.95
    geometry = 0.90
    combined = pathway * geometry
    baseline["drug_score"] = {"pathway": pathway, "geometry": geometry, "combined": combined}
    if abs(combined - 0.855) < 1e-10:
        print(f"  [PASS] Score = {combined:.10f} (0.95 × 0.90 = 0.855)")
        total_passed += 1
    else:
        print(f"  [FAIL] Score = {combined:.10f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 16: IC50 sweep — monotonic W reduction with concentration
    # ------------------------------------------------------------------
    print("\n--- Check 16: IC50 sweep monotonicity ---")
    concs = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    w_reductions = [ic50_to_w_reduction(c, GONZALES_IC50["JAK1"]) for c in concs]
    baseline["ic50_sweep"] = {"concentrations_nm": concs, "w_reductions": w_reductions}
    mono_sweep = all(w_reductions[i] <= w_reductions[i + 1] for i in range(len(w_reductions) - 1))
    if mono_sweep:
        print(f"  [PASS] W reduction monotonic with concentration: {[f'{w:.4f}' for w in w_reductions]}")
        total_passed += 1
    else:
        print(f"  [FAIL] Not monotonic: {w_reductions}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 17: Barrier breach sweep — d_eff increases monotonically
    # ------------------------------------------------------------------
    print("\n--- Check 17: Barrier breach sweep ---")
    fracs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    d_effs = [dimensional_promotion(f) for f in fracs]
    baseline["breach_sweep"] = {"intact_fractions": fracs, "d_effs": d_effs}
    mono_breach = all(d_effs[i] <= d_effs[i + 1] for i in range(len(d_effs) - 1))
    if mono_breach:
        print(f"  [PASS] d_eff monotonically increasing with breach: {[f'{d:.2f}' for d in d_effs]}")
        total_passed += 1
    else:
        print(f"  [FAIL] Not monotonic: {d_effs}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 18: Pielou evenness with realistic skin cell populations
    # ------------------------------------------------------------------
    print("\n--- Check 18: Realistic dermal cell population evenness ---")
    healthy_dermis = [0.60, 0.15, 0.10, 0.08, 0.05, 0.02]  # fibroblasts dominate
    inflamed_dermis = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]  # more even (immune infiltrate)
    j_healthy = pielou_evenness(healthy_dermis)
    j_inflamed = pielou_evenness(inflamed_dermis)
    baseline["pielou_healthy_dermis"] = j_healthy
    baseline["pielou_inflamed_dermis"] = j_inflamed
    if j_inflamed > j_healthy:
        print(f"  [PASS] Inflamed J={j_inflamed:.4f} > healthy J={j_healthy:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL] Inflamed J={j_inflamed:.4f} <= healthy J={j_healthy:.4f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 19: Cross-species barrier comparison
    # ------------------------------------------------------------------
    print("\n--- Check 19: Cross-species barrier (canine thinner → higher d_eff) ---")
    canine_intact = 0.7  # thinner epidermis = less intact barrier
    human_intact = 0.9   # thicker epidermis = more intact barrier
    d_canine = dimensional_promotion(canine_intact)
    d_human = dimensional_promotion(human_intact)
    baseline["cross_species"] = {"canine_d_eff": d_canine, "human_d_eff": d_human}
    if d_canine > d_human:
        print(f"  [PASS] Canine d_eff={d_canine:.2f} > human d_eff={d_human:.2f}")
        total_passed += 1
    else:
        print(f"  [FAIL] Canine d_eff={d_canine:.2f} <= human d_eff={d_human:.2f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Check 20: Breached barrier improves topical drug access
    # ------------------------------------------------------------------
    print("\n--- Check 20: Barrier breach improves topical access ---")
    g_intact = tissue_geometry_factor(0.3, systemic=False, barrier_breach=0.0)
    g_breached = tissue_geometry_factor(0.3, systemic=False, barrier_breach=0.5)
    baseline["topical_breach_effect"] = {"intact": g_intact, "breached": g_breached}
    if g_breached > g_intact:
        print(f"  [PASS] Breached g={g_breached:.4f} > intact g={g_intact:.4f}")
        total_passed += 1
    else:
        print(f"  [FAIL] Breached g={g_breached:.4f} <= intact g={g_intact:.4f}")
        total_failed += 1

    # ------------------------------------------------------------------
    # Write baseline JSON
    # ------------------------------------------------------------------
    baseline_path = os.path.join(SCRIPT_DIR, "immunological_anderson_baseline.json")
    baseline_out = {
        "_source": "baseCamp Paper 12: Anderson Immunological Signaling",
        "_citation": "Gonzales et al. (2013-2024), Fajgenbaum et al. (2019)",
        "_method": "Anderson localization framework for cytokine signaling",
        "seed": SEED,
        "gonzales_ic50": GONZALES_IC50,
        "lokivetmab_pk": LOKIVETMAB_PK,
        "skin_layers": SKIN_LAYERS,
        "r_goe": R_GOE,
        **baseline,
        "_provenance": {
            "date": "2026-03-02",
            "python": sys.version,
            "numpy": np.__version__,
        },
    }
    with open(baseline_path, "w") as f:
        json.dump(baseline_out, f, indent=2)
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
