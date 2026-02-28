#!/usr/bin/env python3
"""nF-03 Phase C: AlphaFold3 confidence head validation baselines.

Generates deterministic baselines for pLDDT, PAE, pDE, and ranking score
confidence heads used by AlphaFold3 for model quality assessment.

Reference: Abramson et al. "Accurate structure prediction for all molecules"
           Nature 630:493-500 (2024), Supplementary §5.9.

Seed: 42 (deterministic across all platforms).
"""
import json
import math
import numpy as np

SEED = 42
N_RES = 6
D_PAIR = 4
N_BINS_PAE = 8
N_BINS_PDE = 6
MAX_PAE = 31.75
MAX_PDE = 30.0

rng = np.random.default_rng(SEED)

# ─── helpers ────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(logits):
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()

def linear_head(x, w, b):
    """Linear projection: logit = x @ w + b."""
    return x @ w + b

# ─── pLDDT head ─────────────────────────────────────────────────────

def plddt_head(single_repr, w, b):
    """pLDDT: Linear → sigmoid → [0,1] per residue."""
    logits = single_repr @ w + b  # [n_res]
    return sigmoid(logits)

# ─── PAE head ────────────────────────────────────────────────────────

def pae_head(pair_repr, w, b, n_bins, max_dist=31.75):
    """PAE: pair → Linear → softmax → expected alignment error."""
    n_pairs = pair_repr.shape[0]
    bin_centers = np.linspace(0, max_dist, n_bins)

    logits = pair_repr @ w + b  # [n_pairs, n_bins]
    probs = np.zeros_like(logits)
    expected = np.zeros(n_pairs)

    for i in range(n_pairs):
        probs[i] = softmax(logits[i])
        expected[i] = np.sum(probs[i] * bin_centers)

    return expected, probs

# ─── pDE head ────────────────────────────────────────────────────────

def pde_head(pair_repr, w, b, n_bins, max_dist=30.0):
    """pDE: pair → Linear → softmax → predicted distance error."""
    return pae_head(pair_repr, w, b, n_bins, max_dist=max_dist)

# ─── Ranking score ───────────────────────────────────────────────────

def ranking_score(plddt, pae_expected, pde_expected,
                  w_plddt=0.5, w_pae=0.3, w_pde=0.2,
                  max_pae=31.75, max_pde=30.0):
    """Weighted combination of confidence metrics."""
    mean_plddt = np.mean(plddt)
    mean_pae = np.mean(pae_expected)
    mean_pde = np.mean(pde_expected)

    pae_score = max(0.0, 1.0 - mean_pae / max_pae)
    pde_score = max(0.0, 1.0 - mean_pde / max_pde)

    return w_plddt * mean_plddt + w_pae * pae_score + w_pde * pde_score

# ═══════════════════════════════════════════════════════════════════
# Generate baselines
# ═══════════════════════════════════════════════════════════════════

baselines = {}
n_pass = 0
n_fail = 0

def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  [PASS] {name}")
    else:
        n_fail += 1
        print(f"  [FAIL] {name}: {detail}")

# ─── pLDDT ───────────────────────────────────────────────────────────

print("=== pLDDT head ===")
single_repr = rng.standard_normal((N_RES, D_PAIR))
w_plddt = rng.standard_normal((D_PAIR,))
b_plddt = rng.standard_normal(())

plddt = plddt_head(single_repr, w_plddt, b_plddt)
check("pLDDT shape", plddt.shape == (N_RES,), f"got {plddt.shape}")
check("pLDDT in [0,1]", np.all((plddt >= 0) & (plddt <= 1)),
      f"min={plddt.min()}, max={plddt.max()}")
check("pLDDT not all same", plddt.max() - plddt.min() > 1e-6)

baselines["plddt_single_repr"] = single_repr.flatten().tolist()
baselines["plddt_w"] = w_plddt.tolist()
baselines["plddt_b"] = float(b_plddt)
baselines["plddt_values"] = plddt.tolist()

# ─── PAE ─────────────────────────────────────────────────────────────

print("\n=== PAE head ===")
pair_repr = rng.standard_normal((N_RES * N_RES, D_PAIR))
w_pae = rng.standard_normal((D_PAIR, N_BINS_PAE))
b_pae = rng.standard_normal((N_BINS_PAE,))

pae_expected, pae_probs = pae_head(pair_repr, w_pae, b_pae, N_BINS_PAE)
check("PAE expected shape", pae_expected.shape == (N_RES * N_RES,))
check("PAE probs shape", pae_probs.shape == (N_RES * N_RES, N_BINS_PAE))
check("PAE probs sum to 1",
      np.allclose(pae_probs.sum(axis=1), 1.0, atol=1e-10))
check("PAE expected non-negative", np.all(pae_expected >= 0))

baselines["pae_pair_repr"] = pair_repr.flatten().tolist()
baselines["pae_w"] = w_pae.flatten().tolist()
baselines["pae_b"] = b_pae.tolist()
baselines["pae_expected"] = pae_expected.tolist()
baselines["pae_probs"] = pae_probs.flatten().tolist()

# ─── pDE ─────────────────────────────────────────────────────────────

print("\n=== pDE head ===")
w_pde = rng.standard_normal((D_PAIR, N_BINS_PDE))
b_pde = rng.standard_normal((N_BINS_PDE,))

pde_expected, pde_probs = pde_head(pair_repr, w_pde, b_pde, N_BINS_PDE, MAX_PDE)
check("pDE expected shape", pde_expected.shape == (N_RES * N_RES,))
check("pDE probs shape", pde_probs.shape == (N_RES * N_RES, N_BINS_PDE))
check("pDE probs sum to 1",
      np.allclose(pde_probs.sum(axis=1), 1.0, atol=1e-10))
check("pDE expected non-negative", np.all(pde_expected >= 0))
check("pDE expected in range", np.all(pde_expected <= MAX_PDE * 1.01),
      f"max={pde_expected.max():.4f}")

baselines["pde_w"] = w_pde.flatten().tolist()
baselines["pde_b"] = b_pde.tolist()
baselines["pde_expected"] = pde_expected.tolist()
baselines["pde_probs"] = pde_probs.flatten().tolist()

# ─── Ranking score ───────────────────────────────────────────────────

print("\n=== Ranking score ===")
score = ranking_score(plddt, pae_expected, pde_expected)
check("Ranking score finite", np.isfinite(score))
check("Ranking score in [0,1]", 0.0 <= score <= 1.0,
      f"score={score:.6f}")

score_perfect = ranking_score(
    np.ones(N_RES), np.zeros(N_RES * N_RES), np.zeros(N_RES * N_RES))
check("Perfect structure → score 1.0", abs(score_perfect - 1.0) < 1e-10)

score_worst = ranking_score(
    np.zeros(N_RES), np.full(N_RES * N_RES, MAX_PAE),
    np.full(N_RES * N_RES, MAX_PDE))
check("Worst structure → score ~0", abs(score_worst) < 1e-10)

baselines["ranking_score"] = float(score)
baselines["ranking_perfect"] = float(score_perfect)
baselines["ranking_worst"] = float(score_worst)

# ─── Cross-head consistency ──────────────────────────────────────────

print("\n=== Cross-head consistency ===")
check("pLDDT mean in (0,1)", 0 < np.mean(plddt) < 1)
check("PAE mean in (0, max_pae)", 0 < np.mean(pae_expected) < MAX_PAE)
check("pDE mean in (0, max_pde)", 0 < np.mean(pde_expected) < MAX_PDE)

# ═══════════════════════════════════════════════════════════════════
# Save baselines
# ═══════════════════════════════════════════════════════════════════

baselines["n_res"] = N_RES
baselines["d_pair"] = D_PAIR
baselines["n_bins_pae"] = N_BINS_PAE
baselines["n_bins_pde"] = N_BINS_PDE
baselines["max_pde"] = MAX_PDE

out = "control/coral_forge/confidence_baselines.json"
with open(out, "w") as f:
    json.dump(baselines, f)

print(f"\n{'='*60}")
print(f"nF-03 Phase C: {n_pass} pass, {n_fail} fail")
if n_fail > 0:
    print("FAIL")
    exit(1)
print(f"Baselines written to {out}")
