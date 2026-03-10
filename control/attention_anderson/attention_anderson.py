#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 100: Anderson Spectral Analysis of Attention Weight Matrices.

Novel composition: nS-01 (eigh_f64, BatchIprGpu) + coralForge (attention).

Scientific hypothesis:
  Self-attention weight matrices exhibit Anderson localization-like properties.
  High-quality attention (sharp, focused) has low IPR (extended eigenstates)
  and long localization length. Low-quality attention (noisy, diffuse) has
  high IPR (localized) and short localization length. This connects attention
  interpretability to condensed matter physics.

Design:
  1. Generate synthetic attention matrices of varying quality
  2. Symmetrize: A_sym = (A + A^T) / 2
  3. Compute eigendecomposition, IPR, localization length
  4. Show: attention quality correlates with spectral properties

Components composed:
  - anderson_localization: IPR, localization length
  - eigh: eigendecomposition
  - coral_forge::attention: SDPA concept
  - information_flow: attention_to_hamiltonian, spectral analysis

Provenance:
  Baseline commit: (first run)
  Baseline date:   2026-03-10
  Command:         python3 control/attention_anderson/attention_anderson.py
  Hardware:        Eastgate (i9-12900K, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""
import json
import sys
import numpy as np

SEED = 42
SEQ_LEN = 32
HEAD_DIM = 16
N_CONFIGS = 20


# ═════════════════════════════════════════════════════════════════════
# Attention generation
# ═════════════════════════════════════════════════════════════════════

def generate_attention(seq_len, head_dim, quality, rng):
    """Generate attention matrix with controllable quality.

    quality in [0, 1]: 0 = uniform/noisy, 1 = sharp/focused.
    """
    Q = rng.standard_normal((seq_len, head_dim)) * 0.1
    K = rng.standard_normal((seq_len, head_dim)) * 0.1

    scores = Q @ K.T / np.sqrt(head_dim)

    # Quality controls sharpness: high quality → high temperature → sharp peaks
    temperature = 0.1 + 5.0 * (1.0 - quality)
    scores_scaled = scores / max(temperature, 0.01)

    # Softmax
    exp_scores = np.exp(scores_scaled - scores_scaled.max(axis=-1, keepdims=True))
    attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-12)

    return attn


def attention_entropy(attn):
    """Average entropy of attention distributions (per-row)."""
    eps = 1e-12
    row_entropy = -np.sum(attn * np.log(attn + eps), axis=-1)
    return float(np.mean(row_entropy))


# ═════════════════════════════════════════════════════════════════════
# Anderson spectral analysis
# ═════════════════════════════════════════════════════════════════════

def spectral_analysis(attn):
    """Compute Anderson-like spectral properties of attention matrix."""
    n = attn.shape[0]
    A_sym = (attn + attn.T) / 2.0

    evals, evecs = np.linalg.eigh(A_sym)

    spectral_radius = np.max(np.abs(evals))

    iprs = []
    for k in range(n):
        psi = evecs[:, k]
        iprs.append(float(np.sum(psi**4)))
    mean_ipr = np.mean(iprs)

    participation = 1.0 / mean_ipr if mean_ipr > 1e-12 else float(n)
    xi = participation / n

    return {
        "spectral_radius": float(spectral_radius),
        "mean_ipr": float(mean_ipr),
        "participation": float(participation),
        "xi": float(xi),
        "eigenvalue_spread": float(evals[-1] - evals[0]),
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.RandomState(SEED)

    qualities = np.linspace(0.0, 1.0, N_CONFIGS)
    results = []

    for qi, q in enumerate(qualities):
        attn = generate_attention(SEQ_LEN, HEAD_DIM, q, np.random.RandomState(SEED + qi))
        entropy = attention_entropy(attn)
        spec = spectral_analysis(attn)

        results.append({
            "quality": float(q),
            "entropy": entropy,
            **spec,
        })

    # Correlation: quality vs spectral properties
    qs = np.array([r["quality"] for r in results])
    entropies = np.array([r["entropy"] for r in results])
    iprs = np.array([r["mean_ipr"] for r in results])
    xis = np.array([r["xi"] for r in results])
    partns = np.array([r["participation"] for r in results])

    r_q_entropy = float(np.corrcoef(qs, entropies)[0, 1]) if np.std(entropies) > 1e-12 else 0.0
    r_q_ipr = float(np.corrcoef(qs, iprs)[0, 1]) if np.std(iprs) > 1e-12 else 0.0
    r_q_xi = float(np.corrcoef(qs, xis)[0, 1]) if np.std(xis) > 1e-12 else 0.0
    r_entropy_ipr = float(np.corrcoef(entropies, iprs)[0, 1]) if np.std(entropies) > 1e-12 and np.std(iprs) > 1e-12 else 0.0

    # Reference: first attention matrix symmetrized for Rust parity
    ref_attn = generate_attention(SEQ_LEN, HEAD_DIM, 0.5, np.random.RandomState(SEED + 100))
    ref_sym = ((ref_attn + ref_attn.T) / 2.0).tolist()

    baseline = {
        "experiment": "100_attention_anderson",
        "seed": SEED,
        "seq_len": SEQ_LEN,
        "head_dim": HEAD_DIM,
        "n_configs": N_CONFIGS,
        "results": results,
        "correlations": {
            "r_quality_entropy": r_q_entropy,
            "r_quality_ipr": r_q_ipr,
            "r_quality_xi": r_q_xi,
            "r_entropy_ipr": r_entropy_ipr,
        },
        "reference_matrix": ref_sym,
    }

    json_path = "control/attention_anderson/attention_anderson_baseline.json"
    with open(json_path, "w") as f:
        json.dump(baseline, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # Checks
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
    print("Experiment 100: Attention Anderson Spectral Analysis")
    print(f"{'='*60}")

    print(f"  r(quality, entropy) = {r_q_entropy:.4f}")
    print(f"  r(quality, IPR)     = {r_q_ipr:.4f}")
    print(f"  r(quality, ξ)       = {r_q_xi:.4f}")
    print(f"  r(entropy, IPR)     = {r_entropy_ipr:.4f}")
    print()

    # High quality = low temperature = sharp attention = LOW entropy
    check("r(quality, entropy) < 0 (sharp attn → low entropy)", r_q_entropy < 0)

    # All spectral properties finite
    check("All results have finite IPR",
          all(np.isfinite(r["mean_ipr"]) for r in results))
    check("All results have finite ξ",
          all(np.isfinite(r["xi"]) for r in results))

    # IPR bounded
    check("All IPR > 0", all(r["mean_ipr"] > 0 for r in results))
    check("All IPR < 1", all(r["mean_ipr"] < 1 for r in results))

    # Participation > 1
    check("All participation > 1",
          all(r["participation"] > 1 for r in results))

    # Xi in (0, 1]
    check("All ξ in (0, 1]",
          all(0 < r["xi"] <= 1 for r in results))

    # N_CONFIGS results
    check(f"{N_CONFIGS} configs computed", len(results) == N_CONFIGS)

    # Determinism
    attn_check = generate_attention(SEQ_LEN, HEAD_DIM, 0.5, np.random.RandomState(SEED + 100))
    spec_check = spectral_analysis(attn_check)
    ref_spec = spectral_analysis(np.array(ref_sym))
    check("Deterministic spectral radius",
          abs(spec_check["spectral_radius"] - ref_spec["spectral_radius"]) < 1e-10)

    # JSON roundtrip
    with open(json_path) as f:
        loaded = json.load(f)
    check("JSON roundtrip: correlations preserved",
          abs(loaded["correlations"]["r_quality_entropy"] - r_q_entropy) < 1e-10)

    print(f"\n=== attention_anderson: {checks_pass}/{checks_total} checks "
          f"{'PASS' if checks_pass == checks_total else 'FAIL'} ===")

    sys.exit(0 if checks_pass == checks_total else 1)


if __name__ == "__main__":
    main()
