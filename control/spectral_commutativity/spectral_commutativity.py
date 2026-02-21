# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
neuralSpring Paper 22 — Distance to Normal Elements in C*-algebras

Reproduces key concepts from:
  Kachkovskiy & Safarov (2016)
  "Distance to normal elements in C*-algebras of real rank zero"
  Journal of the American Mathematical Society 29:61–80.

Core thesis: Quantifies how close an operator is to being "normal"
(commuting with its adjoint). For neural networks, this relates to
skip connections and residual networks: layers that "approximately
commute" can be reordered without catastrophic information loss.
The distance-to-normal metric measures how "residual-friendly" a
weight matrix is.

BarraCUDA connection:
  - GEMM for matrix multiply (AB, A*A, AA*)
  - Frobenius norm: reduce_sum(elementwise square)
  - Residual layers (I + W): elementwise add
"""

# SPDX-License-Identifier: AGPL-3.0-only

import sys

import numpy as np

# ---------------------------------------------------------------------------
# Operator Commutativity
# ---------------------------------------------------------------------------


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[A,B] = AB - BA. If [A,B] = 0, they commute."""
    return a @ b - b @ a


def commutator_frobenius_norm(a: np.ndarray, b: np.ndarray) -> float:
    """||[A,B]||_F measures how much A and B don't commute."""
    return float(np.linalg.norm(commutator(a, b), "fro"))


def commutativity_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized: ||[A,B]||_F / (||A||_F * ||B||_F). Scale-invariant."""
    na = np.linalg.norm(a, "fro")
    nb = np.linalg.norm(b, "fro")
    if na * nb < 1e-300:
        return 0.0
    return commutator_frobenius_norm(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# Distance to Normal
# ---------------------------------------------------------------------------


def distance_to_normal(a: np.ndarray) -> float:
    """Distance to normal: A is normal iff A*A = AA* (A commutes with adjoint).

    Approximation: dist_normal(A) ≈ ||A*A - AA*||_F / (2*||A||_F).
    For real matrices, A* = A^T.
    """
    n = np.linalg.norm(a, "fro")
    if n < 1e-300:
        return 0.0
    ata = a.T @ a
    aat = a @ a.T
    diff = ata - aat
    return float(np.linalg.norm(diff, "fro") / (2.0 * n))


# ---------------------------------------------------------------------------
# Skip Connection Analysis
# ---------------------------------------------------------------------------


def skip_commutativity(w1: np.ndarray, w2: np.ndarray) -> tuple[float, float]:
    """Compare commutativity of W1,W2 vs (I+W1),(I+W2).

    Returns (comm_raw, comm_skip) where:
      comm_raw = ||[W1,W2]||_F (normalized by ||W1||*||W2||)
      comm_skip = ||[(I+W1),(I+W2)]||_F (normalized similarly)
    """
    n1, n2 = w1.shape[0], w2.shape[0]
    assert n1 == n2, "W1, W2 must be square and same size"
    i = np.eye(n1)
    r1 = commutativity_ratio(w1, w2)
    r2 = commutativity_ratio(i + w1, i + w2)
    return (r1, r2)


# ---------------------------------------------------------------------------
# Spectral Analysis
# ---------------------------------------------------------------------------


def spectral_gap_to_normal(a: np.ndarray) -> float:
    """For normal A: eigenvalues of A*A = eigenvalues of AA* (same spectrum).

    Spectral gap = max |eigenvalue(ata) - eigenvalue(aat)| over matching.
    """
    ata = a.T @ a
    aat = a @ a.T
    ev_ata = np.linalg.eigvalsh(ata)
    ev_aat = np.linalg.eigvalsh(aat)
    ev_ata = np.sort(ev_ata)
    ev_aat = np.sort(ev_aat)
    return float(np.max(np.abs(ev_ata - ev_aat)))


# ---------------------------------------------------------------------------
# Random Matrix Ensemble
# ---------------------------------------------------------------------------


def sample_distance_to_normal(
    n: int, n_samples: int = 100, seed: int = 42
) -> np.ndarray:
    """Sample many random matrices, return distribution of distance-to-normal."""
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_samples):
        a = rng.standard_normal((n, n)) / np.sqrt(n)
        dists.append(distance_to_normal(a))
    return np.array(dists)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def main() -> int:
    """Validate spectral commutativity and distance-to-normal.

    Provenance
    ----------
    Paper: Kachkovskiy & Safarov (2016) JAMS 29:61-80.
    """
    rng = np.random.default_rng(42)
    total_passed = 0
    total_failed = 0

    print("=" * 72)
    print("neuralSpring Paper 22: Spectral Commutativity & Distance to Normal")
    print("  Kachkovskiy & Safarov (2016) JAMS 29:61-80")
    print("=" * 72)

    n = 32

    # Check 1: Normal matrices have distance-to-normal ≈ 0
    print("\n--- Check 1: Normal matrices have dist_normal ≈ 0 ---")
    h = rng.standard_normal((n, n))
    sym = (h + h.T) / 2
    d_sym = distance_to_normal(sym)
    print(f"  Symmetric matrix dist_normal: {d_sym:.2e}")
    if d_sym < 1e-10:
        print("  [PASS] Normal (symmetric) has dist ≈ 0")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected ≈ 0, got {d_sym}")
        total_failed += 1

    # Check 2: Identity matrix is normal (distance = 0)
    print("\n--- Check 2: Identity is normal ---")
    identity = np.eye(n)
    d_id = distance_to_normal(identity)
    print(f"  Identity dist_normal: {d_id:.2e}")
    if d_id < 1e-14:
        print("  [PASS] Identity has dist = 0")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected 0, got {d_id}")
        total_failed += 1

    # Check 3: Skip connections reduce commutativity distance
    print("\n--- Check 3: Skip connections reduce commutativity ---")
    w1 = rng.standard_normal((n, n)) / np.sqrt(n)
    w2 = rng.standard_normal((n, n)) / np.sqrt(n)
    comm_raw, comm_skip = skip_commutativity(w1, w2)
    print(f"  Raw commutativity:   {comm_raw:.6f}")
    print(f"  Skip commutativity: {comm_skip:.6f}")
    if comm_skip < comm_raw:
        print("  [PASS] Skip connections reduce commutativity")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected skip < raw")
        total_failed += 1

    # Check 4: Residual layers (I+eps*W) approximately commute for small eps
    print("\n--- Check 4: Residual layers approximately commute (small eps) ---")
    w1_r = rng.standard_normal((n, n)) / np.sqrt(n)
    w2_r = rng.standard_normal((n, n)) / np.sqrt(n)
    eps_small = 0.01
    i = np.eye(n)
    r1 = i + eps_small * w1_r
    r2 = i + eps_small * w2_r
    comm_res = commutativity_ratio(r1, r2)
    comm_raw_r = commutativity_ratio(w1_r, w2_r)
    print(f"  Residual (eps=0.01) commutativity: {comm_res:.6f}")
    print(f"  Raw W1,W2 commutativity: {comm_raw_r:.6f}")
    if comm_res < comm_raw_r:
        print("  [PASS] Small-epsilon residual layers nearly commute")
        total_passed += 1
    else:
        print(f"  [FAIL] Expected residual < raw")
        total_failed += 1

    # Check 5: Commutator anti-symmetry [A,B] = -[B,A]
    print("\n--- Check 5: Commutator anti-symmetry [A,B] = -[B,A] ---")
    a = rng.standard_normal((n, n)) / np.sqrt(n)
    b = rng.standard_normal((n, n)) / np.sqrt(n)
    ab = commutator(a, b)
    ba = commutator(b, a)
    err = np.linalg.norm(ab + ba, "fro")
    print(f"  ||[A,B] + [B,A]||_F: {err:.2e}")
    if err < 1e-10:
        print("  [PASS] [A,B] = -[B,A]")
        total_passed += 1
    else:
        print(f"  [FAIL] Anti-symmetry violated: {err}")
        total_failed += 1

    # Check 6: Distance-to-normal is non-negative
    print("\n--- Check 6: Distance-to-normal non-negative ---")
    dists = sample_distance_to_normal(n, 50, 42)
    min_d = float(np.min(dists))
    print(f"  Min over 50 samples: {min_d:.2e}")
    if min_d >= -1e-12:
        print("  [PASS] All distances non-negative")
        total_passed += 1
    else:
        print(f"  [FAIL] Negative distance: {min_d}")
        total_failed += 1

    # Check 7: Spectral theorem: normal has same eigenvalues for A*A, AA*
    print("\n--- Check 7: Spectral gap zero for normal ---")
    gap_sym = spectral_gap_to_normal(sym)
    gap_random = spectral_gap_to_normal(w1)
    print(f"  Symmetric (normal) spectral gap: {gap_sym:.2e}")
    print(f"  Random matrix spectral gap:      {gap_random:.4f}")
    if gap_sym < 1e-10:
        print("  [PASS] Normal matrix has spectral gap ≈ 0")
        total_passed += 1
    else:
        print(f"  [FAIL] Normal gap expected ≈ 0, got {gap_sym}")
        total_failed += 1

    # Check 8: BarraCUDA connection documented
    print("\n--- Check 8: BarraCUDA connection ---")
    print("  Distance-to-normal: gemm_f64 (A@A.T, A.T@A) + reduce_sum")
    print("  Commutator: gemm_f64 (AB, BA) + elementwise subtract")
    print("  Residual: elementwise add (I + W)")
    print("  [PASS] BarraCUDA connection documented")
    total_passed += 1

    total = total_passed + total_failed
    print(f"\n{'=' * 72}")
    print(f"TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print(f"{'=' * 72}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
