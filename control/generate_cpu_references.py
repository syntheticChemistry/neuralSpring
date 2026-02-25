#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generate cross-language reference data for CPU math parity validation.

Produces control/cpu_parity_references.json with deterministic inputs
and expected outputs for every core kernel.  Rust reads this JSON and
verifies that its pure-CPU math produces the same numbers (within
CROSS_LANGUAGE tolerance).

Usage:
    python3 control/generate_cpu_references.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "cpu_parity_references.json")


# ── Primitives ────────────────────────────────────────────────────────


def gen_variance():
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    arr = np.array(data, dtype=np.float64)
    return {"data": data, "expected": float(np.var(arr, ddof=0))}


def gen_pearson():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.1, 3.9, 6.1, 7.9, 10.1]
    r = float(np.corrcoef(x, y)[0, 1])
    return {"x": x, "y": y, "expected": r}


def gen_chi_squared():
    observed = [10.0, 20.0, 30.0]
    expected = [20.0, 20.0, 20.0]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return {"observed": observed, "expected_vals": expected, "expected": chi2}


def gen_shannon_entropy():
    probs = [0.1, 0.2, 0.3, 0.4]
    entropy = -sum(p * math.log(p) for p in probs)
    return {"probs": probs, "expected": entropy}


def gen_softmax():
    logits = [1.0, 2.0, 3.0, 4.0]
    arr = np.array(logits, dtype=np.float64)
    shifted = arr - arr.max()
    exps = np.exp(shifted)
    result = (exps / exps.sum()).tolist()
    return {"logits": logits, "expected": result}


def gen_gelu():
    xs = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
    results = []
    for x in xs:
        results.append(0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))))
    return {"inputs": xs, "expected": results}


def gen_matmul():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    b = [16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    am = np.array(a, dtype=np.float64).reshape(4, 4)
    bm = np.array(b, dtype=np.float64).reshape(4, 4)
    cm = (am @ bm).flatten().tolist()
    return {"a": a, "b": b, "n": 4, "expected": cm}


def gen_frobenius():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    norm = float(np.linalg.norm(a))
    return {"a": a, "n": 3, "expected": norm}


def gen_l2_distance():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0]
    d = float(np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b))))
    return {"a": a, "b": b, "expected": d}


# ── Paper kernels ─────────────────────────────────────────────────────


def gen_hmm_forward():
    transition = [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.3, 0.5],
    ]
    emission = [
        [0.5, 0.4, 0.05, 0.05],
        [0.1, 0.3, 0.3, 0.3],
        [0.05, 0.1, 0.35, 0.5],
    ]
    initial = [0.6, 0.3, 0.1]
    observations = [0, 1, 2, 3, 0, 1, 2, 0, 3, 1, 0, 2, 1, 3, 0, 2, 3, 1, 0, 1]

    n_states = 3
    T = len(observations)
    A = np.array(transition, dtype=np.float64)
    B = np.array(emission, dtype=np.float64)
    pi = np.array(initial, dtype=np.float64)

    alpha = np.zeros((T, n_states), dtype=np.float64)
    scales = np.zeros(T, dtype=np.float64)
    alpha[0] = pi * B[:, observations[0]]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, observations[t]]
        scales[t] = alpha[t].sum()
        if scales[t] > 0:
            alpha[t] /= scales[t]
    log_likelihood = float(np.sum(np.log(scales + 1e-300)))

    return {
        "n_states": n_states,
        "n_obs_symbols": 4,
        "transition_flat": [x for row in transition for x in row],
        "emission_flat": [x for row in emission for x in row],
        "initial": initial,
        "observations": observations,
        "expected_log_likelihood": log_likelihood,
        "expected_final_alpha": alpha[-1].tolist(),
    }


def gen_replicator():
    b, c = 3.0, 1.0
    payoff = np.array([[b - c, -c], [b, 0.0]], dtype=np.float64)
    x = np.array([0.5, 0.5], dtype=np.float64)
    n_steps, dt = 1000, 0.001

    for _ in range(n_steps):
        fitness = payoff @ x
        avg_fitness = x @ fitness
        dx = x * (fitness - avg_fitness)
        x = x + dt * dx
        x = np.maximum(x, 0.0)
        x /= x.sum()

    return {
        "payoff_flat": payoff.flatten().tolist(),
        "initial": [0.5, 0.5],
        "n_steps": n_steps,
        "dt": dt,
        "expected_final": x.tolist(),
    }


def gen_commutator():
    a = [
        1.0, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8,
        0.5, 2.0, 0.7, 0.2, 0.3, 0.5, 0.1, 0.9,
        0.3, 0.7, 3.0, 0.4, 0.1, 0.6, 0.2, 0.7,
        0.1, 0.2, 0.4, 4.0, 0.5, 0.3, 0.8, 0.1,
        0.2, 0.3, 0.1, 0.5, 5.0, 0.7, 0.4, 0.6,
        0.4, 0.5, 0.6, 0.3, 0.7, 6.0, 0.9, 0.2,
        0.6, 0.1, 0.2, 0.8, 0.4, 0.9, 7.0, 0.3,
        0.8, 0.9, 0.7, 0.1, 0.6, 0.2, 0.3, 8.0,
    ]
    b = [
        8.0, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7,
        0.2, 7.0, 0.1, 0.5, 0.3, 0.4, 0.8, 0.6,
        0.4, 0.1, 6.0, 0.3, 0.7, 0.2, 0.9, 0.5,
        0.6, 0.5, 0.3, 5.0, 0.2, 0.8, 0.1, 0.4,
        0.1, 0.3, 0.7, 0.2, 4.0, 0.6, 0.3, 0.9,
        0.3, 0.4, 0.2, 0.8, 0.6, 3.0, 0.7, 0.1,
        0.5, 0.8, 0.9, 0.1, 0.3, 0.7, 2.0, 0.4,
        0.7, 0.6, 0.5, 0.4, 0.9, 0.1, 0.4, 1.0,
    ]
    am = np.array(a, dtype=np.float64).reshape(8, 8)
    bm = np.array(b, dtype=np.float64).reshape(8, 8)
    comm = am @ bm - bm @ am
    frob = float(np.linalg.norm(comm, "fro"))
    return {"a": a, "b": b, "dim": 8, "expected_frobenius": frob}


def gen_hamming():
    seqs = [
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ]
    n_seqs = len(seqs)
    seq_len = len(seqs[0])
    distances = []
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            diff = sum(1 for a, b in zip(seqs[i], seqs[j]) if a != b)
            distances.append(diff / seq_len)
    return {
        "seqs_flat": [x for s in seqs for x in s],
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "expected_distances": distances,
    }


def gen_jaccard():
    pa = [
        [1, 0, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ]
    n_genes = len(pa)
    n_genomes = len(pa[0])
    pa_np = np.array(pa, dtype=np.float64)
    distances = []
    for i in range(n_genomes):
        for j in range(i + 1, n_genomes):
            intersection = float(np.sum(pa_np[:, i] * pa_np[:, j]))
            union = float(np.sum(np.maximum(pa_np[:, i], pa_np[:, j])))
            d = 1.0 - intersection / union if union > 0 else 0.0
            distances.append(d)
    return {
        "pa_flat": [x for row in pa for x in row],
        "n_genes": n_genes,
        "n_genomes": n_genomes,
        "expected_distances": distances,
    }


def gen_pairwise_l2():
    n, dim = 5, 4
    features = [float(i) * 0.1 for i in range(n * dim)]
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                sum(
                    (features[i * dim + k] - features[j * dim + k]) ** 2
                    for k in range(dim)
                )
            )
            distances.append(d)
    return {
        "features": features,
        "n": n,
        "dim": dim,
        "expected_distances": distances,
    }


def gen_multi_obj():
    genotypes = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ]
    n_obj = 3
    expected = []
    for g in genotypes:
        arr = np.array(g, dtype=np.float64)
        n = len(arr)
        chunk = n // n_obj
        fitnesses = []
        for i in range(n_obj):
            start = i * chunk
            end = start + chunk if i < n_obj - 1 else n
            seg = arr[start:end]
            fitnesses.append(float(np.mean(seg) + 0.1 * np.std(seg, ddof=0)))
        expected.append(fitnesses)
    return {
        "genotypes": genotypes,
        "n_objectives": n_obj,
        "expected": expected,
    }


def gen_hill_gate():
    test_points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.3),
        (2.0, 2.0),
        (5.0, 5.0),
    ]
    vmax, k1, k2, n1, n2 = 1.0, 0.5, 0.3, 2.0, 2.0
    expected = []
    for cdg, ai in test_points:
        h1 = (cdg**n1) / (k1**n1 + cdg**n1 + 1e-30)
        h2 = (ai**n2) / (k2**n2 + ai**n2 + 1e-30)
        expected.append(vmax * h1 * h2)
    return {
        "test_points": test_points,
        "params": {"vmax": vmax, "k1": k1, "k2": k2, "n1": n1, "n2": n2},
        "expected": expected,
    }


def gen_swarm_nn():
    params = [float(i) / 33.0 for i in range(33)]
    senses = [0.0, 0.25, 0.5, 0.75, 1.0]
    expected_actions = []
    for sense in senses:
        p = np.array(params, dtype=np.float64)
        w1 = p[:4].reshape(1, 4)
        b1 = p[4:8]
        w2 = p[8:28].reshape(4, 5)
        b2 = p[28:33]

        def sigmoid(x):
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

        h = sigmoid(sense * w1 + b1)
        out = sigmoid(h @ w2 + b2)
        expected_actions.append(int(np.argmax(out)))
    return {
        "params": params,
        "senses": senses,
        "expected_actions": expected_actions,
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    refs = {
        "meta": {
            "generated_by": "control/generate_cpu_references.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
        "primitives": {
            "variance": gen_variance(),
            "pearson": gen_pearson(),
            "chi_squared": gen_chi_squared(),
            "shannon_entropy": gen_shannon_entropy(),
            "softmax": gen_softmax(),
            "gelu": gen_gelu(),
            "matmul": gen_matmul(),
            "frobenius": gen_frobenius(),
            "l2_distance": gen_l2_distance(),
        },
        "kernels": {
            "hmm_forward": gen_hmm_forward(),
            "replicator": gen_replicator(),
            "commutator": gen_commutator(),
            "hamming": gen_hamming(),
            "jaccard": gen_jaccard(),
            "pairwise_l2": gen_pairwise_l2(),
            "multi_objective": gen_multi_obj(),
            "hill_gate": gen_hill_gate(),
            "swarm_nn": gen_swarm_nn(),
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(refs, f, indent=2)

    n_prim = len(refs["primitives"])
    n_kern = len(refs["kernels"])
    print(f"Generated {OUT_PATH}")
    print(f"  {n_prim} primitives + {n_kern} kernels = {n_prim + n_kern} test groups")
    print(f"  Python {refs['meta']['python_version']}, NumPy {refs['meta']['numpy_version']}")


if __name__ == "__main__":
    main()
