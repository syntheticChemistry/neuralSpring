# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Global FST (Weir-Cockerham) for 6 populations × 20 individuals × 100 loci."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def generate_populations(n_pops, n_loci, n_individuals, fst_target, rng, temperatures):
    """Generate synthetic diploid genotype data for multiple populations."""
    ancestral_p = rng.beta(2, 2, n_loci)
    populations = []
    for pop_idx in range(n_pops):
        drift_scale = fst_target * 2.0
        p = ancestral_p + rng.normal(0, drift_scale, n_loci)
        temp_effect = (temperatures[pop_idx] - np.mean(temperatures)) * 0.002
        thermal_loci = slice(0, n_loci // 5)
        p[thermal_loci] += temp_effect
        p = np.clip(p, 0.01, 0.99)
        genotypes = np.zeros((n_individuals, n_loci), dtype=np.float64)
        for ind in range(n_individuals):
            for loc in range(n_loci):
                a1 = 1 if rng.random() < p[loc] else 0
                a2 = 1 if rng.random() < p[loc] else 0
                genotypes[ind, loc] = float(a1 + a2)
        populations.append(genotypes)
    return populations


def allele_frequencies(pop, n_individuals, n_loci):
    """Allele frequencies from diploid genotype matrix."""
    return np.mean(pop.reshape(n_individuals, n_loci), axis=0) / 2.0


def global_fst(populations, n_individuals_list, n_loci):
    """Weir-Cockerham FST estimator."""
    n_pops = len(populations)
    ns = np.array(n_individuals_list, dtype=float)
    n_total = np.sum(ns)
    n_bar = n_total / n_pops
    n_c = (n_total - np.sum(ns**2) / n_total) / (n_pops - 1)

    afs = []
    for i in range(n_pops):
        af = allele_frequencies(populations[i], n_individuals_list[i], n_loci)
        afs.append(af)
    afs = np.array(afs)

    p_bar = np.sum(ns[:, None] * afs, axis=0) / n_total

    s_sq = np.zeros(n_loci)
    for i in range(n_pops):
        s_sq += ns[i] * (afs[i] - p_bar) ** 2
    s_sq /= (n_pops - 1) * n_bar

    h_bar = np.zeros(n_loci)
    for i in range(n_pops):
        pop = populations[i].reshape(n_individuals_list[i], n_loci)
        het_count = np.sum(pop == 1, axis=0)
        h_bar += het_count
    h_bar /= n_total

    a = (n_bar / n_c) * (s_sq - (1 / (n_bar - 1)) * (p_bar * (1 - p_bar) - ((n_pops - 1) / n_pops) * s_sq - h_bar / 4))
    b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar) - ((n_pops - 1) / n_pops) * s_sq - (2 * n_bar - 1) / (4 * n_bar) * h_bar)
    c = h_bar / 2

    return float(np.sum(a) / np.sum(a + b + c))


def bench_fn(func, warmup=WARMUP, iters=ITERATIONS):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        func()
        timings.append(time.perf_counter_ns() - t0)
    timings.sort()
    return timings[len(timings) // 2] / 1000.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N_POPS = 6
    N_LOCI = 100
    N_INDIVIDUALS = 20

    temperatures = np.array([65.0, 70.0, 75.0, 80.0, 85.0, 90.0])
    populations = generate_populations(N_POPS, N_LOCI, N_INDIVIDUALS, 0.15, rng, temperatures)
    n_individuals_list = [N_INDIVIDUALS] * N_POPS

    def run():
        global_fst(populations, n_individuals_list, N_LOCI)

    median_us = bench_fn(run)

    print(f"META_GLOBAL_FST_6x20x100_US={median_us:.1f}")
    print()
    print(f"Python/NumPy meta-population FST benchmark — NumPy {np.__version__}")
    print(f"  Config: N_POPS={N_POPS}, N_INDIVIDUALS={N_INDIVIDUALS}, N_LOCI={N_LOCI}")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
