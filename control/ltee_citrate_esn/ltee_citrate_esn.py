#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
LTEE B4: ESN Early-Warning Classifier for Citrate Innovation.

Reproduction of key findings from:
  Blount et al. "Historical contingency and the evolution of a key
  innovation in an experimental population of Escherichia coli"
  PNAS 105:7899-7906 (2008)

Scientific context:
  In Lenski's LTEE, only one of 12 populations (Ara-3) evolved the
  ability to metabolize citrate (Cit+) after ~31,500 generations.
  Blount's replay experiments showed this required "potentiating"
  mutations that accumulated ~2000 generations before Cit+ appeared.

  This module trains an ESN early-warning classifier on synthetic
  population trajectories to detect pre-potentiation regime shifts —
  generation windows where statistical signatures (rising variance,
  frequency fluctuations) predict an upcoming innovation event.

Architecture:
  ESN(input=4, reservoir=256, output=2) — binary early-warning
  (pre-potentiation vs normal). Ridge regression readout.

  Input features per generation window:
    1. Mean fitness (relative to ancestral)
    2. Fitness variance across clones
    3. Allele frequency entropy (Shannon H)
    4. Frequency change rate (delta-f per generation)

  Labels: 0 = normal, 1 = pre-potentiation (within WINDOW_GENS of Cit+)

Reference: Blount et al. (2008), PNAS 105:7899-7906
License: AGPL-3.0-or-later

Provenance:
  Baseline commit: (first run)
  Baseline date:   2026-05-19
  Command:         python3 control/ltee_citrate_esn/ltee_citrate_esn.py
  Hardware:        Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
N_TRAJECTORIES = 200
N_GENERATIONS = 100
INPUT_DIM = 4
RESERVOIR_SIZE = 256
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 0.5
RIDGE_ALPHA = 0.01
TEST_FRACTION = 0.2
WINDOW_GENS = 20

CIT_PLUS_GEN = 63
POTENTIATION_GEN = 43


def generate_trajectory(rng, has_potentiation):
    """Generate one synthetic LTEE population trajectory.

    Simulates generation-indexed features for a population that either
    undergoes potentiation → Cit+ innovation or remains normal.
    """
    fitness = np.ones(N_GENERATIONS)
    variance = np.zeros(N_GENERATIONS)
    entropy = np.zeros(N_GENERATIONS)
    delta_f = np.zeros(N_GENERATIONS)

    base_rate = 0.001
    for g in range(1, N_GENERATIONS):
        fitness[g] = fitness[g - 1] + base_rate + rng.normal(0, 0.0005)
        variance[g] = 0.001 + rng.exponential(0.001)
        n_alleles = max(2, int(3 + rng.poisson(1)))
        freqs = rng.dirichlet(np.ones(n_alleles))
        entropy[g] = -np.sum(freqs * np.log(freqs + 1e-10))
        delta_f[g] = rng.normal(0, 0.01)

    if has_potentiation:
        pot_start = POTENTIATION_GEN
        cit_gen = CIT_PLUS_GEN
        for g in range(pot_start, min(cit_gen, N_GENERATIONS)):
            t = (g - pot_start) / max(1, cit_gen - pot_start)
            variance[g] += 0.005 * t
            entropy[g] += 0.3 * t
            delta_f[g] += 0.02 * t + rng.normal(0, 0.005)
        if cit_gen < N_GENERATIONS:
            fitness[cit_gen:] += 0.05
            variance[cit_gen:] *= 0.5

    X = np.column_stack([fitness, variance, entropy, delta_f])
    return X


def generate_labels(has_potentiation):
    """Label each generation: 1 if within WINDOW_GENS before Cit+."""
    labels = np.zeros(N_GENERATIONS, dtype=int)
    if has_potentiation:
        start = max(0, CIT_PLUS_GEN - WINDOW_GENS)
        end = min(CIT_PLUS_GEN, N_GENERATIONS)
        labels[start:end] = 1
    return labels


def generate_dataset(n_traj, seed):
    """Generate training dataset of population trajectories.

    Half with potentiation (Ara-3-like), half without — balanced for
    binary classification training. Real LTEE ratio is 1/12, but
    balanced training improves early-warning sensitivity.
    """
    rng = np.random.RandomState(seed)
    all_X = []
    all_labels = []
    n_positive = n_traj // 2

    for i in range(n_traj):
        has_pot = i < n_positive
        X = generate_trajectory(rng, has_pot)
        labels = generate_labels(has_pot)
        all_X.append(X)
        all_labels.append(labels)

    return all_X, all_labels


def esn_reservoir_drive(X_seq, W_in, W_res, b_res, reservoir_size):
    """Drive ESN reservoir over a generation sequence (2-step recurrence)."""
    n_steps = X_seq.shape[0]
    H = np.zeros((n_steps, reservoir_size))
    for t in range(n_steps):
        x = X_seq[t]
        h = np.tanh(W_in @ x + b_res)
        h = np.tanh(W_in @ x + W_res @ h + b_res)
        H[t] = h
    return H


def ridge_regression(H_train, y_train, alpha=RIDGE_ALPHA):
    """Ridge regression readout: w = (H'H + aI)^-1 H'y."""
    n_feat = H_train.shape[1]
    A = H_train.T @ H_train + alpha * np.eye(n_feat)
    b = H_train.T @ y_train
    w = np.linalg.solve(A, b)
    return w


def esn_classify(H, w_out, threshold=0.5):
    """Binary classification from reservoir states."""
    scores = H @ w_out
    return (scores > threshold).astype(int), scores


def early_warning_metrics(predictions, labels):
    """Compute early-warning detection metrics."""
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    n_pos = max(tp + fn, 1)
    n_neg = max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(labels), 1)
    tpr = tp / n_pos
    fpr = fp / n_neg
    precision = tp / max(tp + fp, 1)
    return {
        "accuracy": float(accuracy),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    print("LTEE B4: ESN Early-Warning Classifier (Blount et al. 2008)")
    print(f"  Seed: {SEED}, Trajectories: {N_TRAJECTORIES}, "
          f"Reservoir: {RESERVOIR_SIZE}")

    trajectories, labels_list = generate_dataset(N_TRAJECTORIES, SEED)

    perm = rng.permutation(N_TRAJECTORIES)
    trajectories = [trajectories[i] for i in perm]
    labels_list = [labels_list[i] for i in perm]

    W_in = rng.randn(RESERVOIR_SIZE, INPUT_DIM) * INPUT_SCALE
    W_res = rng.randn(RESERVOIR_SIZE, RESERVOIR_SIZE) * 0.1
    eigvals = np.linalg.eigvalsh(W_res)
    rho = np.max(np.abs(eigvals))
    if rho > 0:
        W_res *= SPECTRAL_RADIUS / rho
    b_res = rng.randn(RESERVOIR_SIZE) * 0.01

    n_test = int(N_TRAJECTORIES * TEST_FRACTION)
    n_train = N_TRAJECTORIES - n_test

    H_train_all, y_train_all = [], []
    for i in range(n_train):
        H = esn_reservoir_drive(trajectories[i], W_in, W_res, b_res,
                                RESERVOIR_SIZE)
        H_train_all.append(H)
        y_train_all.append(labels_list[i])

    H_train_cat = np.vstack(H_train_all)
    y_train_cat = np.concatenate(y_train_all)

    w_out = ridge_regression(H_train_cat, y_train_cat, RIDGE_ALPHA)

    train_preds, train_scores = esn_classify(H_train_cat, w_out)
    train_metrics = early_warning_metrics(train_preds, y_train_cat)
    print(f"  Train: accuracy={train_metrics['accuracy']:.4f}, "
          f"TPR={train_metrics['tpr']:.4f}, FPR={train_metrics['fpr']:.4f}")

    H_test_all, y_test_all = [], []
    for i in range(n_train, N_TRAJECTORIES):
        H = esn_reservoir_drive(trajectories[i], W_in, W_res, b_res,
                                RESERVOIR_SIZE)
        H_test_all.append(H)
        y_test_all.append(labels_list[i])

    H_test_cat = np.vstack(H_test_all)
    y_test_cat = np.concatenate(y_test_all)

    test_preds, test_scores = esn_classify(H_test_cat, w_out)
    test_metrics = early_warning_metrics(test_preds, y_test_cat)
    print(f"  Test:  accuracy={test_metrics['accuracy']:.4f}, "
          f"TPR={test_metrics['tpr']:.4f}, FPR={test_metrics['fpr']:.4f}")

    first_traj_H = esn_reservoir_drive(trajectories[0], W_in, W_res, b_res,
                                       RESERVOIR_SIZE)
    first_preds, first_scores = esn_classify(first_traj_H, w_out)

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.2f}s")

    results = {
        "experiment": "ltee_b4_citrate_esn",
        "paper": "Blount2008",
        "paper_id": "B4",
        "litho_module": 4,
        "seed": SEED,
        "n_trajectories": N_TRAJECTORIES,
        "n_generations": N_GENERATIONS,
        "input_dim": INPUT_DIM,
        "reservoir_size": RESERVOIR_SIZE,
        "spectral_radius": SPECTRAL_RADIUS,
        "input_scale": INPUT_SCALE,
        "ridge_alpha": RIDGE_ALPHA,
        "window_gens": WINDOW_GENS,
        "cit_plus_gen": CIT_PLUS_GEN,
        "potentiation_gen": POTENTIATION_GEN,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "w_in": W_in.flatten().tolist(),
        "w_res": W_res.flatten().tolist(),
        "b_res": b_res.tolist(),
        "w_out": w_out.tolist(),
        "first_trajectory": {
            "features": trajectories[0].tolist(),
            "labels": labels_list[0].tolist(),
            "predictions": first_preds.tolist(),
            "scores": first_scores.tolist(),
        },
        "elapsed_seconds": elapsed,
    }

    out_path = os.path.join(SCRIPT_DIR, "expected_values.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote: {out_path}")
    return results


if __name__ == "__main__":
    results = main()
    acc = results["test_metrics"]["accuracy"]
    if acc < 0.80:
        print(f"WARNING: test accuracy {acc:.4f} < 0.80")
        sys.exit(1)
