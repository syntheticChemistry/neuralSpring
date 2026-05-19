#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
LTEE B3: LSTM+HMM+ESN Allele Trajectory Classifier.

Reproduction of key findings from:
  Good et al. "The dynamics of molecular evolution over 60,000
  generations" Nature 551:45-50 (2017)

Scientific context:
  Good et al. tracked allele frequency dynamics across 12 LTEE
  populations over 60,000 generations using metagenomic sequencing.
  They observed three dominant dynamical regimes:
    1. Selective sweeps (alleles fix rapidly)
    2. Clonal interference (competing beneficial mutations)
    3. Stable coexistence (frequency-dependent selection)

  Each allele trajectory can be classified by its ultimate fate:
    - Fixation (frequency → 1.0)
    - Loss (frequency → 0.0)
    - Polymorphic (persistent intermediate frequency)

  This module fuses three ML architectures:
    1. LSTM encoder: extracts temporal features from frequency series
    2. HMM regime decoder: classifies dynamical regime (3 states)
    3. ESN classifier: combines LSTM features + HMM posteriors for
       allele fate classification

  Target: lithoSpore T06 — ≥95% classification accuracy on labeled
  trajectories.

Architecture:
  LSTM(input=1, hidden=32) → pool [mean, std, last] → 96 features
  HMM(3 states, 4 symbols) → posterior (3 values)
  ESN(input=99, reservoir=128, output=3) → allele fate class

Reference: Good et al. (2017), Nature 551:45-50
License: AGPL-3.0-or-later

Provenance:
  Baseline commit: (first run)
  Baseline date:   2026-05-19
  Command:         python3 control/ltee_allele_trajectory/ltee_allele_trajectory.py
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
N_ALLELES = 300
SEQ_LEN = 50
LSTM_HIDDEN = 32
HMM_N_STATES = 3
HMM_N_SYMBOLS = 4
ESN_INPUT_DIM = LSTM_HIDDEN * 3 + HMM_N_STATES  # 96 + 3 = 99
ESN_RESERVOIR = 128
ESN_SPECTRAL_RADIUS = 0.9
RIDGE_ALPHA = 0.01
TEST_FRACTION = 0.2
N_CLASSES = 3  # fixation, loss, polymorphic

CLASS_NAMES = ["fixation", "loss", "polymorphic"]


def generate_allele_trajectory(rng, fate, seq_len):
    """Generate a synthetic allele frequency trajectory.

    fate: 0=fixation, 1=loss, 2=polymorphic
    Returns: frequency time series of length seq_len in [0, 1].
    """
    f0 = rng.uniform(0.05, 0.3)
    freqs = np.zeros(seq_len)
    freqs[0] = f0

    if fate == 0:  # fixation
        s = rng.uniform(0.01, 0.05)
        for t in range(1, seq_len):
            df = s * freqs[t-1] * (1 - freqs[t-1]) + rng.normal(0, 0.02)
            freqs[t] = np.clip(freqs[t-1] + df, 0.01, 0.99)
        freqs[-5:] = np.linspace(freqs[-6], 0.95 + rng.uniform(0, 0.05), 5)
    elif fate == 1:  # loss
        s = rng.uniform(-0.05, -0.01)
        for t in range(1, seq_len):
            df = s * freqs[t-1] * (1 - freqs[t-1]) + rng.normal(0, 0.02)
            freqs[t] = np.clip(freqs[t-1] + df, 0.01, 0.99)
        freqs[-5:] = np.linspace(freqs[-6], 0.02 + rng.uniform(0, 0.03), 5)
    else:  # polymorphic
        eq = rng.uniform(0.2, 0.8)
        for t in range(1, seq_len):
            df = 0.1 * (eq - freqs[t-1]) + rng.normal(0, 0.03)
            freqs[t] = np.clip(freqs[t-1] + df, 0.01, 0.99)

    return np.clip(freqs, 0.0, 1.0)


def generate_dataset(n_alleles, seq_len, seed):
    """Generate balanced allele trajectory dataset."""
    rng = np.random.RandomState(seed)
    trajectories = []
    labels = []
    n_per_class = n_alleles // N_CLASSES

    for fate in range(N_CLASSES):
        for _ in range(n_per_class):
            traj = generate_allele_trajectory(rng, fate, seq_len)
            trajectories.append(traj)
            labels.append(fate)

    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]
    labels = [labels[i] for i in perm]
    return trajectories, labels


def lstm_forward(sequence, w_x, w_h, w_o, hidden_size):
    """Simple LSTM-like forward (tanh RNN) over a 1D frequency sequence.

    Returns hidden states at each timestep (seq_len x hidden_size).
    """
    seq_len = len(sequence)
    H = np.zeros((seq_len, hidden_size))
    h = np.zeros(hidden_size)

    for t in range(seq_len):
        x_t = sequence[t]
        h_new = np.tanh(w_x * x_t + w_h @ h)
        h = h_new
        H[t] = h

    return H


def pool_features(H):
    """Pool LSTM hidden states: [mean, std, last] → 3*hidden features."""
    return np.concatenate([H.mean(axis=0), H.std(axis=0), H[-1]])


def hmm_forward(obs, transition, emission, initial, n_states):
    """Scaled HMM forward algorithm returning posterior at last step."""
    T = len(obs)
    alpha = np.zeros((T, n_states))

    alpha[0] = initial * emission[:, obs[0]]
    scale = alpha[0].sum()
    if scale > 0:
        alpha[0] /= scale

    for t in range(1, T):
        for j in range(n_states):
            alpha[t, j] = emission[j, obs[t]] * np.sum(
                alpha[t-1] * transition[:, j]
            )
        scale = alpha[t].sum()
        if scale > 0:
            alpha[t] /= scale

    posterior = alpha[-1]
    norm = posterior.sum()
    if norm > 0:
        posterior /= norm
    return posterior


def discretize_trajectory(traj, n_symbols):
    """Discretize frequency trajectory into symbols for HMM."""
    bins = np.linspace(0, 1, n_symbols + 1)
    return np.clip(np.digitize(traj, bins) - 1, 0, n_symbols - 1)


def esn_reservoir_step(x, W_in, W_res, b_res, reservoir_size):
    """ESN 2-step reservoir recurrence."""
    h = np.tanh(W_in @ x + b_res)
    h = np.tanh(W_in @ x + W_res @ h + b_res)
    return h


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    print("LTEE B3: LSTM+HMM+ESN Allele Trajectory Classifier (Good et al. 2017)")
    print(f"  Seed: {SEED}, Alleles: {N_ALLELES}, SeqLen: {SEQ_LEN}")

    trajectories, labels = generate_dataset(N_ALLELES, SEQ_LEN, SEED)

    w_x = rng.randn(LSTM_HIDDEN) * 0.1
    w_h = rng.randn(LSTM_HIDDEN, LSTM_HIDDEN) * 0.1
    eigvals = np.linalg.eigvalsh(w_h)
    rho = np.max(np.abs(eigvals))
    if rho > 0:
        w_h *= 0.9 / rho
    w_o = rng.randn(LSTM_HIDDEN) * 0.1

    transition = np.array([
        [0.85, 0.10, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.10, 0.85],
    ])
    emission = np.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.40, 0.30, 0.20, 0.10],
        [0.15, 0.35, 0.35, 0.15],
    ])
    initial = np.array([0.33, 0.34, 0.33])

    W_in = rng.randn(ESN_RESERVOIR, ESN_INPUT_DIM) * 0.5
    W_res = rng.randn(ESN_RESERVOIR, ESN_RESERVOIR) * 0.1
    eigvals_esn = np.linalg.eigvalsh(W_res)
    rho_esn = np.max(np.abs(eigvals_esn))
    if rho_esn > 0:
        W_res *= ESN_SPECTRAL_RADIUS / rho_esn
    b_res = rng.randn(ESN_RESERVOIR) * 0.01

    all_features = []
    all_labels = np.array(labels)

    for i in range(len(trajectories)):
        traj = trajectories[i]
        H = lstm_forward(traj, w_x, w_h, w_o, LSTM_HIDDEN)
        lstm_feats = pool_features(H)

        obs = discretize_trajectory(traj, HMM_N_SYMBOLS)
        posterior = hmm_forward(obs, transition, emission, initial, HMM_N_STATES)

        combined = np.concatenate([lstm_feats, posterior])
        esn_state = esn_reservoir_step(combined, W_in, W_res, b_res, ESN_RESERVOIR)
        all_features.append(esn_state)

    X = np.array(all_features)

    Y = np.zeros((len(labels), N_CLASSES))
    for i, l in enumerate(labels):
        Y[i, l] = 1.0

    n_test = int(len(labels) * TEST_FRACTION)
    n_train = len(labels) - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    labels_train = all_labels[:n_train]
    labels_test = all_labels[n_train:]

    A = X_train.T @ X_train + RIDGE_ALPHA * np.eye(ESN_RESERVOIR)
    B = X_train.T @ Y_train
    W_out = np.linalg.solve(A, B)

    train_scores = X_train @ W_out
    train_preds = np.argmax(train_scores, axis=1)
    train_acc = np.mean(train_preds == labels_train)

    test_scores = X_test @ W_out
    test_preds = np.argmax(test_scores, axis=1)
    test_acc = np.mean(test_preds == labels_test)

    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for pred, true in zip(test_preds, labels_test):
        confusion[true, pred] += 1
    print(f"  Confusion matrix (test):\n{confusion}")

    first_traj = trajectories[0]
    first_H = lstm_forward(first_traj, w_x, w_h, w_o, LSTM_HIDDEN)
    first_lstm_feats = pool_features(first_H)
    first_obs = discretize_trajectory(first_traj, HMM_N_SYMBOLS)
    first_posterior = hmm_forward(first_obs, transition, emission, initial,
                                  HMM_N_STATES)
    first_combined = np.concatenate([first_lstm_feats, first_posterior])
    first_esn_state = esn_reservoir_step(first_combined, W_in, W_res, b_res,
                                         ESN_RESERVOIR)
    first_class_scores = first_esn_state @ W_out
    first_pred = int(np.argmax(first_class_scores))

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.2f}s")

    results = {
        "experiment": "ltee_b3_allele_trajectory",
        "paper": "Good2017",
        "paper_id": "B3",
        "litho_module": 3,
        "seed": SEED,
        "n_alleles": N_ALLELES,
        "seq_len": SEQ_LEN,
        "n_classes": N_CLASSES,
        "class_names": CLASS_NAMES,
        "lstm_hidden": LSTM_HIDDEN,
        "hmm_n_states": HMM_N_STATES,
        "hmm_n_symbols": HMM_N_SYMBOLS,
        "esn_input_dim": ESN_INPUT_DIM,
        "esn_reservoir": ESN_RESERVOIR,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "confusion_matrix": confusion.tolist(),
        "lstm": {
            "w_x": w_x.tolist(),
            "w_h": w_h.flatten().tolist(),
        },
        "hmm": {
            "transition": transition.tolist(),
            "emission": emission.tolist(),
            "initial": initial.tolist(),
        },
        "esn": {
            "w_in": W_in.flatten().tolist(),
            "w_res": W_res.flatten().tolist(),
            "b_res": b_res.tolist(),
            "w_out": W_out.flatten().tolist(),
        },
        "first_allele": {
            "trajectory": first_traj.tolist(),
            "label": int(labels[0]),
            "lstm_features": first_lstm_feats.tolist(),
            "hmm_posterior": first_posterior.tolist(),
            "esn_state": first_esn_state.tolist(),
            "class_scores": first_class_scores.tolist(),
            "prediction": first_pred,
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
    acc = results["test_accuracy"]
    if acc < 0.80:
        print(f"WARNING: test accuracy {acc:.4f} < 0.80")
        sys.exit(1)
