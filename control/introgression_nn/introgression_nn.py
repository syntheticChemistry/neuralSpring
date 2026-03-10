#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 099: HMM Introgression Detection on Neural Network Weight Layers.

Novel composition: nS-04 (introgression.rs / HMM) + weight_spectral (pretrained).

Scientific hypothesis:
  Neural network layers can be analyzed as "genomic loci" in an introgression
  framework. Adjacent layers that share similar weight statistics are "concordant"
  (homologous). Layers with abrupt statistical shifts are "introgression-like"
  (knowledge transfer from a different source). This detects anomalous layers
  in a trained network — analogous to detecting gene flow between species.

Design:
  1. Generate synthetic "layer statistics" for a deep network
  2. Inject "introgression" at specific layers (abrupt weight distribution shift)
  3. Run HMM Viterbi decoding to detect introgressed layers
  4. Validate: detection rate, false positive rate, localization accuracy

Components composed:
  - introgression (Paper 018): PhyloNet-HMM, Viterbi decoding
  - hmm: Forward, backward, Viterbi algorithms
  - weight_spectral (nS-01): weight matrix spectral analysis concept

Provenance:
  Baseline commit: (first run)
  Baseline date:   2026-03-10
  Command:         python3 control/introgression_nn/introgression_nn.py
  Hardware:        Eastgate (i9-12900K, Pop!_OS 22.04)
  Environment:     Python 3.10.12, NumPy 2.2.6, seed=42
"""
import json
import sys
import numpy as np

SEED = 42
N_LAYERS = 100
INTROGRESSION_LAYERS = list(range(20, 30)) + list(range(50, 62)) + list(range(78, 85))
N_STATES = 2
N_OBS = 3


# ═════════════════════════════════════════════════════════════════════
# HMM primitives
# ═════════════════════════════════════════════════════════════════════

def build_hmm():
    """2-state HMM: state 0 = normal, state 1 = introgressed."""
    transition = np.array([[0.92, 0.08], [0.08, 0.92]])
    emission = np.array([
        [0.80, 0.15, 0.05],  # normal → strongly obs 0
        [0.05, 0.10, 0.85],  # introgressed → strongly obs 2
    ])
    initial = np.array([0.80, 0.20])
    return transition, emission, initial


def viterbi(observations, transition, emission, initial):
    """Standard Viterbi algorithm."""
    n = len(observations)
    n_states = len(initial)
    delta = np.zeros((n, n_states))
    psi = np.zeros((n, n_states), dtype=int)

    delta[0] = np.log(initial + 1e-30) + np.log(emission[:, observations[0]] + 1e-30)

    for t in range(1, n):
        for j in range(n_states):
            scores = delta[t - 1] + np.log(transition[:, j] + 1e-30)
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + np.log(emission[j, observations[t]] + 1e-30)

    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def forward_log_likelihood(observations, transition, emission, initial):
    """Log-likelihood via forward algorithm."""
    n = len(observations)
    n_states = len(initial)
    alpha = np.log(initial + 1e-30) + np.log(emission[:, observations[0]] + 1e-30)

    for t in range(1, n):
        new_alpha = np.zeros(n_states)
        for j in range(n_states):
            new_alpha[j] = np.logaddexp.reduce(
                alpha + np.log(transition[:, j] + 1e-30)
            ) + np.log(emission[j, observations[t]] + 1e-30)
        alpha = new_alpha

    return np.logaddexp.reduce(alpha)


# ═════════════════════════════════════════════════════════════════════
# Neural network layer → observation mapping
# ═════════════════════════════════════════════════════════════════════

def generate_layer_observations(n_layers, introgressed_layers, rng):
    """Generate synthetic observations from NN layer statistics.

    Normal layers emit mostly obs=0 (concordant weight stats).
    Introgressed layers emit mostly obs=2 (anomalous weight stats).
    """
    obs = []
    true_states = []
    for i in range(n_layers):
        if i in introgressed_layers:
            true_states.append(1)
            p = rng.random()
            if p < 0.05:
                obs.append(0)
            elif p < 0.15:
                obs.append(1)
            else:
                obs.append(2)
        else:
            true_states.append(0)
            p = rng.random()
            if p < 0.80:
                obs.append(0)
            elif p < 0.95:
                obs.append(1)
            else:
                obs.append(2)
    return np.array(obs), np.array(true_states)


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.RandomState(SEED)

    transition, emission, initial = build_hmm()
    obs, true_states = generate_layer_observations(N_LAYERS, INTROGRESSION_LAYERS, rng)

    path = viterbi(obs, transition, emission, initial)

    # Detection metrics
    true_pos = sum(1 for i in range(N_LAYERS) if path[i] == 1 and true_states[i] == 1)
    false_pos = sum(1 for i in range(N_LAYERS) if path[i] == 1 and true_states[i] == 0)
    true_neg = sum(1 for i in range(N_LAYERS) if path[i] == 0 and true_states[i] == 0)
    false_neg = sum(1 for i in range(N_LAYERS) if path[i] == 0 and true_states[i] == 1)

    n_introg = sum(true_states)
    n_normal = N_LAYERS - n_introg
    tpr = true_pos / max(n_introg, 1)
    fpr = false_pos / max(n_normal, 1)
    accuracy = (true_pos + true_neg) / N_LAYERS

    introg_fraction = sum(path) / N_LAYERS

    log_lik_full = forward_log_likelihood(obs, transition, emission, initial)
    normal_only_transition = np.array([[1.0]])
    normal_only_emission = emission[0:1, :]
    normal_only_initial = np.array([1.0])
    log_lik_null = forward_log_likelihood(obs, normal_only_transition,
                                           normal_only_emission, normal_only_initial)
    llr = 2.0 * (log_lik_full - log_lik_null)

    print(f"Observations: {obs[:20]}...")
    print(f"True states:  {true_states[:20]}...")
    print(f"Viterbi path: {path[:20]}...")
    print(f"TPR={tpr:.3f}, FPR={fpr:.3f}, Accuracy={accuracy:.3f}")
    print(f"Introg fraction: {introg_fraction:.3f}")
    print(f"LLR: {llr:.2f}")

    baseline = {
        "experiment": "099_introgression_nn",
        "seed": SEED,
        "n_layers": N_LAYERS,
        "n_introgressed": int(n_introg),
        "hmm": {
            "transition": transition.tolist(),
            "emission": emission.tolist(),
            "initial": initial.tolist(),
        },
        "observations": obs.tolist(),
        "true_states": true_states.tolist(),
        "viterbi_path": path.tolist(),
        "metrics": {
            "tpr": float(tpr),
            "fpr": float(fpr),
            "accuracy": float(accuracy),
            "introgression_fraction": float(introg_fraction),
            "log_likelihood_full": float(log_lik_full),
            "log_likelihood_null": float(log_lik_null),
            "llr": float(llr),
        },
    }

    json_path = "control/introgression_nn/introgression_nn_baseline.json"
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
    print("Experiment 099: HMM Introgression on NN Layers")
    print(f"{'='*60}\n")

    check("TPR > 0.5 (detect most introgressed)", tpr > 0.5)
    check("FPR < 0.3 (few false positives)", fpr < 0.3)
    check("Accuracy > 0.7", accuracy > 0.7)
    check("LLR > 0 (introgression model better)", llr > 0)
    check("Introgression fraction < 0.5", introg_fraction < 0.5)
    check("Introgression fraction > 0.01", introg_fraction > 0.01)

    check("Viterbi path length = N_LAYERS", len(path) == N_LAYERS)
    check("Observations all valid [0,2]",
          all(0 <= o <= 2 for o in obs))
    check("Path all valid [0,1]",
          all(0 <= s <= 1 for s in path))

    check("Deterministic", np.array_equal(
        viterbi(obs, transition, emission, initial), path))

    with open(json_path) as f:
        loaded = json.load(f)
    check("JSON roundtrip: LLR preserved",
          abs(loaded["metrics"]["llr"] - llr) < 1e-10)

    print(f"\n=== introgression_nn: {checks_pass}/{checks_total} checks "
          f"{'PASS' if checks_pass == checks_total else 'FAIL'} ===")

    sys.exit(0 if checks_pass == checks_total else 1)


if __name__ == "__main__":
    main()
