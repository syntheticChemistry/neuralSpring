#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Python baseline drift detector for neuralSpring CI.
#
# Re-runs all 39 Python control experiments and verifies they still produce
# the same pass/fail results. Any failure indicates baseline drift — either
# a dependency update changed numeric behavior, or a script was modified
# without updating the Rust validation targets.
#
# Usage:
#   ./control/check_drift.sh              # run all baselines
#   ./control/check_drift.sh hmm_phylo    # run a single module
#
# Requirements:
#   Python 3.10+ with packages from control/requirements.txt
#   (pinned versions for reproducibility)
#
# Exit codes:
#   0 — all baselines pass (no drift)
#   1 — one or more baselines failed (drift detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0
SKIP=0
FAILED_NAMES=()

MODULES=(
    "surrogate/surrogate_validation.py"
    "transformer/transformer_inference.py"
    "lenet/lenet_mnist.py"
    "transfer/transfer_learning.py"
    "isomorphic/isomorphic_catalog.py"
    "lstm_weather/lstm_era5.py"
    "quantized/quantized_inference.py"
    "sequence/sequence_forecasting.py"
    "counterdiabatic/counterdiabatic_evolution.py"
    "modes/modes_toolbox.py"
    "eco_dynamics/eco_dynamics.py"
    "directed_evolution/directed_evolution.py"
    "hmm_phylo/hmm_phylo.py"
    "game_theory/game_theory.py"
    "regulatory_network/regulatory_network.py"
    "signal_integration/signal_integration.py"
    "swarm_robotics/swarm_robotics.py"
    "sate_alignment/sate_alignment.py"
    "introgression/introgression.py"
    "spectral_commutativity/spectral_commutativity.py"
    "anderson_localization/anderson_localization.py"
    "pangenome_selection/pangenome_selection.py"
    "meta_population/meta_population.py"
    "pinn/pinn_burgers.py"
    "deeponet/deeponet_antideriv.py"
    "wdm/eos_surrogate.py"
    "wdm/transport_surrogate.py"
    "wdm/transfer_classical_to_wdm.py"
    "wdm/sqw_peak_predictor.py"
    "wdm/esn_regime_classifier.py"
    "ml_inference/generate_baselines.py"
    "coral_forge/evoformer_primitives.py"
    "coral_forge/alphafold2_evoformer_block.py"
    "coral_forge/alphafold3_diffusion.py"
    "coral_forge/alphafold3_pairformer.py"
    "coral_forge/alphafold3_confidence.py"
    "training_trajectory/training_trajectory.py"
    "hessian_eigenanalysis/hessian_eigenanalysis.py"
    "anderson_multiagent/anderson_multiagent.py"
    "immunological_anderson/immunological_anderson.py"
    "immunological_anderson/immunological_anderson_extended.py"
)

run_module() {
    local script="$1"
    local name
    name="$(dirname "$script")"

    printf "  %-35s " "$name"

    if [[ ! -f "$script" ]]; then
        echo "SKIP (not found)"
        SKIP=$((SKIP + 1))
        return
    fi

    local output exit_code
    output=$(python3 "$script" 2>&1) && exit_code=0 || exit_code=$?

    local pass_count fail_count
    pass_count=$(echo "$output" | grep -c '\[PASS\]' || true)
    fail_count=$(echo "$output" | grep -c '\[FAIL\]' || true)

    if [[ $exit_code -eq 0 ]]; then
        echo "PASS ($pass_count checks)"
        PASS=$((PASS + 1))
    else
        echo "FAIL ($pass_count pass, $fail_count fail, exit $exit_code)"
        FAIL=$((FAIL + 1))
        FAILED_NAMES+=("$name")
    fi
}

echo "=== neuralSpring Python baseline drift check ==="
echo "  Python: $(python3 --version 2>&1)"
echo "  NumPy:  $(python3 -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "  SciPy:  $(python3 -c 'import scipy; print(scipy.__version__)' 2>&1)"
echo ""

if [[ $# -gt 0 ]]; then
    for mod in "$@"; do
        found=0
        for script in "${MODULES[@]}"; do
            if [[ "$script" == "$mod/"* || "$script" == "$mod" ]]; then
                run_module "$script"
                found=1
            fi
        done
        if [[ $found -eq 0 ]]; then
            echo "  Unknown module: $mod"
            FAIL=$((FAIL + 1))
            FAILED_NAMES+=("$mod")
        fi
    done
else
    for script in "${MODULES[@]}"; do
        run_module "$script"
    done
fi

echo ""
TOTAL=$((PASS + FAIL + SKIP))
echo "=== drift check: $PASS/$TOTAL PASS, $FAIL FAIL, $SKIP SKIP ==="

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "DRIFT DETECTED in: ${FAILED_NAMES[*]}"
    echo "  → Re-run failed modules individually to diagnose"
    echo "  → If intentional, update Rust provenance and tolerance values"
    exit 1
fi
