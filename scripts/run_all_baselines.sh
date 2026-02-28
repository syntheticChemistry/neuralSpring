#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later

# neuralSpring — Run all Phase 0 + Phase 0+ + Phase 0++ Python/PyTorch baselines
#
# Exit codes per script:
#   0  = all checks PASS
#   1  = one or more checks FAIL
#   77 = SKIPPED (missing dependency)
#
# Usage: bash scripts/run_all_baselines.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
SKIP=0
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_FILE="$RESULTS_DIR/baseline_${TIMESTAMP}.json"

declare -a RESULT_ENTRIES

run_experiment() {
    local name="$1"
    local script="$2"
    echo ""
    echo "================================================================"
    echo "  Running: $name"
    echo "================================================================"
    local start_ms
    start_ms=$(($(date +%s%N 2>/dev/null || echo "$(date +%s)000000000")/1000000))
    python3 "$script" 2>&1
    local rc=$?
    local end_ms
    end_ms=$(($(date +%s%N 2>/dev/null || echo "$(date +%s)000000000")/1000000))
    local elapsed_ms=$((end_ms - start_ms))
    local status
    if [ "$rc" -eq 0 ]; then
        PASS=$((PASS + 1))
        status="pass"
    elif [ "$rc" -eq 77 ]; then
        SKIP=$((SKIP + 1))
        status="skip"
        echo "  *** SKIPPED: $name (missing dependency) ***"
    else
        FAIL=$((FAIL + 1))
        status="fail"
        echo "  *** FAILED: $name ***"
    fi
    RESULT_ENTRIES+=("{\"name\":\"$name\",\"script\":\"$script\",\"status\":\"$status\",\"exit_code\":$rc,\"elapsed_ms\":$elapsed_ms}")
}

echo "================================================================"
echo "  neuralSpring Phase 0 + Phase 0+ + Phase 0++ — Full Baseline Suite"
echo "  $(date)"
echo "================================================================"

run_experiment "Exp 001: Neural Surrogate Validation" \
    control/surrogate/surrogate_validation.py

run_experiment "Exp 002: Transformer Inference Baseline" \
    control/transformer/transformer_inference.py

run_experiment "Exp 003: Sequence Forecasting (LSTM/GRU)" \
    control/sequence/sequence_forecasting.py

run_experiment "Exp 004: Transfer Learning" \
    control/transfer/transfer_learning.py

run_experiment "Exp 005: Isomorphic Pattern Catalog" \
    control/isomorphic/isomorphic_catalog.py

echo ""
echo "================================================================"
echo "  Phase 0+: Scholarly Reproduction Studies"
echo "================================================================"

run_experiment "Study 001: PINN Burgers (Raissi 2019)" \
    control/pinn/pinn_burgers.py

run_experiment "Study 002: DeepONet Antiderivative (Lu 2021)" \
    control/deeponet/deeponet_antideriv.py

run_experiment "Study 003: LeNet-5 MNIST (LeCun 1998)" \
    control/lenet/lenet_mnist.py

run_experiment "Study 004: LSTM ERA5 Weather (Gauch 2021)" \
    control/lstm_weather/lstm_era5.py

run_experiment "Study 005: Quantized Inference (Q8/Q4)" \
    control/quantized/quantized_inference.py

echo ""
echo "================================================================"
echo "  Phase 0++: Paper Reproductions"
echo "================================================================"

run_experiment "Paper 011: Counterdiabatic Evolution (Iram/Dolson 2020)" \
    control/counterdiabatic/counterdiabatic_evolution.py

run_experiment "Paper 012: MODES Toolbox (Dolson 2019)" \
    control/modes/modes_toolbox.py

run_experiment "Paper 013: Ecological Dynamics (Dolson & Ofria 2018)" \
    control/eco_dynamics/eco_dynamics.py

run_experiment "Paper 014: Directed Evolution (Dolson 2022)" \
    control/directed_evolution/directed_evolution.py

run_experiment "Paper 016: HMM Forward/Backward/Viterbi (Liu 2014)" \
    control/hmm_phylo/hmm_phylo.py

run_experiment "Paper 019: Game Theory & QS Cooperation (Bruger/Waters 2018)" \
    control/game_theory/game_theory.py

run_experiment "Paper 015: Heterogeneous Swarm Robotics (Foreback/Dolson 2025)" \
    control/swarm_robotics/swarm_robotics.py

run_experiment "Paper 017: SATé Alignment (Liu 2009)" \
    control/sate_alignment/sate_alignment.py

run_experiment "Paper 018: Introgression Detection (Liu 2015)" \
    control/introgression/introgression.py

run_experiment "Paper 020: Regulatory Network (Mhatre/Waters 2020)" \
    control/regulatory_network/regulatory_network.py

run_experiment "Paper 021: Signal Integration (Srivastava/Waters 2011)" \
    control/signal_integration/signal_integration.py

run_experiment "Paper 022: Spectral Commutativity (Kachkovskiy 2016)" \
    control/spectral_commutativity/spectral_commutativity.py

run_experiment "Paper 023: Anderson Localization (Bourgain/Kachkovskiy 2018)" \
    control/anderson_localization/anderson_localization.py

run_experiment "Paper 024: Pangenome Selection (Liu genomics)" \
    control/pangenome_selection/pangenome_selection.py

run_experiment "Paper 025: Meta-Population Dynamics (Liu population genetics)" \
    control/meta_population/meta_population.py

echo ""
echo "================================================================"
echo "  Supplementary: Data Generation"
echo "================================================================"

run_experiment "WDM EOS Surrogate Baselines (nW-02)" \
    control/wdm/eos_surrogate.py

run_experiment "WDM Transport Surrogate (nW-01)" \
    control/wdm/transport_surrogate.py

run_experiment "WDM Transfer Classical→WDM (nW-04)" \
    control/wdm/transfer_classical_to_wdm.py

run_experiment "WDM S(q,ω) Peak Predictor (nW-03)" \
    control/wdm/sqw_peak_predictor.py

run_experiment "WDM ESN Regime Classifier (nW-05)" \
    control/wdm/esn_regime_classifier.py

run_experiment "ML Inference Baselines (MLP + Transformer JSON)" \
    control/ml_inference/generate_baselines.py

echo ""
echo "================================================================"
echo "  Publication Experiments (Exp-050, Exp-052, Exp-053)"
echo "================================================================"

run_experiment "Exp-050: Training Trajectory Spectral Analysis (Paper A)" \
    control/training_trajectory/training_trajectory.py

run_experiment "Exp-052: Hessian Eigenanalysis at Trained Minima (Paper D)" \
    control/hessian_eigenanalysis/hessian_eigenanalysis.py

run_experiment "Exp-053: Anderson Multi-Agent Coordination (Paper C)" \
    control/anderson_multiagent/anderson_multiagent.py

echo ""
echo "================================================================"
echo "  coralForge: Sovereign Structure Prediction (nF-01/02/03)"
echo "================================================================"

run_experiment "nF-01: Evoformer Primitives (AlphaFold2)" \
    control/coral_forge/evoformer_primitives.py

run_experiment "nF-02: AlphaFold2 Full Evoformer Block" \
    control/coral_forge/alphafold2_evoformer_block.py

run_experiment "nF-03a: AlphaFold3 Diffusion" \
    control/coral_forge/alphafold3_diffusion.py

run_experiment "nF-03b: AlphaFold3 Pairformer" \
    control/coral_forge/alphafold3_pairformer.py

run_experiment "nF-03c: AlphaFold3 Confidence Heads" \
    control/coral_forge/alphafold3_confidence.py

echo ""
echo "================================================================"
echo "  GRAND SUMMARY"
echo "  Passed: $PASS, Failed: $FAIL, Skipped: $SKIP"
echo "  Total: $((PASS + FAIL + SKIP)) experiments"
echo "================================================================"

# Write JSON results for longitudinal tracking
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git diff --quiet 2>/dev/null && echo "clean" || echo "dirty")
NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
SCIPY_VER=$(python3 -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")

sep=""
{
    echo "{"
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"hostname\": \"$(hostname)\","
    echo "  \"python\": \"$(python3 --version 2>&1)\","
    echo "  \"numpy\": \"$NUMPY_VER\","
    echo "  \"scipy\": \"$SCIPY_VER\","
    echo "  \"torch\": \"$TORCH_VER\","
    echo "  \"commit\": \"$GIT_COMMIT\","
    echo "  \"tree_state\": \"$GIT_DIRTY\","
    echo "  \"pass\": $PASS, \"fail\": $FAIL, \"skip\": $SKIP,"
    echo "  \"experiments\": ["
    for entry in "${RESULT_ENTRIES[@]}"; do
        echo "    ${sep}${entry}"
        sep=","
    done
    echo "  ]"
    echo "}"
} > "$RESULTS_FILE"
echo "  Results written to: $RESULTS_FILE"

if [ "$FAIL" -gt 0 ]; then
    exit 1
elif [ "$SKIP" -gt 0 ]; then
    echo "  WARNING: $SKIP experiments skipped — install all dependencies"
    exit 77
else
    exit 0
fi
