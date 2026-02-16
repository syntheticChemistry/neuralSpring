#!/usr/bin/env bash
# neuralSpring — Run all Phase 0 Python/PyTorch baselines
# Usage: bash scripts/run_all_baselines.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PASS=0
FAIL=0

run_experiment() {
    local name="$1"
    local script="$2"
    echo ""
    echo "================================================================"
    echo "  Running: $name"
    echo "================================================================"
    if python3 "$script" 2>&1; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "  *** FAILED: $name ***"
    fi
}

echo "================================================================"
echo "  neuralSpring Phase 0 — Full Baseline Suite"
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
echo "  GRAND SUMMARY"
echo "  Total: $PASS PASS, $FAIL FAIL out of $((PASS + FAIL)) experiments"
echo "================================================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
