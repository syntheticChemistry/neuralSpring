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
echo "  GRAND SUMMARY"
echo "  Total: $PASS PASS, $FAIL FAIL out of $((PASS + FAIL)) experiments"
echo "================================================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
