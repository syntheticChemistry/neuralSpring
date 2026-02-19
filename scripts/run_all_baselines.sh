#!/usr/bin/env bash
# neuralSpring — Run all Phase 0 + Phase 0+ Python/PyTorch baselines
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
echo "  neuralSpring Phase 0 + Phase 0+ — Full Baseline Suite"
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
echo "  Passed: $PASS, Failed: $FAIL, Skipped: $SKIP"
echo "  Total: $((PASS + FAIL + SKIP)) experiments"
echo "================================================================"

# Write JSON results for longitudinal tracking
sep=""
{
    echo "{"
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"hostname\": \"$(hostname)\","
    echo "  \"python\": \"$(python3 --version 2>&1)\","
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
