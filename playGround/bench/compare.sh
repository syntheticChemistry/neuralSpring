#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run PyTorch/CUDA and barraCuda/WGSL benchmarks side by side.
#
# Usage:
#   ./playGround/bench/compare.sh [--seq-len 128] [--ops-only]
#
# Produces JSON files in playGround/bench/results/ and a summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

SEQ_LEN="${SEQ_LEN:-128}"
OPS_ONLY=""

for arg in "$@"; do
    case "$arg" in
        --ops-only) OPS_ONLY="--ops-only" ;;
        --seq-len=*) SEQ_LEN="${arg#*=}" ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  neuralSpring playGround — Inference Benchmark"
echo "  PyTorch/CUDA vs barraCuda/WGSL"
echo "  Sequence length: $SEQ_LEN"
echo "============================================================"
echo

# --- PyTorch/CUDA ---
PYTORCH_JSON="$RESULTS_DIR/pytorch_cuda.json"
echo "[1/4] PyTorch/CUDA benchmark..."
if python3 "$SCRIPT_DIR/pytorch_baseline.py" --device cuda --seq-len "$SEQ_LEN" $OPS_ONLY --json > "$PYTORCH_JSON" 2>/dev/null; then
    echo "  -> $PYTORCH_JSON"
else
    echo "  SKIP (no CUDA or PyTorch not available)"
    echo '{"framework": "pytorch", "device": "cuda", "error": "not available"}' > "$PYTORCH_JSON"
fi

# --- PyTorch/CPU ---
PYTORCH_CPU_JSON="$RESULTS_DIR/pytorch_cpu.json"
echo "[2/4] PyTorch/CPU benchmark..."
if python3 "$SCRIPT_DIR/pytorch_baseline.py" --device cpu --seq-len "$SEQ_LEN" $OPS_ONLY --json > "$PYTORCH_CPU_JSON" 2>/dev/null; then
    echo "  -> $PYTORCH_CPU_JSON"
else
    echo "  SKIP (PyTorch not available)"
    echo '{"framework": "pytorch", "device": "cpu", "error": "not available"}' > "$PYTORCH_CPU_JSON"
fi

# --- barraCuda/WGSL ---
BARRACUDA_JSON="$RESULTS_DIR/barracuda_wgsl.json"
echo "[3/4] barraCuda/WGSL benchmark..."
cd "$PROJECT_DIR"
if cargo run -p neuralspring-playground --release --bin neuralspring_bench_inference -- --seq-len "$SEQ_LEN" $OPS_ONLY --json > "$BARRACUDA_JSON" 2>/dev/null; then
    echo "  -> $BARRACUDA_JSON"
else
    echo "  FAILED"
    echo '{"framework": "barracuda", "error": "build or runtime failure"}' > "$BARRACUDA_JSON"
fi

# --- Comparison ---
echo
echo "[4/4] Comparison"
echo
python3 -c "
import json, sys

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {'error': 'not available'}

pytorch_cuda = load('$PYTORCH_JSON')
pytorch_cpu  = load('$PYTORCH_CPU_JSON')
barracuda    = load('$BARRACUDA_JSON')

print('=' * 80)
print(f\"{'Operation':<20} {'PyTorch/CUDA':>14} {'PyTorch/CPU':>14} {'barraCuda/WGSL':>14} {'CUDA/bCuda':>12}\")
print('=' * 80)

bc_ops = barracuda.get('ops', {})
pt_ops = pytorch_cuda.get('ops', {})
cpu_ops = pytorch_cpu.get('ops', {})

for op in ['matmul', 'layer_norm', 'gelu', 'softmax', 'sdpa']:
    bc = bc_ops.get(op, {}).get('median_us', 0)
    pt = pt_ops.get(op, {}).get('median_us', 0)
    cpu = cpu_ops.get(op, {}).get('median_us', 0)
    ratio = f'{pt/bc:.2f}x' if bc > 0 and pt > 0 else 'N/A'

    bc_str = f'{bc:.1f}µs' if bc else 'N/A'
    pt_str = f'{pt:.1f}µs' if pt else 'N/A'
    cpu_str = f'{cpu:.1f}µs' if cpu else 'N/A'

    print(f'{op:<20} {pt_str:>14} {cpu_str:>14} {bc_str:>14} {ratio:>12}')

print('-' * 80)

# Forward pass comparison
bc_fwd = barracuda.get('forward', {}).get('forward_pass', {})
pt_fwd = pytorch_cuda.get('forward', {}).get('forward_pass', {})
cpu_fwd = pytorch_cpu.get('forward', {}).get('forward_pass', {})

bc_ms = bc_fwd.get('median_ms', 0)
pt_ms = pt_fwd.get('median_ms', 0)
cpu_ms = cpu_fwd.get('median_ms', 0)
ratio = f'{pt_ms/bc_ms:.2f}x' if bc_ms > 0 and pt_ms > 0 else 'N/A'

bc_str = f'{bc_ms:.2f}ms' if bc_ms else 'N/A'
pt_str = f'{pt_ms:.2f}ms' if pt_ms else 'N/A'
cpu_str = f'{cpu_ms:.2f}ms' if cpu_ms else 'N/A'

print(f\"{'forward_pass':<20} {pt_str:>14} {cpu_str:>14} {bc_str:>14} {ratio:>12}\")
print('=' * 80)
print()
print('CUDA/bCuda ratio > 1.0 means barraCuda is faster')
print('CUDA/bCuda ratio < 1.0 means PyTorch/CUDA is faster')
" 2>/dev/null || echo "(install python3 + json for comparison table)"

echo
echo "Raw results in $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
