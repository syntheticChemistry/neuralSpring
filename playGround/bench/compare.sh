#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run PyTorch/CUDA and barraCuda/WGSL benchmarks side by side.
#
# Usage:
#   ./playGround/bench/compare.sh [--seq-len 128] [--ops-only]
#
# Produces JSON files in playGround/bench/results/ and a summary.
# Runs barraCuda in both cold (per-call session) and hot (reused session)
# modes for dispatch overhead analysis.

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
echo "  PyTorch/CUDA vs barraCuda/WGSL (cold + hot dispatch)"
echo "  Sequence length: $SEQ_LEN"
echo "============================================================"
echo

# --- PyTorch/CUDA ---
PYTORCH_JSON="$RESULTS_DIR/pytorch_cuda.json"
echo "[1/5] PyTorch/CUDA benchmark..."
if python3 "$SCRIPT_DIR/pytorch_baseline.py" --device cuda --seq-len "$SEQ_LEN" $OPS_ONLY --json > "$PYTORCH_JSON" 2>/dev/null; then
    echo "  -> $PYTORCH_JSON"
else
    echo "  SKIP (no CUDA or PyTorch not available)"
    echo '{"framework": "pytorch", "device": "cuda", "error": "not available"}' > "$PYTORCH_JSON"
fi

# --- PyTorch/CPU ---
PYTORCH_CPU_JSON="$RESULTS_DIR/pytorch_cpu.json"
echo "[2/5] PyTorch/CPU benchmark..."
if python3 "$SCRIPT_DIR/pytorch_baseline.py" --device cpu --seq-len "$SEQ_LEN" $OPS_ONLY --json > "$PYTORCH_CPU_JSON" 2>/dev/null; then
    echo "  -> $PYTORCH_CPU_JSON"
else
    echo "  SKIP (PyTorch not available)"
    echo '{"framework": "pytorch", "device": "cpu", "error": "not available"}' > "$PYTORCH_CPU_JSON"
fi

# --- barraCuda/WGSL (cold) ---
BARRACUDA_COLD_JSON="$RESULTS_DIR/barracuda_cold.json"
echo "[3/5] barraCuda/WGSL (cold dispatch)..."
cd "$PROJECT_DIR"
if cargo run -p neuralspring-playground --release --bin neuralspring_bench_inference -- --seq-len "$SEQ_LEN" $OPS_ONLY --json > "$BARRACUDA_COLD_JSON" 2>/dev/null; then
    echo "  -> $BARRACUDA_COLD_JSON"
else
    echo "  FAILED"
    echo '{"framework": "barracuda", "dispatch_mode": "cold", "error": "build or runtime failure"}' > "$BARRACUDA_COLD_JSON"
fi

# --- barraCuda/WGSL (hot) ---
BARRACUDA_HOT_JSON="$RESULTS_DIR/barracuda_hot.json"
echo "[4/5] barraCuda/WGSL (hot dispatch — reused TensorSession)..."
if cargo run -p neuralspring-playground --release --bin neuralspring_bench_inference -- --seq-len "$SEQ_LEN" $OPS_ONLY --hot --json > "$BARRACUDA_HOT_JSON" 2>/dev/null; then
    echo "  -> $BARRACUDA_HOT_JSON"
else
    echo "  FAILED"
    echo '{"framework": "barracuda", "dispatch_mode": "hot", "error": "build or runtime failure"}' > "$BARRACUDA_HOT_JSON"
fi

# --- Comparison ---
echo
echo "[5/5] Comparison"
echo
python3 -c "
import json

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {'error': 'not available'}

pytorch_cuda = load('$PYTORCH_JSON')
pytorch_cpu  = load('$PYTORCH_CPU_JSON')
bc_cold      = load('$BARRACUDA_COLD_JSON')
bc_hot       = load('$BARRACUDA_HOT_JSON')

print('=' * 100)
print(f\"{'Operation':<16} {'PyTorch/CUDA':>12} {'PyTorch/CPU':>12} {'bCuda cold':>12} {'bCuda hot':>12} {'CUDA/hot':>10} {'cold/hot':>10}\")
print('=' * 100)

pt_ops  = pytorch_cuda.get('ops', {})
cpu_ops = pytorch_cpu.get('ops', {})
cold_ops = bc_cold.get('ops', {})
hot_ops  = bc_hot.get('ops', {})

for op in ['matmul', 'layer_norm', 'gelu', 'softmax', 'sdpa']:
    pt   = pt_ops.get(op, {}).get('median_us', 0)
    cpu  = cpu_ops.get(op, {}).get('median_us', 0)
    cold = cold_ops.get(op, {}).get('median_us', 0)
    hot  = hot_ops.get(op, {}).get('median_us', 0)

    cuda_hot = f'{pt/hot:.1f}x' if hot > 0 and pt > 0 else 'N/A'
    cold_hot = f'{cold/hot:.1f}x' if hot > 0 and cold > 0 else 'N/A'

    pt_s   = f'{pt:.1f}us' if pt else 'N/A'
    cpu_s  = f'{cpu:.1f}us' if cpu else 'N/A'
    cold_s = f'{cold:.1f}us' if cold else 'N/A'
    hot_s  = f'{hot:.1f}us' if hot else 'N/A'

    print(f'{op:<16} {pt_s:>12} {cpu_s:>12} {cold_s:>12} {hot_s:>12} {cuda_hot:>10} {cold_hot:>10}')

print('-' * 100)

# Forward pass
pt_fwd  = pytorch_cuda.get('forward', {}).get('forward_pass', {})
cpu_fwd = pytorch_cpu.get('forward', {}).get('forward_pass', {})
cold_fwd = bc_cold.get('forward', {}).get('forward_pass', {})
hot_fwd  = bc_hot.get('forward', {}).get('forward_pass', {})

pt_ms   = pt_fwd.get('median_ms', 0)
cpu_ms  = cpu_fwd.get('median_ms', 0)
cold_ms = cold_fwd.get('median_ms', 0)
hot_ms  = hot_fwd.get('median_ms', 0)

cuda_hot = f'{pt_ms/hot_ms:.1f}x' if hot_ms > 0 and pt_ms > 0 else 'N/A'
cold_hot = f'{cold_ms/hot_ms:.1f}x' if hot_ms > 0 and cold_ms > 0 else 'N/A'

print(f\"{'forward_pass':<16} {f'{pt_ms:.2f}ms' if pt_ms else 'N/A':>12} {f'{cpu_ms:.2f}ms' if cpu_ms else 'N/A':>12} {f'{cold_ms:.2f}ms' if cold_ms else 'N/A':>12} {f'{hot_ms:.2f}ms' if hot_ms else 'N/A':>12} {cuda_hot:>10} {cold_hot:>10}\")
print('=' * 100)
print()
print('CUDA/hot  < 1.0 = barraCuda hot dispatch faster than PyTorch/CUDA')
print('cold/hot  > 1.0 = speedup from pipeline reuse (dispatch overhead eliminated)')
print()
print('To close the CUDA gap, next steps:')
print('  1. ToadStool compute.submit — persistent sessions + pipeline caching')
print('  2. coralReef shader.compile — WGSL to native SASS/GFX (bypass Vulkan)')
print('  3. GpuSessionBuilder::pre_warm() — compile pipelines at startup')
" 2>/dev/null || echo "(install python3 + json for comparison table)"

echo
echo "Raw results in $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
