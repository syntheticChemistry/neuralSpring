# neuralSpring → ToadStool/BarraCUDA V85 Handoff — Session 127

**Date**: March 5, 2026
**Upstream pin**: BarraCUDA v0.3.3, ToadStool S94b, wgpu 28
**Previous**: V84 (S126, cross-spring fused op absorption)

## Summary

Paper 026 (Chuna LSTM blood glucose prediction) promoted to all 4 validation tiers,
closing the last gap in the full-pipeline validation story. All 26 papers are now
fully validated across Python baseline → Rust CPU parity → BarraCUDA CPU benchmark →
GPU pure workload → dispatch parity.

## Changes

### Validation tier completion for Paper 026

| Tier | Binary | New checks | Total |
|------|--------|------------|-------|
| CPU bench | `validate_barracuda_cpu_bench` | LSTM glucose domain (15th) | 15 domains |
| CPU math parity | `validate_cpu_math_parity` | autocorrelation + R² kernel | 10 kernels |
| GPU pure workload | `validate_gpu_pure_workload_all` | LSTM Tensor matmul gate projection | 13 domains |
| Dispatch parity | `validate_barracuda_dispatch_parity` | variance + pearson on CGM data | 55 checks |

### Python baseline closure

- `run_all_baselines.sh` now includes `glucose_prediction.py` — all 26 papers in unified runner
- `control/generate_cpu_references.py` extended with `gen_glucose_lstm()` (autocorrelation + R²)
- New Python bench: `control/glucose_prediction/bench_glucose_lstm.py`
- Reference JSON regenerated: 9 primitives + 10 kernels = 19 test groups

### New tolerance

- `GPU_LSTM_GLUCOSE_F32 = 0.05` — multi-step LSTM f32 Tensor chain tolerance

## Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt` | clean |
| `cargo clippy` (pedantic+nursery) | 0 warnings |
| `cargo doc` | 0 warnings |
| `cargo test --lib` | 871/883 (12 upstream GPU SIGSEGV) |
| `validate_all` | 218/218 |
| Python baselines | 41/41 |

## Evolution chain

```text
Python baseline (control/)
  ↓  cross-language validation (1e-10)
Rust CPU (neuralSpring lib) — 26 papers
  ↓  BarraCUDA CPU ports (pure Rust math)
BarraCUDA CPU (barracuda crate) — 15 bench domains, 38.6× vs Python
  ↓  GPU Tensor / WGSL shader dispatch
BarraCUDA GPU (pure GPU workload) — 13 domains
  ↓  dispatch parity (CPU ↔ GPU)
Cross-dispatch (55 checks) — identical math, portable execution
  ↓
ToadStool streaming → NUCLEUS → biomeOS
```

## Upstream notes for BarraCUDA/ToadStool team

- The 12 GPU SIGSEGV failures remain upstream (BarraCUDA/wgpu 28 runtime on llvmpipe).
  All involve `Tensor::from_data` or `ComputeDispatch`. Not blocking neuralSpring work.
- LSTM operations use `Tensor::matmul` + `Tensor::add` per step — no new WGSL shaders needed.
  The gate sigmoid/tanh are CPU-side. A fused LSTM cell WGSL shader would eliminate the
  per-step host round-trip and is a natural ToadStool streaming candidate.
- `autocorrelation` and `r2_score` are CPU-only in neuralSpring. If these become
  BarraCUDA dispatch ops, neuralSpring can wire them immediately.
