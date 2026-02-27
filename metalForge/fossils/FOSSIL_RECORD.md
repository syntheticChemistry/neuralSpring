# Fossil Record — neuralSpring Evolved Modules

> **Pattern**: `hotSpring` evolve → validate → hand off → absorb → fossil  
> **Spring**: neuralSpring (Feb 2026)  
> **Absorbed by**: ToadStool `77f70b2e` / BarraCUDA 0.2+  
> **ToadStool HEAD**: `e96576ee` — absorption span `d45fdfb3`..`e96576ee` (Sessions 42–68)

## Purpose

This directory preserves the locally-evolved GPU-resident ops and fused
inference pipelines that neuralSpring developed as workarounds for
BarraCUDA shortcomings S-01 through S-11. ToadStool has since absorbed
all eleven shortcomings; the code here is archived for reference, not
active compilation.

To trace what each shortcoming was and how ToadStool fixed it, see
`specs/TOADSTOOL_HANDOFF.md` and `whitePaper/BARRACUDA_EVOLUTION.md`.

## Fossil Inventory

### evolved_s01_s11/ — Evolved GPU-Resident Ops

| File | LOC | Shortcoming | BarraCUDA Replacement | Status |
|------|-----|-------------|----------------------|--------|
| `fused_pipeline.rs` | 680 | S-01 (no session API) | `TensorSession` | Absorbed |
| `fused_mlp.rs` | 356 | S-01, S-11 | `TensorSession::{matmul, relu, gelu, run}` | Absorbed |
| `fused_transformer.rs` | 725 | S-01, S-11 | `TensorSession::{head_split, attention, head_concat}` | Absorbed |
| `layer_norm.rs` | 268 | S-08 (readback) | `Tensor::layer_norm_wgsl()` | Absorbed |
| `log_softmax.rs` | 259 | S-09 (readback) | `Tensor::log_softmax_wgsl()` | Absorbed |
| `matmul_cpu_tiled.wgsl` | 270 | S-02 (single kernel) | `ops::matmul` 4-tier `KernelRouter` | Absorbed |
| `matmul_gpu_evolved.wgsl` | 306 | S-02 (single kernel) | `ops::matmul` 4-tier `KernelRouter` | Absorbed |
| `eigh_local.rs` | 543 | S-12 (Jacobi accuracy) | `ops::linalg::eigh_householder_qr` | Absorbed |

**Total fossilized evolved code**: ~3,407 LOC

### bench/ — Fused Inference Benchmarks

| File | LOC | Why fossilized |
|------|-----|----------------|
| `bench_fused_inference.rs` | 688 | Deep coupling to fused_pipeline/fused_mlp/fused_transformer; replaced by native TensorSession benchmarks |
| `bench_scaling.rs` | 439 | Same fused dependencies; scaling now benchmarked via native Tensor ops |
| `bench_inference.py` | 200 | Python baseline for fused inference benchmark; only consumer was `bench_fused_inference.rs` |
| `bench_scaling.py` | 206 | Python scaling benchmark; only consumer was `bench_scaling.rs` |

**Total fossilized bench code**: ~2,533 LOC (Rust + Python)

### evolved_s13/ — PooledBuffer Race Workaround

| File | LOC | Shortcoming | BarraCUDA Fix | Status |
|------|-----|-------------|---------------|--------|
| `tensor_sync.rs` | 179 | S-13 (PooledBuffer drop-before-completion race) | `d45fdfb3` — device.poll(Wait) in PooledBuffer::drop | **FIXED upstream** |

Provided `gpu_fence`, `materialize`, `fenced_matmul` — proving the correctness
of the sync approach. Zero callers remained after upstream fix at `d45fdfb3`.

### diagnostics/ — S-15 Investigation Scripts

| File | LOC | Why fossilized |
|------|-----|----------------|
| `validate_barracuda_gpu_s15_diagnostic.rs` | ~90 | S-15 matmul hang diagnostic — used to root-cause the WGPU/Vulkan driver bug. Not in Cargo.toml |
| `validate_barracuda_gpu_minimal_test.rs` | ~50 | Minimal S-15 repro case — smallest possible matmul to isolate hang. Not in Cargo.toml |

These diagnostic scripts were one-off investigation tools. The findings are
documented in `specs/TOADSTOOL_HANDOFF.md` (S-15 section) and the V11 handoff.

## What Remains Active

One evolved module survives in `src/evolved/`:

| Module | Why active | Path to absorption |
|--------|-----------|-------------------|
| `mha.rs` | Thin wrapper delegating to upstream `barracuda::ops::mha::MultiHeadAttention`. Retained until all callers migrate to the 3D API. | Retire when callers use upstream directly |

`hmm_forward_gpu.rs` was fossilized in Session 40 — `HmmBatchForwardF64` (wetSpring origin) is primary.
S-03b fully resolved upstream at `ToadStool` `0c998992` (S60–S61).

## Timeline

| Date | Event |
|------|-------|
| Jan 2026 | neuralSpring begins evolving workarounds for S-01 through S-11 |
| Feb 19-22, 2026 | ToadStool `dc540afd`→`77f70b2e` absorbs all 12 shortcomings (S-01..S-12) |
| Feb 20, 2026 | neuralSpring completes rewiring to native APIs; deprecated modules fossilized |
| Feb 22, 2026 | `bench_inference.py` and `bench_scaling.py` moved from `control/` to fossils (orphaned by fossilized Rust) |
| Feb 22, 2026 | `eigh_local.rs` fossilized — `barracuda::ops::linalg::eigh_householder_qr` (`77f70b2e`) absorbed S-12 |
| Feb 22, 2026 | `tensor_sync.rs` fossilized — S-13 `PooledBuffer` race **FIXED** upstream at `d45fdfb3` (Session 39). Zero callers |
| Feb 23, 2026 | `hmm_forward_log.wgsl` fossilized — absorbed by BarraCUDA `HmmBatchForwardF64` (wetSpring origin). Zero `include_str!` references remaining. Moved to `absorbed_shaders/`. |

## How to Revive

If you ever need to understand or compare the evolved implementations:

```bash
# View the fossilized code
ls metalForge/fossils/evolved_s01_s11/

# Compare evolved matmul vs native BarraCUDA KernelRouter
diff metalForge/fossils/evolved_s01_s11/matmul_gpu_evolved.wgsl \
     ../phase1/toadstool/crates/barracuda/src/ops/matmul/shaders/
```

These files will never compile in isolation — they reference removed module
wiring. They exist purely as a record of neuralSpring's shader evolution.
