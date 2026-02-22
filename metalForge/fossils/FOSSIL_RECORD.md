# Fossil Record — neuralSpring Evolved Modules

> **Pattern**: `hotSpring` evolve → validate → hand off → absorb → fossil  
> **Spring**: neuralSpring (Feb 2026)  
> **Absorbed by**: ToadStool `dc540afd` / BarraCUDA 0.2+

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

**Total fossilized evolved code**: ~2,864 LOC

### bench/ — Fused Inference Benchmarks

| File | LOC | Why fossilized |
|------|-----|----------------|
| `bench_fused_inference.rs` | 688 | Deep coupling to fused_pipeline/fused_mlp/fused_transformer; replaced by native TensorSession benchmarks |
| `bench_scaling.rs` | 439 | Same fused dependencies; scaling now benchmarked via native Tensor ops |
| `bench_inference.py` | 200 | Python baseline for fused inference benchmark; only consumer was `bench_fused_inference.rs` |
| `bench_scaling.py` | 206 | Python scaling benchmark; only consumer was `bench_scaling.rs` |

**Total fossilized bench code**: ~2,533 LOC (Rust + Python)

## What Remains Active

Two evolved modules survive in `src/evolved/`:

| Module | Why active | Path to absorption |
|--------|-----------|-------------------|
| `mha.rs` | Native `Tensor::multi_head_attention` projection shaders hang (S-03b) | ToadStool: debug `project_with_head_split` / `concat_and_project` GPU execution flow |
| `hmm_forward_gpu.rs` | Active metalForge evolution — no BarraCUDA equivalent | Candidate for `ops::hmm` in BarraCUDA |

## Timeline

| Date | Event |
|------|-------|
| Jan 2026 | neuralSpring begins evolving workarounds for S-01 through S-11 |
| Feb 19, 2026 | ToadStool `dc540afd` absorbs all 11 shortcomings |
| Feb 20, 2026 | neuralSpring completes rewiring to native APIs; deprecated modules fossilized |
| Feb 22, 2026 | `bench_inference.py` and `bench_scaling.py` moved from `control/` to fossils (orphaned by fossilized Rust) |

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
