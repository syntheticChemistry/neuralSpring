# neuralSpring — Deprecation & Migration Guide

**Date**: February 20, 2026
**ToadStool HEAD**: `dc540afd` (Session 25)
**Status**: Migration complete — deprecated modules fossilized

All 11 neuralSpring shortcomings (S-01 through S-11) are absorbed by
ToadStool. Deprecated workaround modules have been removed from the
active codebase and fossilized in `metalForge/fossils/evolved_s01_s11/`.

---

## Fossilized (~2,864 LOC evolved + ~1,127 LOC bench)

See `metalForge/fossils/FOSSIL_RECORD.md` for the full inventory.

### Evolved Modules → `metalForge/fossils/evolved_s01_s11/`

| Module | LOC | Shortcoming | BarraCUDA Replacement |
|--------|-----|-------------|----------------------|
| `fused_pipeline.rs` | 680 | S-01 | `TensorSession` |
| `fused_mlp.rs` | 356 | S-01/S-11 | `TensorSession::{matmul, relu, gelu, run}` |
| `fused_transformer.rs` | 725 | S-01/S-11 | `TensorSession::{head_split, attention, layer_norm}` |
| `layer_norm.rs` | 268 | S-08 | `Tensor::layer_norm_wgsl()` |
| `log_softmax.rs` | 259 | S-09 | `Tensor::log_softmax_wgsl()` |
| `matmul_cpu_tiled.wgsl` | 270 | S-02 | `ops::matmul` CpuTiled32 |
| `matmul_gpu_evolved.wgsl` | 306 | S-02 | `ops::matmul` GpuEvolved32 |

### Bench Binaries → `metalForge/fossils/bench/`

| Binary | Why fossilized |
|--------|----------------|
| `bench_fused_inference.rs` | Deep coupling to fused pipeline |
| `bench_scaling.rs` | Deep coupling to fused pipeline |

---

## Rewired (Active — Using Native APIs)

| Binary | What changed | Date |
|--------|-------------|------|
| `bench_barracuda_tensor` | Evolved `layer_norm`/`log_softmax` → native `Tensor::layer_norm_wgsl()`/`log_softmax_wgsl()` | Feb 20 |
| `validate_barracuda_tensor` | Same rewiring (earlier) | Feb 20 |
| `gpu.rs` | CPU path → `WgpuDevice::new_cpu_relaxed()` | Feb 20 |

---

## Still Active in `src/evolved/` (2 modules)

| Module | LOC | Why active | Path to absorption |
|--------|-----|-----------|-------------------|
| `mha.rs` | 182 | Native `Tensor::multi_head_attention` projection shaders hang (S-03b) | ToadStool: debug `project_with_head_split` GPU execution |
| `hmm_forward_gpu.rs` | 270 | No BarraCUDA equivalent | Candidate for `ops::hmm` |

### S-03b: Native MHA Projection Shader Hang

The z-dispatch fix (S-03) was absorbed, but the native
`Tensor::multi_head_attention` hangs during execution on
RTX 4070 / Vulkan. The evolved MHA (matmul + CPU head
split/concat + attention) works correctly and remains active.

**Binaries using evolved MHA**:
- `validate_barracuda_ml_inference`
- `bench_transformer_block`

---

## Migration Complete

| Priority | Action | Status |
|----------|--------|--------|
| ~~Done~~ | Rewire `validate_barracuda_tensor` | **Complete** |
| ~~Done~~ | Rewire `gpu.rs` to `new_cpu_relaxed()` | **Complete** |
| ~~Done~~ | Rewire `bench_barracuda_tensor` to native ops | **Complete** |
| ~~P1~~ | Migrate MHA to native — **blocked by S-03b** | Kept evolved::mha |
| ~~P2~~ | Migrate fused benchmarks | **Fossilized** |
| ~~P3~~ | Remove WGSL shaders | **Fossilized** |
| ~~P4~~ | Remove evolved modules | **Fossilized** (except mha + hmm) |

---

*Migration guide — neuralSpring rewired to modern ToadStool/BarraCUDA.*
