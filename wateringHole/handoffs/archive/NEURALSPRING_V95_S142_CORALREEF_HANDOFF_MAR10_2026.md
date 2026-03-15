<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V95 → coralReef Detailed Handoff

| Field | Value |
|-------|-------|
| **Date** | 2026-03-10 |
| **From** | neuralSpring S142 (1048 lib + 71 forge + 9 integration tests, 233 binaries) |
| **To** | coralReef team |
| **Synced against** | coralReef Iteration 29 (`2779c88`), barraCuda `83aa08a`, toadStool S142 (`a86bc546`) |
| **License** | AGPL-3.0-or-later |

---

## Executive Summary

neuralSpring is the **learning and validation Spring**: 26 scholarly paper
reproductions + 6 baseCamp sub-theses + 5 WDM surrogates, all validated from
Python baselines through Rust CPU to GPU dispatch. This handoff documents:

1. What neuralSpring needs from coralReef (sovereign shader compilation)
2. What neuralSpring has learned about GPU precision that's relevant to coralReef
3. The current coralReef bridge state in metalForge
4. Cross-spring shader inventory relevant to coralReef's corpus
5. The `enable f64;` PTXAS regression and its implications for coralReef

---

## Part 1: What neuralSpring Needs from coralReef

### Primary Need: Sovereign WGSL→Native Compilation

neuralSpring currently compiles all WGSL shaders through wgpu's naga→SPIR-V→driver
pipeline. coralReef's WGSL→native path (SASS for NVIDIA, GFX ISA for AMD) would:

- Eliminate the SPIR-V middleman (where many precision bugs originate)
- Enable architecture-specific optimizations (FMA patterns, register allocation)
- Remove dependency on proprietary driver shader compilers (PTXAS, ACO)

### Specific Shaders neuralSpring Would Benefit From

| Shader | Domain | Current Path | Benefit from coralReef |
|--------|--------|-------------|----------------------|
| 15 coralForge df64 shaders | Structure prediction (AlphaFold3-class) | `compile_shader_df64` → naga→SPIR-V | Native DF64 lowering avoids PTXAS bugs |
| `batched_eigh_nak_optimized_f64.wgsl` | Eigendecomposition (Anderson, spectral) | `compile_shader_f64` → naga→SPIR-V | Native f64 FMA patterns |
| `hmm_forward_f64.wgsl` | HMM phylogenetics | `compile_shader_f64` → naga→SPIR-V | Stable f64 exp/log on all hardware |
| `df64_core.wgsl` | All DF64 arithmetic | Included in df64 shaders | Key dependency — coralReef already has it |
| 6 neuralSpring metalForge shaders | Bio domains (stencil, RK45, PRNG, etc.) | `compile_shader` | Portability to non-Vulkan hardware |

### Bridge Status

neuralSpring's coralReef integration lives in `metalForge/forge/src/coralreef_bridge.rs`:

- **Compile-time path** (`coralreef` feature flag): Links `coral-reef` crate directly
- **Runtime discovery**: Probes `$XDG_RUNTIME_DIR/biomeos/coralreef.sock` (primary),
  then `$XDG_RUNTIME_DIR/ecoPrimals/*.json` capability manifests (fallback)
- **`CoralCompiler::auto()`**: Discovers and wraps coralReef if available
- **`compile_wgsl()`**: Returns `CoralResult<CompiledShader>`
- **Fallback**: Without coralReef, returns `CoralError::NotAvailable` — wgpu pipeline
  proceeds normally

**What's missing**: The runtime IPC client that would call coralReef's
`shader.compile.wgsl` JSON-RPC endpoint. The socket discovery is implemented but
actual compilation requests are not yet wired.

---

## Part 2: Precision Lessons Relevant to coralReef

### The `enable f64;` PTXAS Bug

**Discovery**: neuralSpring's comprehensive validation (55 fused GPU ops) uncovered
that `enable f64;` in WGSL causes NVIDIA PTXAS on Ada Lovelace (SM89) to silently
produce broken shaders that return zeros for complex multi-function shaders.

**Relevance to coralReef**: Since coralReef compiles WGSL→native **bypassing
PTXAS entirely**, this bug should not affect coralReef's compilation path. However:

1. coralReef should **never** emit `enable f64;` in intermediate WGSL if any path
   goes through naga→SPIR-V→driver
2. coralReef's `df64_preamble.wgsl` auto-prepend should verify it doesn't include
   the directive
3. If coralReef ever generates SPIR-V as an intermediate format, the same class of
   driver bugs may surface

### DF64 Precision Characteristics (from neuralSpring validation)

neuralSpring has the most comprehensive DF64 validation data in the ecosystem:

| Operation | N | CPU f64 vs GPU DF64 | Tolerance Used |
|-----------|---|---------------------|----------------|
| Variance | 5 | 0.0 (exact) | `TENSOR_EXACT_F32` (1e-6) |
| Pearson correlation | 1008 | 1.7e-5 | `GPU_DF64_TRANSCENDENTAL` (5e-4) |
| Shannon entropy | 100 | 1.6e-11 | sub-epsilon |
| chi² | small | exact | `TENSOR_EXACT_F32` |
| KL divergence | small | exact | `TENSOR_EXACT_F32` |
| HMM log-likelihood | 3×5000 | machine epsilon | `TENSOR_EXACT_F32` |

**Key finding**: DF64 5-accumulator reductions on large N (>1000) accumulate
measurable error (~1e-5). This is inherent to 48-bit mantissa arithmetic and
cannot be eliminated by better compilation — it's a precision floor.

coralReef's `Fp64Strategy` correctly routes between native f64 and DF64 based on
hardware capability. neuralSpring's tolerance data can inform coralReef's
precision documentation.

### Ada Lovelace (SM89) f64 Behavior

Ada Lovelace (RTX 40xx, consumer) has a **1/64 FP64 rate**. barraCuda classifies
this as `F64NativeNoSharedMem` → `Fp64Strategy::Hybrid`:

- Native f64 works for simple operations
- DF64 is preferred for throughput-sensitive reductions
- The `enable f64;` bug only affects the naga→SPIR-V→PTXAS path, not native f64
  capability itself

For coralReef's SASS backend: SM89 supports f64 natively in registers but shared
memory operations with f64 may have performance cliffs. neuralSpring's
`diagnose_f64_regression.rs` diagnostic binary can serve as a smoke test for any
new compilation backend.

---

## Part 3: Cross-Spring Shader Inventory

### Shaders neuralSpring Has Validated That coralReef Should Know About

| Category | Shader | Domain | Validated Checks | Status |
|----------|--------|--------|-----------------|--------|
| **Precision** | `df64_core.wgsl` | All DF64 ops | 37/37 GPU + 67/67 CPU | In coralReef corpus |
| **Precision** | `fused_welford_f64.wgsl` | Stable variance | 14/14 | hotSpring lineage |
| **Bio** | `hmm_forward_f64.wgsl` | HMM | 13/13 | wetSpring lineage |
| **Bio** | `fst_variance_decomposition.wgsl` | Population genetics | 7/7 | wetSpring lineage |
| **Bio** | `batch_fitness_eval.wgsl` | Evolutionary fitness | 20/20 | neuralSpring evolved |
| **Bio** | `stencil_cooperation.wgsl` | QS spatial | 3/3 | neuralSpring evolved |
| **Bio** | `wright_fisher_step.wgsl` | Pop genetics | 4/4 | neuralSpring evolved |
| **Linalg** | `batched_eigh_nak_optimized_f64.wgsl` | Eigendecomp | 9/9 | hotSpring lineage |
| **ODE** | `rk45_adaptive.wgsl` | ODE integration | 6/6 | neuralSpring evolved |
| **PRNG** | `xoshiro128ss.wgsl` | GPU random | 5/5 | neuralSpring evolved |
| **Stability** | `logsumexp_reduce.wgsl` | HMM/softmax | 5/5 | neuralSpring evolved |
| **Structure** | 15 coralForge df64 shaders | AlphaFold3-class | 37/37 | coralForge lineage |

**Total validated shader checks from neuralSpring**: 163+ across 27 shaders.

### coralReef Corpus Alignment

8 neuralSpring shaders are already in coralReef's corpus (Iteration 29). The
remaining neuralSpring-evolved shaders (`xoshiro128ss`, `logsumexp_reduce`,
`stencil_cooperation`, `rk45_adaptive`, `wright_fisher_step`) are candidates for
inclusion once absorbed by barraCuda.

---

## Part 4: What neuralSpring Provides for coralReef Testing

### Validation Infrastructure

neuralSpring offers the **most comprehensive WGSL shader validation** in the
ecosystem: 233 binaries spanning 23 domains. If coralReef implements a new
compilation backend, neuralSpring's validation suite can serve as an end-to-end
correctness test:

```bash
cargo run --release --bin validate_barracuda_dispatch_parity  # 55 fused GPU ops
cargo run --release --bin validate_modern_cross_spring        # 68 cross-spring checks
cargo run --release --bin validate_gpu_pure_workload_all      # 11 pure GPU domains
```

### Diagnostic Binary

`src/bin/diagnose_f64_regression.rs` tests f64 shader compilation with and without
`enable f64;`, DF64 sum and variance shaders, and specific fused operations.
Useful for validating any new WGSL→native compilation path.

---

## Part 5: Recommendations for coralReef

| Priority | Recommendation |
|----------|---------------|
| **P1** | Verify `df64_preamble.wgsl` auto-prepend does not include `enable f64;` |
| **P1** | Ensure coralReef's SASS backend handles multi-function f64 shaders (>20 functions) correctly on SM89 |
| **P2** | Add neuralSpring's `diagnose_f64_regression` test vectors to coralReef's CI |
| **P2** | Implement `shader.compile.wgsl` JSON-RPC — neuralSpring's bridge is ready to call it |
| **P3** | Consider including neuralSpring's 6 bio/ODE/PRNG shaders in the coralReef corpus |
| **P3** | Document DF64 precision floors (1e-5 at N>1000) in coralReef's precision architecture docs |

---

## Part 6: Metrics

| Metric | Value |
|--------|-------|
| Validated WGSL shaders | 27 (12 upstream + 15 coralForge + 6 local) |
| Total GPU shader checks | 163+ |
| Domains covered | 23 (26 papers + 6 baseCamp + 5 WDM + coralForge) |
| Hardware tested | RTX 4070 (Vulkan, proprietary) + TITAN V (NVK, open-source) |
| coralReef bridge | Implemented (metalForge), discovery wired, IPC client pending |
| Precision tiers validated | F32, F64, DF64 (full matrix) |

---

*This handoff is unidirectional: neuralSpring → coralReef. No response expected.*
*License: AGPL-3.0-or-later*
