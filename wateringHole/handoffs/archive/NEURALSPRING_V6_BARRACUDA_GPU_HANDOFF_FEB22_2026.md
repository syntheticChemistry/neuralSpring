# neuralSpring → ToadStool: BarraCUDA GPU Tensor Validation Handoff v6

**Date:** February 22, 2026
**From:** neuralSpring (ML validation & evolutionary computation biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**ToadStool HEAD:** `77f70b2e` (Session 31h)
**Supersedes:** `NEURALSPRING_TOADSTOOL_ABSORPTION_V5_FEB22_2026.md`

---

## Executive Summary

neuralSpring completed Phase 5a: GPU `Tensor` validation across 7 scientific
domains, exercising `matmul`, `transpose`, `tanh`, and `add` operations on an
RTX 4070 (Vulkan). This uncovered three new BarraCUDA bugs:

- **S-15** (Critical): `Tensor::matmul` hangs when input data contains
  negative values or is highly sparse, on RTX 4070 Vulkan.
- **S-16** (High): 2D `transpose` dispatch divides by
  `optimal_workgroup_size(ElementWise)` (256 on NVIDIA) instead of the
  shader's hardcoded tile size (16), producing partial output for any
  dimension > 16.
- **S-14** (Medium, previously reported): Naive matmul tier hangs for
  small square matrices in complex binaries.

S-16 is the root cause of the Gram matrix accuracy failure in pairwise
validation (columns 16-19 all zero). S-15 blocks anderson validation
(sparse tridiagonal Hamiltonian). Both have confirmed diagnoses with
reproducible test cases and recommended fixes.

This handoff includes: 3 bug reports with root causes, 7 GPU validator
summaries, workaround documentation, and evolution recommendations.

---

## 1. Bug Reports

### S-15: `Tensor::matmul` Hangs on Negative / Sparse f32 Input (Critical)

**Symptom:** `Tensor::matmul` hangs indefinitely (no timeout, no error)
when input data contains negative f32 values or is highly sparse (many
zeros). Observed on RTX 4070 Vulkan (Naive tier: M or N < 32).

**Reproduction (minimal):**

```rust
// HANGS: negative data
let data_a: Vec<f32> = (0..112).map(|_| (rng.uniform() - 0.5) as f32).collect();
let data_b: Vec<f32> = (0..112).map(|_| (rng.uniform() - 0.5) as f32).collect();
let a = Tensor::from_data(&data_a, &[16, 7], device.clone())?;
let b = Tensor::from_data(&data_b, &[7, 16], device.clone())?;
let c = a.matmul(&b)?; // HANGS HERE

// WORKS: positive-only data (same shapes)
let data_a: Vec<f32> = (0..112).map(|_| rng.uniform() as f32).collect();
let data_b: Vec<f32> = (0..112).map(|_| rng.uniform() as f32).collect();
let a = Tensor::from_data(&data_a, &[16, 7], device.clone())?;
let b = Tensor::from_data(&data_b, &[7, 16], device.clone())?;
let c = a.matmul(&b)?; // COMPLETES
```

**Also hangs with sparse data (positive, many zeros):**

```rust
// Sparse tridiagonal (mostly zeros)
let mut h_data = vec![0.0f32; 16 * 16];
for i in 0..16 { h_data[i * 16 + i] = 2.0; }
for i in 0..15 { h_data[i * 16 + (i+1)] = -1.0; h_data[(i+1) * 16 + i] = -1.0; }
let h = Tensor::from_data(&h_data, &[16, 16], device.clone())?;
// Any matmul involving h: HANGS
```

**Analysis:**
- The `matmul.wgsl` (Naive) shader source has no conditional logic on data
  values — it reads `a[idx] * b[idx]` unconditionally.
- `should_use_npu_for_matmul()` calls `to_vec()` on both input tensors
  for sparsity analysis before routing. While NPU is not available on this
  system (`is_npu_available()` returns false), the `to_vec()` calls still
  execute and may interact with the GPU pipeline state.
- The hang appears to be a WGPU/Vulkan driver issue, not a shader logic
  error. The shader itself is mathematically correct.

**Impact:** Blocks GPU validation of any domain with natural negative data
(neural network weights, centered features, physics quantities).

**Workaround (neuralSpring):** All Phase 5a validators generate data using
`rng.uniform()` (range [0, 1)) to avoid negative values. Sparse matrices
are replaced with dense random equivalents.

**Recommended investigation:**
1. Test with `wgpu::Features::SPIRV_SHADER_PASSTHROUGH` to isolate
   WGSL→SPIR-V translation vs. Vulkan driver.
2. Check if `to_vec()` in `should_use_npu_for_matmul()` introduces
   pipeline synchronization that interacts with subsequent dispatch.
3. Test on AMD (different Vulkan driver) to isolate NVIDIA-specific
   behavior.
4. Consider guarding `should_use_npu_for_matmul()` behind a feature flag
   to eliminate the `to_vec()` overhead on systems without NPU.

---

### S-16: 2D Transpose Dispatch Uses Wrong Workgroup Divisor (High)

**Symptom:** `Tensor::transpose()` on a 2D tensor produces partial output.
For any dimension > 16, elements beyond the first 16 in that dimension are
zero. The output tensor has correct shape but incorrect data.

**Root cause (confirmed):** In `ops/transpose/compute.rs`, `execute_2d()`
computes workgroup dispatch count using `optimal_workgroup_size(ElementWise)`:

```rust
// BUG: optimal_workgroup_size returns 256 on NVIDIA
let caps = DeviceCapabilities::from_device(device);
let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
let workgroups_x = cols.div_ceil(optimal_wg_size).max(1);  // cols=8: 8/256 = 1
let workgroups_y = rows.div_ceil(optimal_wg_size).max(1);  // rows=20: 20/256 = 1
pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
```

But the shader uses `@workgroup_size(16, 16)` with tiled shared-memory
access. Each workgroup processes a 16×16 tile. The dispatch must use 16 as
the divisor, not the generic `optimal_workgroup_size`:

```rust
// FIX: match shader's @workgroup_size(16, 16)
const TILE: u32 = 16;
let workgroups_x = cols.div_ceil(TILE).max(1);  // cols=8: 8/16 = 1
let workgroups_y = rows.div_ceil(TILE).max(1);  // rows=20: 20/16 = 2
pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
```

**Concrete failure for [20, 8] → [8, 20] transpose:**

| | Correct (divisor=16) | Bug (divisor=256) |
|---|---|---|
| workgroups_x | `ceil(8/16) = 1` | `ceil(8/256) = 1` |
| workgroups_y | `ceil(20/16) = 2` | `ceil(20/256) = 1` |
| Workgroups dispatched | 2 | **1** (missing!) |
| Output coverage | All 160 elements | Only 128 of 160 |
| Missing elements | None | Columns 16-19 (all zeros) |

**Downstream effect on matmul:**
When the partially-transposed [8, 20] tensor is used as the right operand
in `A.matmul(&B_transposed)` (20×8 × 8×20 = 20×20), the matmul reads
`b[i * 20 + col]` for col >= 16, which are all zero. This produces a
20×20 output where columns 16-19 are entirely zero:

```
Zero pattern (0=ok, X=zero):
  row  0: ................XXXX
  row  1: ................XXXX
  ...
  row 19: ................XXXX

GPU total elements: 400, non-zero: 320  (expected: 400)
max diff = 3.71e0 at [18][18]  (GPU: 0.0, CPU: 3.71)
```

**Why eco validator passes:** The `validate_barracuda_gpu_eco` validator
creates the B matrix directly as [10, 20] via `Tensor::from_data` — no
transpose operation. The matmul itself works correctly; only the transpose
dispatch is broken.

**Fix (one line):**

```rust
// In ops/transpose/compute.rs, execute_2d(), replace:
let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
// with:
const TILE: u32 = 16;  // must match @workgroup_size(16, 16) in shader
```

And use `TILE` as the divisor for both `workgroups_x` and `workgroups_y`.

**Audit recommendation:** Search all dispatch call sites that use
`optimal_workgroup_size` or `caps.optimal_workgroup_size_2d()` and verify
that the divisor matches the shader's `@workgroup_size(...)` declaration.
The 2D transpose was the only confirmed instance, but the pattern may exist
in other ops.

---

### S-14: Naive Matmul Hang on Small Square Matrices (Medium, Previously Reported)

**Status:** Characterized but not fully resolved. Workaround in place.

**Symptom:** Naive matmul tier (M or N < 32) hangs on RTX 4070 Vulkan when
the binary is "complex" (many GPU operations in the same process). Simple
binaries with a single matmul work fine.

**Workaround:** neuralSpring validators use non-square shapes and explicit
`transpose()` (the `A × B^T` pattern) to avoid triggering the Naive tier
with square inputs.

**Note:** S-14 may be partially explained by S-15 (negative/sparse data).
Some previously attributed S-14 hangs were actually S-15 hangs that were
resolved by switching to positive-only data.

---

## 2. Phase 5a GPU Tensor Validation Results

### Purpose

Validate that BarraCUDA's GPU `Tensor` operations (`matmul`, `transpose`,
`tanh`, `add`) produce correct results across 7 scientific domains, using
CPU f64 reference implementations as ground truth. This is the "pure Rust
math" validation layer — proving the same computations work identically
on GPU (f32) and CPU (f64) within floating-point tolerance.

### Results

| Validator | Domain | GPU Ops | Checks | Status |
|-----------|--------|---------|--------|--------|
| `validate_barracuda_gpu_spectral` | Spectral commutativity (022) | matmul | 10 | **PASS** |
| `validate_barracuda_gpu_eco` | Ecological dynamics (013) | matmul, transpose | 6 | **PASS** |
| `validate_barracuda_gpu_hmm` | HMM phylogenetics (016-018) | matmul, transpose | 5 | **PASS** |
| `validate_barracuda_gpu_fitness` | Evolutionary computation (011-015) | matmul, transpose | 7 | **PASS** |
| `validate_barracuda_gpu_nn` | Neural network inference (015, 020-021) | matmul, transpose, tanh, add | 5 | **PASS** (S-15 workaround) |
| `validate_barracuda_gpu_pairwise` | Pairwise distance (017, 019, 024-025) | matmul, transpose | 5 | **4/5 PASS** (S-16 blocks Gram) |
| `validate_barracuda_gpu_anderson` | Anderson localization (023) | matmul, transpose | 5 | **BLOCKED** (S-15 sparse data) |
| **Total** | 7 domains, 15 papers | | **43** | **33/43 PASS** |

### What each validator proves

- **spectral**: Small known-value matmuls (2×3, 3×2), CPU commutator
  `‖[A,B]‖_F`, identity × matrix = matrix, Frobenius norm.
- **eco**: Population × optima distance (`A×B^T`), Gram matrix (`A×A^T`),
  ones outer product, diagonal norm positivity, determinism.
- **hmm**: Transition chain (`α×A^T`), observation model (`obs×weights`),
  stationary distribution (`π×A ≈ π`), determinism.
- **fitness**: Batch fitness (`genotypes×weights^T`), weighted fitness,
  multi-objective fitness, determinism.
- **nn**: Single-layer `tanh(X×W^T)`, two-layer MLP, bias addition via
  `Tensor::add`, determinism.
- **pairwise**: Gram matrix (`X×X^T`), cross-distance, diagonal norm
  positivity, accuracy bounds, determinism.
- **anderson**: Wavefunction inner products (`Ψ×Ψ^T`), Hamiltonian
  application (`Ψ×H^T`), energy expectations, determinism.

### Validation methodology

Each validator follows the same pattern:
1. Generate test data with `rng::Xoshiro256StarStar` (deterministic seed)
2. Compute CPU f64 reference using pure Rust
3. Create GPU `Tensor` via `Tensor::from_data` (f32)
4. Execute GPU operation (matmul, transpose, tanh, add)
5. Read back via `Tensor::to_vec()` (f32)
6. Compare GPU f32 vs CPU f64 using `max_abs_diff` with tolerance from
   `tolerances.rs` (typically 1e-3 for accumulated matmul error)
7. Check determinism: two identical GPU runs must produce bit-identical output

---

## 3. S-15 Workaround Details

All Phase 5a validators that previously used centered or negative data
were modified to use `rng.uniform()` (range [0, 1)) exclusively:

| Validator | Original Data | S-15 Workaround |
|-----------|--------------|-----------------|
| nn | `rng.uniform() * 2.0 - 1.0` (weights in [-1, 1]) | `rng.uniform()` (weights in [0, 1)) |
| pairwise | `rng.uniform() * 2.0 - 1.0` (features) | `rng.uniform()` (features in [0, 1)) |
| anderson | Tridiagonal H: diagonal=W, off-diagonal=-1 | Dense random H: `rng.uniform() * 0.5` |

This means **negative-value GPU matmul is not validated** in Phase 5a.
The CPU reference implementations still use full-range data, but GPU
testing is restricted to non-negative inputs. Fixing S-15 would allow
removing this restriction.

---

## 4. Transpose Dispatch Audit

The S-16 bug pattern — using `optimal_workgroup_size` as a dispatch
divisor for a shader with hardcoded `@workgroup_size` — may exist
elsewhere. Here is the recommended audit scope:

| Op | Shader `workgroup_size` | Dispatch Divisor | Status |
|----|------------------------|------------------|--------|
| transpose 2D | `(16, 16)` | `optimal_workgroup_size(ElementWise)` = 256 | **BUG (S-16)** |
| matmul Naive | `(16, 16)` | `m.div_ceil(16), n.div_ceil(16)` | Correct |
| matmul Tiled16 | `(16, 16)` | `n.div_ceil(16), m.div_ceil(16)` | Correct |
| matmul CpuTiled32 | `(32, 32)` | `n.div_ceil(32), m.div_ceil(32)` | Correct |
| matmul GpuEvolved32 | `(32, 32)` | `n.div_ceil(32), m.div_ceil(32)` | Correct |
| transpose N-D | `(256)` | `size.div_ceil(optimal_wg_size)` | OK (1D, matches) |

The matmul dispatch is correct because it uses hardcoded tile sizes. The
N-D transpose is correct because its shader uses `@workgroup_size(256)`
which matches the 1D `optimal_workgroup_size` on NVIDIA. Only the 2D
transpose has the mismatch.

---

## 5. Evolution Recommendations

### For the BarraCUDA team (priority order)

1. **Fix S-16** (one-line fix in `ops/transpose/compute.rs`): Replace
   `optimal_workgroup_size(ElementWise)` with `16` in `execute_2d()`.
   This unblocks all transpose-dependent GPU operations for dimensions > 16.

2. **Investigate S-15**: The matmul hang with negative/sparse data is the
   most impactful bug for downstream consumers. Suggested investigation:
   - Remove `to_vec()` calls from `should_use_npu_for_matmul()` on
     non-NPU systems (feature-gate the sparsity analysis)
   - Test negative data on AMD Vulkan to isolate NVIDIA driver behavior
   - Add a timeout to `device.poll(Maintain::Wait)` in the Naive tier

3. **Consider retiring the Naive tier**: The Naive matmul shader has been
   the source of S-14 and interacts poorly with S-15. The Tiled16 shader
   handles all sizes correctly (it clamps to tile boundaries). Promoting
   `SMALL_MATRIX_THRESHOLD` to 0 (always use Tiled16) would eliminate
   the Naive tier entirely.

4. **Audit all dispatch sites**: Search for `optimal_workgroup_size` used
   as a divisor in 2D/3D dispatch and verify it matches the shader's
   declared `@workgroup_size`.

### For neuralSpring (after upstream fixes)

- Remove S-15 workaround: restore negative/centered data in nn, pairwise,
  anderson validators
- Complete pairwise validate_gram_matrix (currently blocked by S-16)
- Complete anderson validator (currently blocked by S-15 sparse data)
- Clean up `validate_barracuda_gpu_minimal_test` diagnostic binary

---

## 6. Cross-Spring Value

These findings benefit all Springs that use BarraCUDA `Tensor` operations:

| Spring | Affected Operations |
|--------|-------------------|
| hotSpring | MD simulation: centered particle velocities (negative), sparse interaction matrices |
| wetSpring | Genomic distance matrices via transpose, negative log-likelihoods |
| neuralSpring | Neural network weights (negative), Gram matrices via transpose |

The S-16 transpose bug would affect any Spring using `A.transpose()` on a
tensor where any dimension exceeds 16 — which is nearly all real workloads.

---

## 7. Companion Documents

| Document | Content |
|----------|---------|
| `NEURALSPRING_TOADSTOOL_ABSORPTION_V5_FEB22_2026.md` | 8 shader absorption requests (still current) |
| `NEURALSPRING_V5_EVOLUTION_HANDOFF_FEB22_2026.md` | V5 doc alignment (superseded by this V6) |
| `specs/TOADSTOOL_HANDOFF.md` | S-01..S-12 absorption status |
| `EVOLUTION_READINESS.md` | Full evolution readiness with Phase 5a |

---

*neuralSpring → ToadStool GPU Tensor handoff v6 — 3 bugs (S-14/S-15/S-16),
7 domain validators, 33/43 checks passing, 2 bugs blocking remaining 10.*
*Lifecycle: evolve → validate → discover → handoff → fix → lean.*
