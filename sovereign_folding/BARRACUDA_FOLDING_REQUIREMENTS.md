# BarraCUDA Shader Requirements — Sovereign Folding

**Last Updated**: February 12, 2026
**Purpose**: Scope all WGSL shader primitives needed to port OpenFold3/AlphaFold
from PyTorch/CUDA to BarraCUDA/Vulkan for sovereign structure prediction.

---

## Architecture Overview

AlphaFold2/OpenFold3 has three major computational blocks:

```
Input Features → [Evoformer] → [Structure Module] → 3D Coordinates
     ↑               ↑                ↑
  MSA + Pair     48 blocks         8 iterations
  representations   (~80% compute)    (~15% compute)
```

The Evoformer is 48 identical blocks, each containing:
1. MSA row attention (with pair bias)
2. MSA column attention
3. MSA transition (feed-forward)
4. Outer product mean (MSA → pair update)
5. Triangle multiplication (outgoing)
6. Triangle multiplication (incoming)
7. Triangle attention (starting node)
8. Triangle attention (ending node)
9. Pair transition (feed-forward)

The Structure Module has 8 iterations of:
1. Invariant Point Attention (IPA)
2. Backbone update
3. Side-chain torsion prediction

---

## Existing BarraCUDA Primitives (Reusable)

| Primitive | Shader | Precision | Status | Folding Use |
|-----------|--------|-----------|--------|-------------|
| **GEMM** | `gemm_f64.wgsl` | f64 | ✅ Validated | Feed-forward layers, projections |
| **Batched GEMM** | (in gemm_f64) | f64 | ✅ Validated | Multi-head attention QKᵀ, AV |
| **Matvec** | (in gemm_f64) | f64 | ✅ Validated | Bias addition |
| **SVD** | `svd_f64.wgsl` | f64 | ✅ Validated | Structure refinement |
| **NMF** | `nmf_f64.wgsl` | f64 | ✅ NEW | Not directly used, but validates matrix ops pattern |
| **Attention (QKᵀ)** | `attention_matmul.wgsl` | f32 | ✅ Validated | Standard MHA (needs f64 variant) |
| **Softmax** | `attention_softmax.wgsl` | f32 | ✅ Validated | Attention normalization |
| **Attention (AV)** | `attention_apply.wgsl` | f32 | ✅ Validated | Weighted value aggregation |
| **FFT** | `fft_1d_f64.wgsl` | f64 | ✅ Validated | Not directly used |
| **Eigendecomposition** | `eigh_f64.wgsl` | f64 | ✅ Validated | Structure alignment (Kabsch) |
| **Cholesky** | `cholesky_f64.wgsl` | f64 | ✅ Validated | Gaussian processes, uncertainty |
| **LU decomposition** | `lu_decomp_f64.wgsl` | f64 | ✅ Validated | Linear solves in structure module |
| **Triangular solve** | `triangular_solve_f64.wgsl` | f64 | ✅ Validated | Backsubstitution |
| **Scalar reduction** | `ReduceScalarPipeline` | f64 | ✅ Validated | Loss computation, norms |
| **FusedMapReduce** | via ToadStool | f64 | ✅ Validated | Layer norm, statistics |

**Coverage**: 15 of ~25 required primitives already exist in BarraCUDA.

---

## New Primitives Needed

### Priority 1 — Evoformer Core (blocks Phase B)

#### 1. Triangle Multiplication (Outgoing)

```
Input:  pair[N_res × N_res × C_pair]
Output: pair[N_res × N_res × C_pair] (updated)

Algorithm:
  a = Linear(pair)  →  [N_res, N_res, C_hidden]
  b = Linear(pair)  →  [N_res, N_res, C_hidden]
  gate = Sigmoid(Linear(pair))  →  [N_res, N_res, C_pair]
  tri[i,j] = sum_k a[i,k] * b[j,k]   ← batched outer product over k
  pair += gate * Linear(tri)

Key operations: 2× Linear projections (GEMM), 1× einsum("ikc,jkc->ijc"),
                1× sigmoid, 1× gated addition
```

**WGSL shader needed**: `triangle_mul_outgoing_f64.wgsl`
- Workgroup: (16, 16, 1) per (i, j) pair
- Inner loop over k (N_res)
- Memory: pair repr ~N² × C_pair × 8 bytes

#### 2. Triangle Multiplication (Incoming)

Same as outgoing but transposes the contraction:
```
tri[i,j] = sum_k a[k,i] * b[k,j]   ← note index swap
```

**WGSL shader needed**: `triangle_mul_incoming_f64.wgsl`

#### 3. Triangle Attention (Starting Node)

```
Input:  pair[N_res × N_res × C_pair]
Output: pair[N_res × N_res × C_pair] (updated)

Algorithm:
  For each row i of the pair matrix:
    Q = Linear(pair[i, :, :])  →  [N_res, H, C_head]
    K = Linear(pair[i, :, :])  →  [N_res, H, C_head]
    V = Linear(pair[i, :, :])  →  [N_res, H, C_head]
    bias = Linear(pair[i, :, :])  →  [N_res, H]
    attn = softmax(Q @ K^T / sqrt(C_head) + bias)
    pair[i] += Linear(attn @ V)

This is standard MHA but applied row-by-row on the pair matrix
with a learned bias from the pair representation itself.
```

**WGSL shader needed**: `triangle_attention_f64.wgsl`
- Reuses attention pattern from `attention_matmul.wgsl` but with pair bias
- f64 precision needed for structural accuracy

#### 4. Triangle Attention (Ending Node)

Same as starting node but transposes: attention along columns instead of rows.

**WGSL shader needed**: same as #3 with transposed input

#### 5. Outer Product Mean (MSA → Pair)

```
Input:  msa[N_seq × N_res × C_msa]
Output: pair[N_res × N_res × C_pair] (updated)

Algorithm:
  a = Linear(msa)  →  [N_seq, N_res, C_a]
  b = Linear(msa)  →  [N_seq, N_res, C_b]
  outer = mean_seq(a[:, i, :] ⊗ b[:, j, :])  →  [N_res, N_res, C_a × C_b]
  pair += Linear(flatten(outer))

Key: outer product between two positions, averaged over all MSA sequences.
This is how evolutionary covariance becomes structural contact information.
```

**WGSL shader needed**: `outer_product_mean_f64.wgsl`
- Reduction over N_seq (typically 512)
- Outer product per (i, j) pair

#### 6. MSA Row Attention (with Pair Bias)

```
Standard multi-head attention applied to each row of the MSA,
with an additive bias from the pair representation.

Input:  msa[N_seq × N_res × C_msa], pair[N_res × N_res × C_pair]
Output: msa[N_seq × N_res × C_msa] (updated)

The pair bias is the critical difference from standard attention:
  attn[h, i, j] = Q[h,i] · K[h,j] / sqrt(d) + bias[h, i, j]
```

**WGSL shader needed**: `msa_row_attention_f64.wgsl`
- Extends existing attention shaders with additive pair bias

#### 7. MSA Column Attention

```
Attention applied to each column of the MSA (across sequences at each position).
No pair bias. Simpler than row attention.

Input:  msa[N_seq × N_res × C_msa]
Output: msa[N_seq × N_res × C_msa] (updated)
```

**WGSL shader needed**: `msa_col_attention_f64.wgsl`

### Priority 2 — Structure Module (blocks Phase B, later)

#### 8. Invariant Point Attention (IPA)

```
The most complex primitive. SE(3)-equivariant attention that operates on
both the pair representation AND the 3D backbone coordinates.

Input:  single[N_res × C_s], pair[N_res × N_res × C_z],
        backbone_frames[N_res × (rotation_3x3 + translation_3)]
Output: single[N_res × C_s] (updated)

Algorithm:
  Q, K, V projections (standard)
  + Q_points, K_points, V_points in 3D (projected through backbone frames)
  + pair bias (from pair representation)

  attn = softmax(
    w_l * Q·K / sqrt(d)
    + w_b * pair_bias
    + w_p * sum_p ||T_i @ q_p - T_j @ k_p||²  ← 3D point distance
  )

  output = attn @ V  +  T_i^{-1} @ (attn @ (T_j @ V_points))

Requires:
  - 3D rotation/translation operations (SO(3))
  - Frame composition and inversion
  - Point projection through frames
  - Distance computation in 3D
```

**WGSL shaders needed**:
- `ipa_attention_f64.wgsl` — main IPA kernel
- `frame_ops_f64.wgsl` — SO(3) rotation, translation, composition
- `point_projection_f64.wgsl` — project points through frames

This is the hardest primitive to port. f64 precision is critical here
because small rotational errors accumulate over 8 structure iterations.

#### 9. Backbone Update

```
Update backbone frames from single representation.
quaternion_to_rotation + translation update.

Input:  single[N_res × C_s], current_frames[N_res × 7] (quat + trans)
Output: updated_frames[N_res × 7]
```

**WGSL shader needed**: `backbone_update_f64.wgsl`

#### 10. Torsion Angle Prediction

```
Predict side-chain torsion angles from single representation.
ResNet → atan2(sin, cos) for each angle.

Input:  single[N_res × C_s]
Output: torsions[N_res × 7 × 2] (sin, cos for 7 angles)
```

**WGSL shader needed**: `torsion_predict_f64.wgsl`

### Priority 3 — Utility Primitives

#### 11. Layer Normalization (f64)

```
y = (x - mean) / sqrt(var + eps) * gamma + beta
Already have FusedMapReduce for mean/var. Need fused kernel for full LayerNorm.
```

**WGSL shader needed**: `layer_norm_f64.wgsl`

#### 12. GELU Activation (f64)

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

**WGSL shader needed**: `gelu_f64.wgsl` (or fused into transition blocks)

#### 13. Sigmoid (f64)

```
σ(x) = 1 / (1 + exp(-x))
Used in gating mechanisms throughout Evoformer.
```

**WGSL shader needed**: `sigmoid_f64.wgsl`

---

## Memory Budget (RTX 4070, 12 GB)

| Component | Small (128 res) | Medium (384 res) | Large (1024 res) |
|-----------|:-:|:-:|:-:|
| MSA repr (512 × N × 256) | 67 MB | 201 MB | 537 MB |
| Pair repr (N × N × 128) | 8 MB | 75 MB | 537 MB |
| Model weights (93M params) | 372 MB | 372 MB | 372 MB |
| Intermediate buffers | ~200 MB | ~1.5 GB | ~8 GB |
| **Total** | **~650 MB** | **~2.1 GB** | **~9.4 GB** |

Small and medium proteins fit comfortably. Large proteins require gradient
checkpointing (recompute intermediates instead of storing them).

---

## Porting Strategy

### Phase B.1 — f64 Attention (1-2 weeks)

Port existing f32 attention shaders to f64. This gives us:
- Triangle attention (starting + ending)
- MSA row attention (with pair bias)
- MSA column attention
- All feed-forward transitions (via GEMM + activation)

### Phase B.2 — Triangle Operations (2-3 weeks)

New shaders for triangle multiplication. These are unique to AlphaFold
and don't exist in standard transformer libraries. The einsum
`"ikc,jkc->ijc"` needs a custom batched outer-product kernel.

### Phase B.3 — Structure Module (3-4 weeks)

IPA is the hardest primitive. Requires:
- SO(3) frame operations at f64
- Point projection at f64
- Combined attention score from scalar, pair, and 3D point terms
- Frame update (quaternion arithmetic)

### Phase B.4 — Integration + Testing (2-3 weeks)

Wire all primitives into a complete Evoformer + Structure Module pipeline.
Validate against PyTorch reference (from Phase A eval) to bit-level.

---

## Sovereign Advantage

| Aspect | PyTorch/CUDA (OpenFold3) | BarraCUDA/WGSL (Sovereign) |
|--------|--------------------------|---------------------------|
| GPU vendor | NVIDIA only (CUDA) | Any Vulkan GPU (NVIDIA, AMD, Intel) |
| f64 precision | Throttled 1:64 on consumer | Native 1:2 via SHADER_F64 |
| Dependencies | Python + PyTorch + CUDA toolkit | Pure Rust + WGSL (zero C) |
| Deployment | pip install, ~5 GB | Single binary, ~50 MB |
| Data sovereignty | Upload to AlphaFold Server | Local only, no network required |
| Rate limits | 20 structures/day (AlphaFold Server) | Unlimited |
| Cost | $10K+ GPU (A100) or cloud rental | $600 RTX 4070 |

---

## Estimated Timeline

```
Phase A: Baseline evaluation ─────────── DONE (Feb 2026)
Phase B: Shader porting ──────────────── 2-3 months
Phase C: Sovereign MSA pipeline ──────── 1-2 months
Phase D: RNA/DNA extension ───────────── 3-6 months
Phase E: Training from scratch ───────── 6-12 months (requires NUCLEUS mesh)
```

Total to first sovereign protein fold: ~4-5 months from Phase B start.
