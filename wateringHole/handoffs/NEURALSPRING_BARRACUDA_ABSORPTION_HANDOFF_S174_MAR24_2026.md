<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring → barraCuda/toadStool Absorption Handoff — S174

**Date**: March 24, 2026 (Session S174)
**From**: neuralSpring
**To**: barraCuda team, toadStool team
**License**: AGPL-3.0-or-later
**Context**: Comprehensive audit execution; all findings resolved. This handoff
documents absorption candidates, evolution learnings, and upstream requests from
the neuralSpring perspective after 174 sessions of validation work.

---

## 1. Absorption Candidates — Generic f64 WGSL Ops

These f64 WGSL shaders in `metalForge/shaders/` are **generic math** used across
multiple neuralSpring domains. They belong in `barracuda::ops` so all springs
can lean on upstream instead of maintaining local copies.

| Shader | Proposed Location | Used By | Priority |
|--------|------------------|---------|----------|
| `gelu_f64.wgsl` | `ops::activation::gelu_f64` | coral_forge, transformer | **P0** |
| `sigmoid_f64.wgsl` | `ops::activation::sigmoid_f64` | coral_forge, regulatory_network | **P0** |
| `layer_norm_f64.wgsl` | `ops::norm::layer_norm_f64` | transformer, coral_forge | **P0** |
| `softmax_f64.wgsl` | `ops::attention::softmax_f64` | transformer, coral_forge, MHA | **P0** |

**Current state**: neuralSpring has validated these extensively (18+ checks per
shader, multi-GPU parity confirmed on RTX 4070 + TITAN V NVK). The shaders
follow the `compile_shader_df64` convention established in barraCuda v0.3.7.
Once absorbed, neuralSpring would lean on upstream and retire local copies.

## 2. Absorption Candidate — `domain-fold` Feature Pack

AlphaFold3/structural biology WGSL shaders are too domain-specific for generic
`ops` but are generic **within** the protein folding domain. Proposed as a gated
feature pack like `domain-nn`, `domain-esn`, `domain-genomics`:

| Shader | Operation | Notes |
|--------|-----------|-------|
| `triangle_mul_outgoing_f64.wgsl` | Triangle attention (outgoing) | Pair representation |
| `triangle_mul_incoming_f64.wgsl` | Triangle attention (incoming) | Pair representation |
| `outer_product_mean_f64.wgsl` | Outer product mean | MSA → pair |
| `sdpa_scores_f64.wgsl` | Scaled dot-product scores | IPA attention |
| `sdpa_softmax_f64.wgsl` | SDPA softmax | IPA attention |
| `sdpa_apply_f64.wgsl` | SDPA value application | IPA attention |
| `triangle_attention_f64.wgsl` | Full triangle attention | Pair update |
| IPA score computation | Invariant Point Attention | Backbone |
| Backbone update | Rigid-body frame updates | Structure module |
| Torsion angle computation | Side-chain angles | Structure module |

**Validation status**: All shaders validated in `coral_forge` experiments
(nF-01, nF-02, nF-03). Group B in `specs/EVOLUTION_MAPPING.md`.

## 3. Upstream Contract Pinning — Pattern for Ecosystem

S174 introduced a pattern where neuralSpring **pins expected values** of
barraCuda tolerance constants and verifies them at validation time:

```rust
pub const UPSTREAM_HYDRO_CROP_COEFFICIENT: f64 = 1e-6;
pub const UPSTREAM_PHYSICS_ANDERSON_EIGENVALUE: f64 = 1e-10;
pub const UPSTREAM_BIO_DIVERSITY_SHANNON: f64 = 1e-10;
pub const UPSTREAM_BIO_DIVERSITY_SIMPSON: f64 = 1e-10;
```

If barraCuda changes `barracuda::tolerances::HYDRO_CROP_COEFFICIENT_ABS_TOL`
from `1e-6` to something else, `validate_toadstool_s93_barracuda_extraction`
will fail with an explicit message. **Recommendation**: barraCuda should
consider publishing a stability contract for tolerance constants, or at minimum
documenting which tolerances are part of the public API.

## 4. Bessel Correction — Algorithmic Difference Documentation

neuralSpring S174 documented a known algorithmic difference:

- **GPU**: `barracuda::ops::bio::MultiObjFitnessGpu` uses **sample** std (÷(n-1))
- **CPU**: `multi_objective_fitness` uses **population** std (÷n)

For genome chunks of length 10 (`genome_len`=40 / `n_objectives`=4), this
yields ~2e-3 observed absolute gap. neuralSpring's `GPU_MULTI_OBJ_BESSEL_F64`
(3e-3) tolerance accounts for this with 50% margin.

**Question for barraCuda team**: Is the sample-vs-population divergence
intentional? If so, document it in `ops::bio` docs. If not, aligning to
population std would let neuralSpring tighten this tolerance to ~1e-6.

## 5. Typed Error Convention — Ecosystem Alignment

neuralSpring now uses `thiserror`-based typed errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("gpu device: {reason}")]
    Device { reason: String },
    #[error("shader compilation: {reason}")]
    Shader { reason: String },
    #[error("buffer: {reason}")]
    Buffer { reason: String },
    // ...
}
```

barraCuda has `barracuda::error::BarracudaError`. Suggestion: consider
structured variants (like above) rather than string-only variants, so
downstream springs can pattern-match on error kind without string parsing.
This would eliminate `.map_err(|e| e.to_string())` at every spring boundary.

## 6. Self-Knowledge Pattern — Lessons for Primals

S174 enforced the self-knowledge principle aggressively:

1. **No cross-primal name hints in config**: Primals discover each other at
   runtime via capability-based IPC, not compile-time constants
2. **Neutral origin strings**: RPC responses describe the operation semantically
   (`"stats.variance → dispatch"`) not the physical stack path
3. **Gated coupling**: Optional integrations (visualization push to petalTongue)
   controlled by environment variables, not hardcoded function calls

**Recommendation**: barraCuda and toadStool should audit for similar patterns.
A primal's code should never embed the name of another primal in a way that
creates a compile-time dependency on that primal's existence.

## 7. Current neuralSpring ↔ barraCuda Surface

| Metric | Count |
|--------|-------|
| barraCuda submodules used | 45+ |
| Files with barracuda imports | 216 |
| Upstream rewires (local → barracuda) | 46 |
| barraCuda version | v0.3.7 (`default-features = false`) |
| Features enabled | `gpu`, `domain-nn`, `domain-esn`, `domain-genomics`, `domain-timeseries` |
| Features dropped (unused) | `domain-pde`, `domain-snn`, `domain-vision` |
| Validation binaries exercising barracuda | 12 (268 checks, all PASS) |
| GPU tensor validators | 7 (98+ checks, all PASS) |
| Named tolerance constants | 232+ |
| Upstream contract pins | 4 |
| Duplicate math | 0 (2 intentional divergences documented) |

## 8. Evolution Roadmap

### Near-term (next 1-2 sessions)
- Absorb 4 generic f64 ops into `barracuda::ops` (P0 above)
- Resolve Bessel correction question
- Continue `Result<T, String>` → typed error migration (neuralSpring side)

### Medium-term
- `domain-fold` feature decision
- `mha.rs` thin wrapper retirement (once barraCuda MHA API stabilizes)
- `cpu_parity_references.json` regeneration from clean HEAD

### Long-term
- All coral_forge shaders absorbed into barraCuda
- Zero local WGSL in neuralSpring (full lean-on-upstream)
- Typed error convention across all ecosystem boundaries

## 9. Files for Reference

| Path | Content |
|------|---------|
| `specs/BARRACUDA_USAGE.md` | Full 1207-line audit of every barracuda import |
| `specs/BARRACUDA_REQUIREMENTS.md` | Gap analysis (all P0/P1 resolved) |
| `specs/EVOLUTION_MAPPING.md` | Tier A/B/C mapping of every module |
| `specs/TOADSTOOL_HANDOFF.md` | 707-line shortcoming tracker (all 17 resolved) |
| `src/tolerances/mod.rs` | Named constants + upstream contract pins |
| `src/tolerances/registry.rs` | Runtime-introspectable registry |
| `metalForge/shaders/` | Local WGSL shaders (absorption candidates) |
| `CONTRIBUTING.md` | Quality standards and barraCuda evolution conventions |
