# neuralSpring → barraCuda Evolution Request: Typed Errors, Domain Fold, Generic Ops

**Date**: March 24, 2026
**From**: neuralSpring (S173)
**To**: barraCuda / toadStool team
**License**: AGPL-3.0-or-later

## Context

neuralSpring S173 introduced `thiserror`-based typed errors at the spring level.
This surfaces a pattern the ecosystem should converge on: springs and primals
returning typed error enums rather than `Result<T, String>`.

## Request 1: Generic Activation/Norm Ops for `barracuda::ops`

neuralSpring maintains these f64 WGSL shaders in `metalForge/shaders/` that are
generic enough for all springs:

| Shader | Proposed Location | Notes |
|--------|------------------|-------|
| `gelu_f64.wgsl` | `ops::activation::gelu_f64` | Used by coral_forge, transformer |
| `sigmoid_f64.wgsl` | `ops::activation::sigmoid_f64` | Used by coral_forge, regulatory |
| `layer_norm_f64.wgsl` | `ops::norm::layer_norm_f64` | Used by transformer, coral_forge |
| `softmax_f64.wgsl` | `ops::attention::softmax_f64` | Used by transformer, coral_forge, MHA |

These are currently duplicated between springs. Absorbing them into barraCuda
ops would let all springs lean on upstream.

## Request 2: `domain-fold` Feature Pack

AlphaFold3/structural biology operations are too specialized for generic `ops`
but would benefit from a gated domain feature (like existing `domain-nn`):

- Triangle attention (incoming/outgoing multiply)
- Outer product mean
- IPA (Invariant Point Attention) scores
- Backbone update, torsion angle computation
- MSA row/column attention

Proposed: `domain-fold` feature in barraCuda, gated like `domain-nn`.

## Request 3: Typed Error Convention

neuralSpring now uses:
```rust
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("gpu device: {reason}")]
    Device { reason: String },
    // ...
}
```

Suggest barraCuda consider adopting a similar pattern for its public API errors
(currently `barracuda::error::BarracudaError`), especially for tensor operations
that springs map through. A shared convention reduces `.map_err(|e| e.to_string())`
boilerplate at spring boundaries.

## Metrics

neuralSpring uses 45+ barraCuda submodules across 200+ files. No duplicate math.
All Group A shaders (25) already absorbed. Feature selection: `gpu`, `domain-nn`,
`domain-esn`, `domain-genomics`, `domain-timeseries` (unused domains dropped).
