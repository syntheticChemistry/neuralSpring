# neuralSpring V131 Audit Remediation Handoff — April 17, 2026

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Spring:** neuralSpring v0.1.0 (S181+)
**Date:** 2026-04-17
**Scope:** Comprehensive audit remediation — manifest reconciliation, code quality,
GPU evolution, tolerance hygiene, upstream hand-backs

---

## Changes Made (neuralSpring)

### Critical: Manifest & Graph Reconciliation

1. **`graphs/neuralspring_deploy.toml`** — Updated `proto_nucleate` reference
   from stale `neuralspring_inference_proto_nucleate` to
   `downstream_manifest::neuralspring` per graph consolidation handoff.
   Updated header comment to point at consolidated manifest.

2. **`src/bin/validate_composition_evolution.rs`** — Updated proto-nucleate
   reference in doc comment and the `include_str!` validation check to match
   the new `downstream_manifest::neuralspring` reference.

3. **`src/bin/validate_nucleus_composition.rs`** — Updated proto-nucleate
   doc reference.

4. **`src/validation/composition.rs`** — Updated proto-nucleate doc reference.

5. **`src/niche.rs`** — Updated module doc to reference downstream manifest path.

### High: Code Quality

6. **`src/bin/bench_basecamp_parity.rs`** — Replaced `Box<dyn std::error::Error>`
   with `String` to eliminate dyn dispatch in error handling.

7. **`src/bin/bench_rewire_evolution.rs`** — Same `Box<dyn>` elimination.

8. **`src/loss_landscape.rs`** — Added module-level documentation justifying
   `&dyn Fn(&[f64]) -> f64` usage (heterogeneous closure captures, negligible
   vs O(n^3) eigensolve cost).

### High: GPU Evolution

9. **`src/nucleus_pipeline/executor.rs`** — Fixed `GpuPreferred` handling:
   stages marked `GpuPreferred` now route through the GPU `Dispatcher` when
   a GPU is available, falling back to CPU otherwise. Previously only `GpuOnly`
   triggered GPU dispatch.

### Medium: Tolerance Hygiene

10. **`src/tolerances/gpu.rs`** — Added two named constants:
    - `GPU_HMM_LOG_LIKELIHOOD_F64` (0.05) — 10× tighter than f32 for f64 shaders
    - `GPU_HMM_LOG_LIKELIHOOD_F32_EXTENDED` (1.0) — 2× base for long sequences

11. **`src/tolerances/registry.rs`** — Registered both new constants.

12. **Validation binaries updated** to use named constants instead of ad-hoc
    multipliers (`* 0.1`, `* 2.0`):
    - `validate_gpu_hmm_forward.rs` (3 sites)
    - `validate_barracuda_hmm_f64.rs` (4 sites)
    - `validate_gpu_phase_c.rs` (1 site)
    - `validate_gpu_phase_b.rs` (1 site)

### Medium: Documentation

13. **`specs/README.md`** — Updated barraCuda version reference from v0.3.11
    to v0.3.12.

14. **`src/wdm_esn/multi_head.rs`** — Documented `ESN_TIKHONOV_REGULARIZATION`
    as a physics/ML constant, not a validation tolerance.

---

## Changes Made (primalSpring — cross-spring)

### Critical: Downstream Manifest

15. **`graphs/downstream/downstream_manifest.toml`** — neuralspring entry:
    - Added `nest_atomic` to `fragments` (was missing; deploy graph includes it)
    - Added `nestgate` to `depends_on` (implied by nest_atomic)

### Critical: Validation Manifest

16. **`graphs/spring_validation/spring_validate_manifest.toml`** — neuralspring entry:
    - Changed `domain` from `"neural"` to `"science"`
    - Aligned capabilities from `neural.*` namespace to `science.*` namespace
      (`science.spectral_analysis`, `science.anderson_localization`,
      `science.hessian_eigen`) matching deploy and proto-nucleate manifests

---

## Hand-Backs to Upstream Primals

### barraCuda: `plasma_dispersion` Feature-Gate Bug (Gap 9)

**Issue:** `special/plasma_dispersion.rs` unconditionally imports from
`ops::lattice::cpu_complex::Complex64`, but `ops::lattice` is gated behind
`#[cfg(feature = "domain-lattice")]`. neuralSpring works around this by
enabling `domain-lattice`, but the fix belongs upstream.

**Requested fix:** Either feature-gate `plasma_dispersion` behind
`domain-lattice`, or make `Complex64` available without the lattice feature.

### primalSpring: particle_profile Mismatch

**Issue:** `NUCLEUS_SPRING_ALIGNMENT.md` lists neuralSpring's particle profile
as `balanced`, but `downstream_manifest.toml` sets `proton_heavy`. The manifest
is treated as parameterized truth; the alignment doc should be updated.

---

## Remaining Open Gaps (unchanged)

- Gap 1: Squirrel provider registration (upstream-blocked)
- Gap 2: barraCuda direct import → IPC migration (deferred)
- Gap 3: coralReef shader compilation via IPC (open)
- Gap 4: toadStool compute dispatch via IPC (open)
- Gap 5: NestGate weight storage (open)
- Gap 6: BearDog/Songbird BTSP session (wip)
