# neuralSpring V131 — Cross-Team Absorption Handoff

**Date**: April 17, 2026 (Session S181+)
**Author**: neuralSpring audit + composition evolution
**Audience**: barraCuda team, primalSpring team, biomeOS team, all spring teams
**License**: AGPL-3.0-or-later

---

## Purpose

This handoff captures everything neuralSpring learned during the comprehensive
audit and composition evolution phase (S181+). It documents patterns for primal
absorption, composition evolution findings, NUCLEUS deployment via biomeOS, and
upstream issues that affect the entire ecosystem.

---

## 1. For barraCuda Team

### 1.1 Absorption Candidates

neuralSpring has 43 WGSL shaders in `metalForge/shaders/`. Of these, 25 (Group A)
are generic bio/evolution kernels already absorbed or ready for `barracuda::ops::bio`.
Two remain as candidates:

| Shader | Target | Priority |
|--------|--------|----------|
| `linear_regression.wgsl` | `ops::linalg::linear_regression` | Medium |
| `matrix_correlation.wgsl` | `ops::correlation` | Medium |

### 1.2 Generic f64 Activations for All Springs

Six shaders in Group B are generic f64 activations/norms that benefit all springs,
not just neuralSpring. Propose for `barracuda::ops::activation` / `ops::norm`:

- `gelu_f64.wgsl` → `ops::activation::gelu_f64`
- `sigmoid_f64.wgsl` → `ops::activation::sigmoid_f64`
- `layer_norm_f64.wgsl` → `ops::norm::layer_norm_f64`
- `softmax_f64.wgsl` → `ops::attention::softmax_f64`
- `sdpa_scores_f64.wgsl` → `nn::attention::sdpa_f64`
- `attention_apply_f64.wgsl` → `nn::attention::apply_f64`

### 1.3 GpuPreferred Dispatch

neuralSpring added `MixedSubstrate::GpuPreferred` to `metalForge/forge/src/mixed.rs`.
This variant dispatches through GPU when available, falls back to CPU transparently.
Use case: batched matmul, IPR reduction — workloads that benefit from GPU but produce
correct results on CPU. The barraCuda team may want to absorb this pattern into the
upstream `MixedSubstrate` enum.

### 1.4 Named Tolerance Constants

Two new HMM tolerance constants added to `src/tolerances/gpu.rs`:
- `GPU_HMM_LOG_LIKELIHOOD_F64` (0.05) — 10× tighter than f32
- `GPU_HMM_LOG_LIKELIHOOD_F32_EXTENDED` (1.0) — 2× base for long sequences (T>100, N>4)

These were derived from empirical GPU validation runs. If barraCuda adopts HMM
tolerance constants upstream, these values should be the baseline.

### 1.5 Upstream Bug Report

`barracuda::spectral::plasma_dispersion` — when called with purely real arguments
(Im(z) = 0), the function diverges. Documented in V131 audit remediation handoff.
Not blocking any neuralSpring work but affects hotSpring plasma physics paths.

---

## 2. For primalSpring Team

### 2.1 Manifest Reconciliations Performed

neuralSpring reconciled three manifest discrepancies during audit:

1. **`downstream_manifest.toml`** — `neuralspring` entry was missing `nest_atomic`
   fragment and `nestgate` dependency. Fixed: fragments now include
   `["tower_atomic", "node_atomic", "nest_atomic", "meta_tier"]`, depends_on
   includes `nestgate`.

2. **`spring_validate_manifest.toml`** — `neuralspring` entry had `domain = "neural"`
   and `neural.*` capabilities. Fixed: `domain = "science"`, capabilities aligned to
   `["science.spectral_analysis", "science.anderson_localization", "science.hessian_eigen"]`.

3. **`neuralspring_deploy.toml`** — `proto_nucleate` pointed to stale standalone file.
   Fixed: `proto_nucleate = "downstream_manifest::neuralspring"`.

**Action**: primalSpring should verify these changes are in its local copy. The
`downstream_manifest.toml` changes are in primalSpring's tree directly.

### 2.2 Self-Discovery Pattern

neuralSpring's composition validators could not discover their own primal socket
because `CARGO_PKG_NAME` produces `neural-spring` while socket paths use the
niche name `neuralspring`. We added `primal_to_pkg_name()` in
`src/validation/composition.rs` that maps between these forms.

**Recommendation for primalSpring**: This mapping should be standardized. Either:
- All springs use hyphenated names in socket paths (breaking change), or
- primalSpring provides a canonical `niche_name ↔ cargo_pkg_name` registry
  that composition validators can query at runtime.

Current known mappings:
```
neuralspring ↔ neural-spring
hotspring ↔ hot-spring
wetspring ↔ wet-spring
airspring ↔ air-spring
groundspring ↔ ground-spring
healthspring ↔ health-spring
```

### 2.3 Science Baselines for IPC Parity

neuralSpring defined 4 science baselines in `src/validation/composition.rs`
(`science_baselines()`) for Tier 3 IPC round-trip validation:

| Capability | Baseline | Description |
|-----------|----------|-------------|
| `science.spectral_analysis` | ESD + IPR for 8×8 Wigner matrix | Spectral density, IPR value |
| `science.ipr` | IPR for 16-site Aubry-André model | Inverse participation ratio |
| `science.hessian_eigen` | 4D Rosenbrock Hessian eigenvalues | Sorted eigenvalues of Hessian |
| `science.disorder_sweep` | Anderson disorder scan W=[0.5,1.0,...,4.0] | Mean IPR per disorder strength |

These baselines are deterministic Rust-computed values. The IPC round-trip must
reproduce them within documented tolerances. This pattern (Rust baseline → IPC call
→ compare) is the composition parity standard.

**Recommendation**: Other springs should adopt this pattern for their own `science.*`
capabilities. primalSpring could provide a template or trait for baseline definitions.

### 2.4 Composition Validation Tiers

neuralSpring now implements all three validation tiers:

| Tier | What | Binary | Status |
|------|------|--------|--------|
| 1 | Python → Rust CPU parity | `validate_cpu_math_parity` | PASS (39/39) |
| 2 | Rust CPU → GPU parity | `validate_gpu_phase_*` | PASS (all phases) |
| 3 | Rust direct → IPC round-trip | `validate_science_composition` | PASS (4 baselines) |

The Tier 3 pattern:
1. Compute result locally in Rust (deterministic, no IPC)
2. Discover primal socket via 5-tier discovery
3. Call same capability via JSON-RPC IPC
4. Compare IPC result to Rust baseline within tolerance
5. Exit 0 (pass), 1 (fail), 2 (primal not running / skip)

Exit code 2 is critical for CI: composition validators can run in environments
where the primal mesh is not up, without failing the build.

### 2.5 Particle Profile Mismatch

`NUCLEUS_SPRING_ALIGNMENT.md` in the wateringHole lists neuralSpring's
`particle_profile` as `proton_core` but the deploy graph and downstream manifest
use `electron_shell` for the Tower Atomic layer. The alignment doc should be updated
to reflect the actual composition.

---

## 3. For biomeOS Team

### 3.1 neuralAPI Deployment Pattern

neuralSpring's deploy graph (`graphs/neuralspring_deploy.toml`) defines the full
NUCLEUS composition for biomeOS deployment:

- **Tower Atomic** (BearDog + Songbird): security + discovery mesh
- **Node Atomic** (ToadStool + barraCuda + coralReef): compute substrate
- **Nest Atomic** (NestGate): weight/model storage
- **Meta-tier** (biomeOS + Squirrel + petalTongue): orchestration + AI + UI

The graph uses `bonding_model = "Metallic"` with `trust_model = "InternalNucleus"`.

### 3.2 Orchestrator Socket Discovery

neuralSpring's composition validators check `$BIOMEOS_ORCHESTRATOR_SOCKET` as the
highest-priority discovery tier (above well-known paths and `$XDG_RUNTIME_DIR`).
biomeOS should document this env var as the canonical way for springs to discover
the orchestrator.

Full discovery order:
1. `$BIOMEOS_ORCHESTRATOR_SOCKET` (env override)
2. `$XDG_RUNTIME_DIR/neuralspring/neuralspring.sock` (XDG standard)
3. `/tmp/neuralspring.sock` (fallback)
4. `$XDG_RUNTIME_DIR/neural-spring/neural-spring.sock` (cargo pkg name variant)
5. `/tmp/neural-spring.sock` (cargo pkg name fallback)

### 3.3 Capability Surface

neuralSpring exposes 30 capabilities via `capability.list`:

**Science** (4): `science.spectral_analysis`, `science.anderson_localization`,
`science.hessian_eigen`, `science.disorder_sweep`

**Inference** (3): `inference.complete`, `inference.embed`, `inference.models`
(routed through Squirrel via `try_squirrel_route`)

**Health** (3): `health.liveness`, `health.readiness`, `health.check`

**Identity** (1): `identity.get`

**MCP** (1): `mcp.tools.list`

Plus 18 additional domain, provenance, and compute capabilities.

### 3.4 Deployment Validation

biomeOS can validate neuralSpring deployment by running:
```bash
cargo run --release --bin validate_science_composition
```
Exit 0 = full composition parity verified. Exit 2 = primal not running (expected
in bench/CI without mesh). Exit 1 = composition parity failure (investigate).

---

## 4. For All Spring Teams

### 4.1 Three-Tier Validation Pattern

Every spring should evolve toward this 3-tier validation stack:

| Tier | Validates | Pattern |
|------|-----------|---------|
| 1 | Python → Rust | Hardcoded baselines from documented Python runs |
| 2 | Rust CPU → GPU | Same computation, different substrate, tolerance comparison |
| 3 | Rust direct → IPC | Same computation, IPC round-trip, composition parity |

Tier 3 is the new layer. It proves that the science works not just in isolated
Rust, but through the full primal composition mesh — the path that biomeOS
will actually deploy.

### 4.2 Composition Evolution Cycle

The maturity cycle we validated:
1. Read manifest entry in `downstream_manifest.toml`
2. Wire IPC to primals by capability (not by identity)
3. Validate composition parity (Tier 3)
4. Discover gaps → hand back to primalSpring
5. Primals evolve → cycle continues

### 4.3 hotSpring Validation Binary Pattern

All validation binaries follow:
- `ValidationHarness` for structured pass/fail
- Named tolerances from centralized `tolerances/` module
- Exit 0 (pass), 1 (fail), 2 (honest skip — resource unavailable)
- No `.expect()` or `.unwrap()` in validation code
- `require!` macro for GPU ops (graceful CI on headless machines)

### 4.4 ecoBin Compliance Checklist

neuralSpring is fully ecoBin compliant. The checklist:
- [x] `#![forbid(unsafe_code)]` on all crates
- [x] Zero C dependencies (Tower Atomic: no ring, rustls, ed25519-dalek)
- [x] `cargo deny check` passes (deny.toml with C-dep bans)
- [x] `cargo fmt --check` clean
- [x] `cargo clippy -- -D warnings` (pedantic+nursery) zero warnings
- [x] Edition 2024, `rust-version = "1.87"`
- [x] `#[expect(lint, reason = "...")]` — zero `#[allow()]` in production
- [x] SPDX headers on all files
- [x] AGPL-3.0-or-later license
- [x] All files under 1000 LOC

### 4.5 Known Upstream Blockers

These affect all springs doing composition validation:

1. **`tensor.*` response keys inconsistent** — some barraCuda IPC returns use
   `"result"` key, others use `"data"`. Need standard.
2. **`compute.dispatch` result shape** — ToadStool returns flat array vs nested
   object depending on dispatch mode. Need canonical shape.
3. **Method catalog** — no standard way to enumerate which methods a primal
   supports beyond `capability.list`. Need `method.list` or similar.
4. **Standard error codes** — JSON-RPC error codes are ad-hoc per primal.
   Need ecosystem-wide error code registry.

---

## 5. Patterns Discovered for NUCLEUS Deployment

### 5.1 Fragment Resolution

Deploy graphs should use `resolve = true` in `[graph.metadata]` to inherit from
NUCLEUS fragments. Custom profiles apply only delta nodes on top of the fragment
base layer. See `primalSpring/graphs/profiles/` for 9 canonical examples.

### 5.2 Bonding Policy

Cross-atomic compositions must declare:
- `bond_type`: Metallic (shared electron pool) vs Ionic (transfer) vs Covalent (shared pair)
- `trust_model`: InternalNucleus (all primals trusted) vs ExternalBridge (verify at boundary)
- `encryption_tiers`: per-atomic boundary (Tower handles external TLS, internal is plaintext)

### 5.3 Capability-Based Discovery

Springs must never hardcode primal names or socket paths. Use:
- `discover_primal("capability.name")` for runtime discovery
- `by_capability` routing in IPC calls
- `capability.list` for self-description
- `health.liveness` + `health.readiness` for deployment health

---

## Summary

| Team | Key Actions |
|------|-------------|
| **barraCuda** | Absorb 2 remaining shaders, 6 generic f64 activations, `GpuPreferred` pattern, `plasma_dispersion` bug |
| **primalSpring** | Verify manifest reconciliations, standardize niche↔pkg name mapping, template science baselines, fix particle_profile in alignment doc |
| **biomeOS** | Document `$BIOMEOS_ORCHESTRATOR_SOCKET`, validate neuralSpring deployment via `validate_science_composition` |
| **All springs** | Adopt 3-tier validation, composition evolution cycle, exit-2 pattern, capability-based discovery |
