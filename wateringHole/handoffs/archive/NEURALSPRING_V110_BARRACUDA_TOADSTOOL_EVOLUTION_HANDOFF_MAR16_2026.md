# neuralSpring V110 → barraCuda / toadStool Evolution Handoff

**Date**: March 16, 2026
**From**: neuralSpring (Session 159, V110)
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V108–V110 (Sessions 157–159)

## Executive Summary

- neuralSpring consumes **13+ barraCuda modules** across **167 binaries** and **80+ library files**
- All 17 shortcomings (S-01–S-17) are resolved upstream — zero pending blockers
- **Zero C dependencies** in neuralSpring workspace (reqwest/ring eliminated via Tower Atomic)
- Only external non-Rust dep: `cc` (build-time) via `blake3` in barraCuda — `pure` feature requested
- neuralSpring adopts `OrExit<T>`, `deny.toml`, `#[expect(reason)]`, `temp-env`, structured logging
- Patterns ready for barraCuda/toadStool absorption: `OrExit`, `deny.toml`, `ValidationHarness`

## 1. barraCuda Consumption Inventory

### Module usage (call sites)

| Module | Sites | Domain |
|--------|-------|--------|
| `barracuda::stats` | ~250+ | Shannon, Simpson, Bray-Curtis, Pearson, bootstrap, hydrology, regression, metrics |
| `barracuda::ops` | ~180+ | bio (HMM, diversity fusion, Hill, swarm NN, pairwise L2, fitness), logsumexp, RK45 |
| `barracuda::dispatch` | ~100+ | matmul, softmax, GELU, variance, L2, mean, transpose, frobenius, HMM forward |
| `barracuda::device` | ~90+ | WgpuDevice, GpuDriverProfile, Fp64Strategy, PrecisionRoutingAdvice, probe |
| `barracuda::tensor` | ~80+ | Tensor::from\_data, SessionTensor, softmax\_dim, argmax\_dim |
| `barracuda::linalg` | ~60+ | NMF, graph Laplacian, effective rank, LU, QR, ridge regression |
| `barracuda::spectral` | ~45+ | level spacing ratio, bandwidth, condition number, phase classification, BatchIprGpu |
| `barracuda::shaders::provenance` | ~25+ | evolution\_report, cross\_spring\_shaders, cross\_spring\_matrix |
| `barracuda::unified_hardware` | ~15+ | BandwidthTier |
| `barracuda::nautilus` | ~10+ | BetaObservation, SpectralNautilusBridge |
| `barracuda::nn` | ~10+ | SimpleMlp, DenseLayer |
| `barracuda::numerical` | ~5+ | gradient\_1d, trapz, numerical\_hessian |
| `barracuda::error` | ~5+ | BarracudaError, Result |

### Dispatch layer (`gpu_dispatch/`)

neuralSpring's `gpu_dispatch/` wraps 47 barraCuda dispatch operations into a unified `Dispatcher`
struct with CPU fallback, precision routing, and provenance tracking. Split into 7 domain files:
`dispatch_activations`, `dispatch_stats`, `dispatch_hmm`, `dispatch_linalg`, `dispatch_bio`,
`dispatch_popgen`, `dispatch_dynamics`.

### metalForge/forge usage

- `bridge.rs`, `pipeline.rs`: `WgpuDevice`
- `bindings.rs`: `WORKGROUP_SIZE_1D`
- `shaders.rs`: Re-exports HMM, batch\_fitness, locus\_variance WGSL constants

### playGround usage

- `barracuda::prelude` for `WgpuDevice`, `Tensor`, `TensorSession`, `AttentionDims`
- Used by Model Lab (GPT-2 inference), compute probe, bench inference

## 2. Evolution Opportunities for barraCuda

### P0 — High Impact

**blake3 `pure` feature**: barraCuda's `blake3` dependency pulls in `cc` (C compiler build tool)
for SIMD assembly. The `blake3` crate has a `pure` feature for Rust-only SIMD. Enabling it would
make barraCuda zero-C, which unblocks full ecoBin compliance for all consumers.

**barraCuda action**: Consider `blake3 = { version = "1.8", default-features = false, features = ["pure"] }`

**Variance semantics documentation**: `barracuda::dispatch::variance_dispatch` uses population
variance (÷N) while `barracuda::stats::correlation::variance` uses sample variance (÷(N-1)).
Both are correct for their contexts but the difference should be documented.

**barraCuda action**: Add doc comments noting population vs sample semantics on each function.

### P1 — Medium Impact

**`OrExit<T>` trait**: neuralSpring absorbed this from wetSpring V123. barraCuda validation binaries
could adopt the same pattern. The trait is trivial (13 LOC) and eliminates `unwrap()`/`expect()`
from binary setup code.

```rust
pub trait OrExit<T> {
    fn or_exit(self, context: &str) -> T;
}
impl<T, E: std::fmt::Display> OrExit<T> for Result<T, E> { /* ... */ }
impl<T> OrExit<T> for Option<T> { /* ... */ }
```

**`deny.toml`**: Supply-chain hygiene configuration. neuralSpring, groundSpring, healthSpring all
have it. Pattern: `wildcards = "deny"`, license allowlist, advisory/yanked/source hygiene.

**`#[expect(reason)]` over `#[allow()]`**: neuralSpring has zero `#[allow()]` in active code.
Every lint suppression uses `#[expect(clippy::X, reason = "...")]` which warns if the
suppression becomes unnecessary. Cross-cfg edge case: use `#[cfg_attr(not(test), expect(...))]`
for lints only triggered in specific compilation modes.

**`temp-env` for env testing**: Rust 2024 edition will make `std::env::set_var` unsafe.
`temp-env = "0.3"` provides `with_var()` and `with_var_unset()` for safe, scoped env mutation.

### P2 — Future / Awareness

**Typed `compute.dispatch` client**: healthSpring V30 has a typed client for direct GPU dispatch
via IPC. neuralSpring's `Dispatcher` could be exposed as a `compute.dispatch` capability provider.

**NDJSON streaming**: rhizoCrypt V13 defines `StreamItem`/`StreamingAppendResult` for pipeline
coordination. barraCuda's future streaming pipeline could adopt this for data flow.

**Content convergence**: sweetGrass ISSUE-013 (collision-preserving provenance) may affect
`barracuda::shaders::provenance` if collision detection is needed at the shader level.

## 3. neuralSpring Patterns for Absorption

### ValidationHarness

neuralSpring's `ValidationHarness` (hotSpring pattern) accumulates pass/fail checks with
tolerances, modes (absolute/relative/upper/lower bounds), and produces machine-readable
summaries. All 260 validation binaries use it. barraCuda could absorb this for its own
validation infrastructure.

### Tolerance registry

80+ named tolerance constants with justifications, organized by domain. Example:
`GPU_VARIANCE_DISPATCH_F32`, `CROSS_LANGUAGE`, `GPU_SOFTMAX_SUM_F32`.
Pattern: `pub const NAME: f64 = value; // justification`.

### Capability-based discovery

All neuralSpring IPC uses `discover_by_capability(cap, hint)` — never hardcoded socket paths.
Constants centralized in `primal_names.rs` and `niche.rs`.

## 4. Quality Metrics

| Metric | neuralSpring V110 |
|--------|-------------------|
| Lib tests | 1128 |
| playGround tests | 61 |
| Forge tests | 73 |
| Binaries | 260 |
| Modules | 47 |
| barracuda imports | 167 binaries + 80+ lib files |
| barracuda modules | 13+ |
| Clippy warnings | 0 (pedantic+nursery, -D warnings) |
| `#[allow()]` | 0 |
| Unfulfilled expectations | 0 |
| C dependencies | 0 |
| Production mocks | 0 |
| Unsafe blocks | 0 (`#![forbid(unsafe_code)]`) |

## 5. Open Questions

1. **blake3 `pure` timeline**: When could barraCuda ship with the `pure` feature?
2. **`Dispatcher` as capability**: Should neuralSpring expose its `Dispatcher` wrapper
   as a `compute.dispatch` provider, or should consumers use barraCuda directly?
3. **Provenance registry**: `barracuda::shaders::provenance` tracks 22+ cross-spring
   shaders. Should this be auto-generated from the shader corpus?
