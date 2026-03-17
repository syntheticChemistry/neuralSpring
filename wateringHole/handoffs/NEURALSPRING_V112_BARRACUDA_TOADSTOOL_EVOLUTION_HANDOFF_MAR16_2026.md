# neuralSpring V112 → barraCuda / toadStool Evolution Handoff

**Date**: March 16, 2026
**From**: neuralSpring (Session 161, V112)
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V108–V112 (Sessions 157–161)
**Supersedes**: V110 bC/tS handoff

## Executive Summary

- neuralSpring consumes **28 barraCuda modules** across **155 files** (265 binaries)
- All 17 shortcomings (S-01–S-17) resolved upstream — zero pending blockers
- **Zero C dependencies** in neuralSpring workspace (Tower Atomic)
- Only non-Rust dep: `cc` (build-time) via `blake3` in barraCuda — `pure` feature requested
- **S160–S161 new**: Structured `IpcError` (7 phases), `call_typed()`, typed `compute.dispatch`
  protocol, `extract_rpc_error()`, hardcoded paths eliminated, zero `eprintln!` in playGround
- Patterns for absorption: `IpcError`, `compute.dispatch` typed client, `ValidationHarness`,
  tolerance registry, `OrExit<T>`, `deny.toml`, `#[expect(reason)]`, `temp-env`

## 1. barraCuda Consumption Inventory

### Module usage (155 files, 28 modules)

| Module | Sites | Domain |
|--------|-------|--------|
| `barracuda::stats` | ~250+ | Shannon, Simpson, Bray-Curtis, Pearson, bootstrap, hydrology, regression |
| `barracuda::ops` | ~180+ | bio (HMM, diversity, Hill, swarm, fitness), logsumexp, RK45, FFT, variance |
| `barracuda::dispatch` | ~100+ | matmul, softmax, GELU, variance, L2, mean, transpose, frobenius |
| `barracuda::device` | ~90+ | WgpuDevice, GpuDriverProfile, Fp64Strategy, PrecisionRoutingAdvice |
| `barracuda::tensor` | ~80+ | Tensor::from\_data, SessionTensor, softmax\_dim, argmax\_dim |
| `barracuda::linalg` | ~60+ | NMF, graph Laplacian, effective rank, LU, QR, ridge, BatchedEighGpu |
| `barracuda::spectral` | ~45+ | level spacing, bandwidth, condition number, phase classification, BatchIprGpu |
| `barracuda::shaders::provenance` | ~25+ | evolution\_report, cross\_spring\_shaders, cross\_spring\_matrix |
| `barracuda::prelude` | ~20+ | TensorSession, WgpuDevice, AttentionDims, Tensor |
| `barracuda::unified_hardware` | ~15+ | BandwidthTier, ComputeExecutor |
| `barracuda::nautilus` | ~10+ | BetaObservation, DriftMonitor, NautilusBrain, SpectralNautilusBridge |
| `barracuda::nn` | ~10+ | SimpleMlp, DenseLayer, Activation |
| `barracuda::numerical` | ~5+ | gradient\_1d, trapz, numerical\_hessian |
| `barracuda::error` | ~5+ | BarracudaError |
| `barracuda::ops::fused_*` | ~20+ | chi_squared, kl_divergence, map_reduce_f64 |
| `barracuda::ops::cosine_similarity_f64` | ~5+ | CosineSimilarityF64 |
| `barracuda::ops::norm_reduce_f64` | ~5+ | NormReduceF64 |
| `barracuda::ops::sum_reduce_f64` | ~5+ | SumReduceF64 |
| `barracuda::ops::weighted_dot_f64` | ~5+ | WeightedDotF64 |
| `barracuda::ops::max_abs_diff_f64` | ~5+ | MaxAbsDiffF64 |

### Dispatch layer (`gpu_dispatch/`)

47 ops wrapped in unified `Dispatcher` with CPU fallback, precision routing, and provenance.
Split into 7 domain files: activations, stats, hmm, linalg, bio, popgen, dynamics.

### playGround IPC layer

| Client | Module | Methods |
|--------|--------|---------|
| `toadstool_client` | `compute.submit`, `gpu.dispatch`, `compute.dispatch.*` | 11 methods |
| `primal_client` | 14 `science.*` capabilities | 15 methods |
| `biomeos_client` | `nucleus.*`, `capability.*` | 8 methods |
| `coralreef_client` | `shader.compile.wgsl` | 3 methods |
| `songbird_http` | `http.request` (Tower Atomic) | 3 methods |

## 2. New in S160–S161

### Structured `IpcError` (S160)

`playGround/src/ipc_client.rs` exports typed IPC error phases:

```rust
pub enum IpcError {
    Connect(std::io::Error),
    Write(std::io::Error),
    Read(std::io::Error),
    InvalidJson(serde_json::Error),
    NoResult,
    RpcError { code: i64, message: String },
    Timeout,
}
```

`is_recoverable()` returns `true` for `Connect`/`Timeout` — enables retry logic.

**toadStool action**: Consider absorbing `IpcError` for toadStool's own IPC client. The
healthSpring V31, rhizoCrypt V13, and wetSpring V125 implementations are convergent.

### `call_typed()` (S160)

`call_typed(socket, method, params, timeout) -> Result<Value, IpcError>` — typed error path.
`call()` preserved for backward compatibility (wraps `call_typed()`).

### Typed `compute.dispatch` Protocol (S160)

`ToadStoolClient` now has:
- `dispatch_submit(operation, input)` → `DispatchHandle { dispatch_id, status }`
- `dispatch_result(dispatch_id)` → `DispatchResult { dispatch_id, status, output, elapsed_ms }`
- `dispatch_capabilities()` → `Vec<String>`

**toadStool action**: Implement `compute.dispatch.submit`, `compute.dispatch.result`,
`compute.dispatch.capabilities` JSON-RPC methods. wetSpring V124 and healthSpring V31
have equivalent client implementations.

### `extract_rpc_error()` (S160)

```rust
pub fn extract_rpc_error(response: &Value) -> Option<(i64, String)>
```

Centralizes RPC error extraction (airSpring V0.8.6 pattern).

### Hardcoded Path Elimination (S161)

All `"biomeos.sock"` / `"biomeos/biomeos.sock"` replaced with `config::BIOMEOS_SOCKET_SUBDIR`
and `config::BIOMEOS_ORCHESTRATOR_SOCKET`. Duplicate constant in ipc_client.rs delegates
to library config.

### Structured Logging Completion (S161)

28 `eprintln!` → `log::info!/warn!/debug!` across playGround binaries. Zero `eprintln!`
remaining in playGround src. All server logging is now RUST_LOG-controllable.

## 3. Evolution Opportunities for barraCuda

### P0 — High Impact

**blake3 `pure` feature**: `blake3` pulls `cc` for SIMD assembly. The `pure` feature enables
Rust-only SIMD. Would make barraCuda zero-C, unlocking full ecoBin compliance.

**barraCuda action**: `blake3 = { version = "1.8", default-features = false, features = ["pure"] }`

**Variance semantics**: `dispatch::variance_dispatch` uses population (÷N),
`stats::correlation::variance` uses sample (÷(N-1)). Both correct but undocumented.

**barraCuda action**: Add doc comments noting population vs sample semantics.

### P1 — Medium Impact

**`OrExit<T>` trait**: 13 LOC, eliminates `unwrap()`/`expect()` from binary setup.
**`deny.toml`**: Supply-chain hygiene. neuralSpring, groundSpring, healthSpring, wetSpring all have it.
**`#[expect(reason)]`**: Zero `#[allow()]` across 6 springs. Self-documenting lint suppressions.
**`temp-env`**: Safe env var testing for Rust 2024.

### P2 — Future

**NDJSON streaming**: rhizoCrypt V13 `StreamItem`/`StreamingAppendResult` for pipeline coordination.
**Content convergence**: sweetGrass collision-preserving provenance may affect `shaders::provenance`.
**Circuit breaker / retry**: `IpcError::is_recoverable()` enables rhizoCrypt-style `CircuitBreaker`.

## 4. neuralSpring Patterns for Absorption

### ValidationHarness

Accumulates pass/fail checks with tolerances, modes (absolute/relative/upper/lower), produces
machine-readable summaries. All 260 binaries use it. 1262 tests. hotSpring pattern.

### Tolerance Registry

80+ named constants with justifications, organized by domain. Python mirror (80+ constants).

### Capability-based Discovery

`discover_by_capability(cap, hint)` — never hardcoded socket paths. Constants in `primal_names.rs`.

## 5. Quality Metrics

| Metric | neuralSpring V112 |
|--------|-------------------|
| Lib tests | 1128 |
| playGround tests | 61 |
| Forge tests | 73 |
| Binaries | 260 |
| Modules | 47 |
| barracuda files | 155 |
| barracuda modules | 28 |
| Clippy warnings | 0 (pedantic+nursery) |
| `#[allow()]` | 0 |
| `eprintln!` in playGround | 0 |
| Hardcoded socket paths | 0 |
| C dependencies | 0 |
| Unsafe blocks | 0 (`#![forbid(unsafe_code)]`) |
