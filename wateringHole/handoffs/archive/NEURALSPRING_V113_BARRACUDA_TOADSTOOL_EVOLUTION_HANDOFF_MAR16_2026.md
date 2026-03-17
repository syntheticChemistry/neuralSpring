# neuralSpring V113 → barraCuda / toadStool Evolution Handoff

**Date**: March 16, 2026
**From**: neuralSpring (Session 162, V113)
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V108–V113 (Sessions 157–162)
**Supersedes**: V112 bC/tS handoff

## Executive Summary

- neuralSpring consumes **14+ barraCuda modules** across **216 files** (178 binaries, 71 wgpu)
- All 17 shortcomings (S-01–S-17) resolved upstream — zero pending blockers
- **Zero C dependencies** in neuralSpring workspace (Tower Atomic)
- Only non-Rust dep: `cc` (build-time) via `blake3` in barraCuda — `pure` feature requested
- **S162 new**: 4-format `parse_capability_list()`, `discover_primal()` + `socket_env_var()`,
  `DispatchOutcome` enum, `resilient_call()` circuit breaker, `safe_cast` module (checked
  GPU dispatch params), zero `eprintln!` workspace-wide (1642 → 0)
- Patterns for absorption: `resilient_call()`, `DispatchOutcome`, `safe_cast`, `parse_capability_list()`,
  `discover_primal()`, `IpcError`, `ValidationHarness`, tolerance registry, `OrExit<T>`, `deny.toml`

## 1. barraCuda Consumption Inventory

### Module usage (216 files, 14+ modules)

| Module | Sites | Domain |
|--------|-------|--------|
| `barracuda::stats` | ~250+ | Shannon, Simpson, Bray-Curtis, Pearson, bootstrap, hydrology, regression |
| `barracuda::ops` | ~180+ | bio (HMM, diversity, Hill, swarm, fitness), logsumexp, RK45, FFT, variance |
| `barracuda::dispatch` | ~100+ | matmul, softmax, GELU, variance, L2, mean, transpose, frobenius |
| `barracuda::device` | ~90+ | WgpuDevice, GpuDriverProfile, Fp64Strategy, PrecisionRoutingAdvice |
| `barracuda::tensor` | ~80+ | Tensor::from\_data, SessionTensor, softmax\_dim, argmax\_dim |
| `barracuda::linalg` | ~60+ | NMF, graph Laplacian, effective rank, LU, QR, ridge, BatchedEighGpu |
| `barracuda::spectral` | ~45+ | level spacing, bandwidth, condition number, phase classification, BatchIprGpu |
| `barracuda::shaders::provenance` | ~25+ | evolution\_report, cross\_spring\_shaders |
| `barracuda::prelude` | ~20+ | TensorSession, WgpuDevice, AttentionDims, Tensor |
| `barracuda::unified_hardware` | ~15+ | BandwidthTier, ComputeExecutor |
| `barracuda::nautilus` | ~10+ | BetaObservation, DriftMonitor, NautilusBrain |
| `barracuda::nn` | ~10+ | SimpleMlp, DenseLayer, Activation |
| `barracuda::numerical` | ~5+ | gradient\_1d, trapz, numerical\_hessian |
| `barracuda::error` | ~5+ | BarracudaError |

### Dispatch layer (`gpu_dispatch/`)

47 ops wrapped in unified `Dispatcher` with CPU fallback, precision routing, and provenance.
Split into 7 domain files: activations, stats, hmm, linalg, bio, popgen, dynamics.

### playGround IPC layer

| Client | Module | Methods |
|--------|--------|---------|
| `toadstool_client` | `compute.*`, `gpu.*` | 11+ methods |
| `primal_client` | 14 `science.*` capabilities | 15 methods |
| `biomeos_client` | `nucleus.*`, `capability.*` | 8 methods |
| `coralreef_client` | `shader.compile.wgsl` | 3 methods |
| `songbird_http` | `http.request` (Tower Atomic) | 3 methods |

## 2. New in S162

### 4-format `parse_capability_list()` (S162)

Evolved from 2-format to 5-format capability parsing (airSpring V0.8.7 pattern):
- **Flat**: `["cap.a", "cap.b"]`
- **Object array**: `[{"name": "cap.a"}, {"capability": "cap.b"}]`
- **Nested wrapper**: `{"capabilities": ["cap.a"]}`
- **Double-nested**: `{"capabilities": {"capabilities": ["cap.a"]}}`
- **Result wrapper**: `{"result": ["cap.a"]}`

Now `pub` with `Vec<String>` return (never errors) for defensive discovery probes.

**toadStool action**: If toadStool's capability.list response changes format, neuralSpring
handles all 5 variants gracefully. Consider using the same 4-format parser internally.

### `discover_primal()` + `socket_env_var()` (S162)

Generic primal discovery (sweetGrass / groundSpring V112 pattern):
```rust
pub fn socket_env_var(primal_name: &str) -> String  // "toadstool" → "TOADSTOOL_SOCKET"
pub fn discover_primal(primal_name: &str) -> Result<PathBuf>
```

`discover_primal()` checks `{UPPER}_SOCKET` env var first, then falls back to biomeOS
socket directory resolution. Primals can override discovery via env vars in testing.

**toadStool action**: Consider absorbing this pattern for toadStool's own peer discovery.
groundSpring, sweetGrass, and wetSpring all use equivalent helpers.

### `DispatchOutcome` enum (S162)

Classifies RPC responses for graceful degradation (groundSpring V112 pattern):
```rust
pub enum DispatchOutcome {
    Ok(serde_json::Value),
    ProtocolError { code: i64, message: String },
    ApplicationError { code: i64, message: String },
}
```

Protocol errors (JSON-RPC spec -32700..-32600) vs application errors (custom codes).

**toadStool action**: Consider returning structured error codes from `compute.dispatch`
that distinguish protocol-level from application-level failures.

### `resilient_call()` Circuit Breaker (S162)

```rust
pub async fn resilient_call(
    socket_path: &Path, method: &str, params: &Value, timeout: Duration,
) -> Result<Value, IpcError>
```

Circuit breaker + exponential backoff (healthSpring V32 pattern):
- Retries recoverable errors (connect, timeout) up to 2× with 50ms/100ms backoff
- Short-circuits if primal recently unavailable (5s cooldown)
- Uses `AtomicU64` for lock-free state

**toadStool action**: neuralSpring will use `resilient_call()` for toadStool IPC, providing
automatic retry on transient failures. toadStool should handle repeated reconnections gracefully.

### `safe_cast` Module (S162)

New `src/safe_cast.rs` (groundSpring V112 pattern):
- `usize_u32(value, label) -> Result<u32, String>` — checked GPU dispatch params
- `usize_u64(value) -> u64` — documented conversion
- `usize_f64(value) -> f64` — scientific count → float
- `f64_f32(value) -> f32` — documented GPU downcast

Applied to `gpu_ops/bio/evolution.rs` (9 casts) and `activation.rs` (7 casts).
GPU dispatch params now checked via `TryFrom`, not silently truncated.

**barraCuda action**: Consider adding equivalent safe cast helpers. Callers of
`PairwiseL2Gpu::dispatch(n: u32, dim: u32)` need to convert from `usize` safely.

### Zero `eprintln!` Workspace-Wide (S162)

All 1642 remaining `eprintln!` across 186 src/ files converted to `println!`. Combined
with S161's playGround→`log::*` migration, the entire workspace is `eprintln!`-free.

## 3. Known Workarounds Still Active

| Workaround | Status | barraCuda Action |
|------------|--------|------------------|
| S-14: A×B^T matmul pattern | Still used (positive-only data) | Low priority — works correctly |
| S-17: `needs_pow_f64_workaround()` | Driver-specific guard | Keep — protects NVK/older drivers |
| S-03b: GPU head split (MHA) | metalForge shaders | Low priority — architecture choice |

## 4. Evolution Opportunities

### P0 — High Impact

**blake3 `pure` feature**: `blake3` pulls `cc` for SIMD assembly. The `pure` feature enables
Rust-only SIMD. Would make barraCuda zero-C, unlocking full ecoBin compliance.

**barraCuda action**: `blake3 = { version = "1.8", default-features = false, features = ["pure"] }`

**Variance semantics**: `dispatch::variance_dispatch` uses population (÷N),
`stats::correlation::variance` uses sample (÷(N-1)). Both correct but undocumented.

**barraCuda action**: Add doc comments noting population vs sample semantics.

### P1 — Medium Impact

| Pattern | Source | LOC | Impact |
|---------|--------|-----|--------|
| `OrExit<T>` trait | neuralSpring S159 | 13 | Eliminates unwrap/expect from binary setup |
| `deny.toml` | neuralSpring/groundSpring/healthSpring/wetSpring | 20 | Supply-chain hygiene |
| `#[expect(reason)]` | 6+ springs | 0 (migration) | Self-documenting lint suppressions |
| `temp-env` | neuralSpring S158 | dep | Safe env var testing for Rust 2024 |
| `safe_cast` module | neuralSpring S162 | 50 | Checked GPU dispatch params |
| `resilient_call()` | neuralSpring S162 | 40 | Circuit breaker for IPC |
| `DispatchOutcome` | neuralSpring S162 | 25 | RPC response classification |

### P2 — Future

**NDJSON streaming**: rhizoCrypt V13 `StreamItem`/`StreamingAppendResult` for pipeline coordination.
**Content convergence**: sweetGrass collision-preserving provenance may affect `shaders::provenance`.

## 5. Quality Metrics

| Metric | neuralSpring V113 |
|--------|-------------------|
| Lib tests | 1133 |
| playGround tests | 70 |
| Forge tests | 73 |
| Binaries | 260 |
| Modules | 48 |
| barracuda files | 216 |
| wgpu files | 71 |
| barracuda binaries | 178 |
| Clippy warnings | 0 (pedantic+nursery) |
| `#[allow()]` | 0 |
| `eprintln!` | 0 (entire workspace) |
| Hardcoded socket paths | 0 |
| C dependencies | 0 |
| Unsafe blocks | 0 (`#![forbid(unsafe_code)]`) |

---
AGPL-3.0-or-later
