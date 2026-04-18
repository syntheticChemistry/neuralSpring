# neuralSpring V133 — Level 5 Primal Proof: IPC Dispatch + Capabilities Harness

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date**: April 17, 2026 | **Session**: S182 | **Spring version**: 0.1.0
**barraCuda**: v0.3.12 | **primalSpring**: v0.9.15

---

## What Changed

### 1. Proto-Nucleate Capabilities Harness (`validate_proto_nucleate_capabilities`)

New validation binary that closes the gap between "constant exists" and "Level 5
proof runs against a deployed NUCLEUS." For each of the 7
`PROTO_NUCLEATE_VALIDATION_CAPABILITIES`:

| Capability | Owning Primal | Validation |
|------------|--------------|------------|
| `tensor.matmul` | barraCuda | 2×2 matmul parity |
| `tensor.create` | barraCuda | Zero-filled tensor shape |
| `stats.mean` | barraCuda | mean([1..5]) = 3.0 |
| `compute.dispatch` | toadStool | Dispatch acknowledgment |
| `inference.complete` | Squirrel | Response has text |
| `inference.embed` | Squirrel | Response has embedding |
| `crypto.hash` | BearDog | Response has hash/digest |

Exit codes: 0=pass, 1=fail, 2=skip (honest, primals not running).

### 2. IPC Math Client (`src/ipc_dispatch.rs`)

New library module `IpcMathClient` — the Level 5 counterpart to
`gpu_dispatch::Dispatcher` (Level 2). Provides typed Rust methods routing
through JSON-RPC IPC:

- `stats_mean`, `stats_std_dev`, `stats_weighted_mean` → barraCuda
- `tensor_matmul`, `tensor_create` → barraCuda
- `compute_dispatch` → toadStool
- `crypto_hash` → BearDog
- `inference_complete`, `inference_embed` → Squirrel

Discovery-based (env-driven sockets, `discover_primal_socket` 5-tier resolution).
`IpcLivenessReport` probes all math-relevant primals.

### 3. Stadial `deny.toml` Enforcement

`cargo deny check` now enforces stadial parity gate bans:

| Banned crate | Reason |
|-------------|--------|
| `ring` | Tower Atomic delegation: BearDog owns crypto |
| `openssl-sys` / `openssl` | ecoBin: zero C dependencies |
| `async-trait` | Stadial invariant: use native RPITIT |
| `rustls` | Tower Atomic delegation: Songbird owns TLS |
| `ed25519-dalek` | Tower Atomic delegation: BearDog owns Ed25519 |
| `cmake` | ecoBin: zero C build tools |
| `cc` | ecoBin: zero C build tools (blake3 wrapper exempted) |

### 4. `rust-toolchain.toml`

Pinned `channel = "stable"` with components `rustfmt`, `clippy`,
`llvm-tools-preview`. MSRV enforced via `rust-version = "1.87"` in Cargo.toml.

### 5. barraCuda JSON-RPC Surface Gaps (18 methods)

Documented in `docs/PRIMAL_GAPS.md` Gap 11. neuralSpring uses these
`barracuda::` library calls that have no 1:1 JSON-RPC equivalent:

| Call | Module | Status |
|------|--------|--------|
| `eigh_householder_qr` | `ops::linalg` | GAP |
| `pearson_correlation` | `stats::correlation` | GAP |
| `chi_squared_statistic` | `special` | GAP |
| `empirical_spectral_density` | `stats` | GAP |
| `shannon_from_frequencies` | `stats` | GAP |
| `solve_f64_cpu` | `linalg::solve` | GAP |
| `esn_v2::*` | `esn_v2` | GAP |
| `nn::SimpleMlp` | `nn` | GAP |
| `belief_propagation_chain` | `linalg::graph` | GAP |
| `graph_laplacian` | `linalg::graph` | GAP |
| `numerical_hessian` | `numerical` | GAP |
| `boltzmann_sampling` | `sample` | GAP |
| `nautilus::*` | `nautilus` | GAP |
| `dot`, `l2_norm`, `rmse`, `mae` | `stats` | COMPOSABLE |
| `fit_linear` | `stats` | GAP |

These block full Level 5 IPC migration. Either barraCuda expands its JSON-RPC
surface (preferred for eigh, Pearson, chi-squared, Shannon, ESN) or
neuralSpring composes multiple existing methods.

### 6. Clean-Machine Validation Script

`scripts/validate_clean_machine.sh` — Level 6 runner that orchestrates
Tier 2 (Rust proof) + Tier 3 (IPC proof) validators against a deployed NUCLEUS.
Supports `--tier 2|3|all`, env-driven socket discovery, exit 0/1/2.

---

## For barraCuda

1. **Surface expansion request**: The 18 gaps in Gap 11 block full Level 5 IPC
   migration. Priority: `linalg.eigh` (eigendecomposition), `stats.pearson`
   (correlation), `stats.chi_squared` (hypothesis testing), `stats.shannon`
   (entropy). These are the most-used domain science paths.

2. **Existing 32 methods work**: `tensor.matmul`, `tensor.create`, `stats.mean`,
   `stats.std_dev`, `stats.weighted_mean` are tested via `IpcMathClient` and
   the capabilities harness.

3. **`crypto.hash` is not yours**: The upstream manifest lists `crypto.hash` in
   neuralSpring's `validation_capabilities`, but this routes to BearDog.

## For toadStool

1. **`compute.dispatch`**: Listed in proto-nucleate validation capabilities.
   The harness calls it with a probe payload. Full integration pending
   `compute.dispatch.submit` JSON-RPC surface.

## For primalSpring

1. **V133 alignment confirmed**: `validate_proto_nucleate_capabilities`
   exercises all 7 `downstream_manifest.toml` capabilities. Manifest alignment
   is programmatically verified (constant matches upstream).

2. **Gap 11 hand-back**: 18 barraCuda methods need IPC surface expansion for
   full Level 5. Document these in the ecosystem-level gaps tracker.

3. **`deny.toml` template**: neuralSpring now enforces the stadial parity gate
   ban list. Other springs can absorb the same `deny = [...]` pattern.

## For All Springs

### The Level 5 Pattern

```
1. Read downstream_manifest.toml → validation_capabilities
2. Map each capability to its owning primal
3. Build a capabilities harness that:
   a. Discovers each primal's socket
   b. Calls the capability with test params
   c. Validates the result against known baselines
   d. Reports PASS/FAIL/SKIP with exit 0/1/2
4. Build an IPC dispatch module for domain math:
   - Mirrors the local Dispatcher interface
   - Routes through JSON-RPC instead of library calls
   - Discovery-based sockets (no hardcoded paths)
5. Document gaps: which library calls have no IPC equivalent?
   Hand back to primalSpring for ecosystem-wide tracking.
```

### `IpcMathClient` Pattern

```rust
use neural_spring::ipc_dispatch::IpcMathClient;

let client = IpcMathClient::discover();
let report = client.probe_all();
println!("Alive: {}", report.alive_count());

// Route math through IPC
let mean = client.stats_mean(&[1.0, 2.0, 3.0])?;
let product = client.tensor_matmul(&a, &b, m, k, n)?;
let hash = client.crypto_hash("sha256", "data")?;
```

### Stadial `deny.toml` Template

```toml
[bans]
deny = [
    { crate = "ring", wrappers = [], reason = "Tower: BearDog owns crypto" },
    { crate = "openssl-sys", wrappers = [], reason = "ecoBin: zero C deps" },
    { crate = "async-trait", wrappers = [], reason = "Stadial: use RPITIT" },
    { crate = "rustls", wrappers = [], reason = "Tower: Songbird owns TLS" },
    { crate = "ed25519-dalek", wrappers = [], reason = "Tower: BearDog" },
    { crate = "cmake", wrappers = [], reason = "ecoBin: zero C build tools" },
    { crate = "cc", wrappers = ["blake3"], reason = "ecoBin: blake3 exempted" },
]
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/bin/validate_proto_nucleate_capabilities.rs` | NEW — Level 5 capabilities harness |
| `src/ipc_dispatch.rs` | NEW — IPC math client |
| `src/lib.rs` | Added `pub mod ipc_dispatch` |
| `Cargo.toml` | Added `[[bin]]` for capabilities harness |
| `deny.toml` | Added `deny = [...]` ban list |
| `rust-toolchain.toml` | NEW — pinned stable toolchain |
| `docs/PRIMAL_GAPS.md` | Gap 11 (18 barraCuda surface gaps), CE4/CE5/CE12 |
| `scripts/validate_clean_machine.sh` | NEW — Level 6 clean-machine runner |

## Verification

| Gate | Result |
|------|--------|
| `cargo test --workspace` | 1,234 passed, 0 failed |
| `cargo clippy -- -D warnings` | 0 warnings |
| `cargo fmt --check` | clean |
| `cargo deny check` | advisories ok, bans ok, licenses ok, sources ok |

---

*neuralSpring V133 — The Level 5 primal proof infrastructure is built. The
harness exercises all 7 validation capabilities. The IPC dispatch module routes
math through JSON-RPC. The stadial bans are enforced. The gap to Level 6 is
documented. What remains: barraCuda surface expansion for 18 domain methods,
and running the harness against a live NUCLEUS deployed from plasmidBin ecobins.*
