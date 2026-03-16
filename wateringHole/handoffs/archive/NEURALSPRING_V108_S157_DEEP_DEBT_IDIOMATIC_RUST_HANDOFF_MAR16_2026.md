# neuralSpring V108 S157 — Deep Debt + Idiomatic Rust Evolution Handoff

**Date**: March 16, 2026
**From**: neuralSpring S157 (V108)
**To**: barraCuda, toadStool, coralReef
**Supersedes**: V107 S156 (Audit + IPC Discovery Fixes)
**License**: AGPL-3.0-or-later

## Pins

| Component | Version | Commit |
|-----------|---------|--------|
| barraCuda | v0.3.5 | `0649cd0` |
| toadStool | S146 | `751b3849` |
| coralReef | Iter 49 | (latest) |
| neuralSpring | S157 | V108 |

## Executive Summary

- **5 blanket lint suppressions eliminated** — all `#![expect(clippy::pedantic, ...)]`
  evolved to targeted `#[expect()]` with documented reasons on specific items
- **Primal binary refactored** — `neuralspring_primal/main.rs` evolved from blanket
  suppression to zero-warning pedantic+nursery with smart function extraction
- **`set_var` eliminated** — unsafe env mutation replaced with value passing
- **expect()/unwrap()/panic!** in production binaries replaced with proper error
  handling (`Result<()>`, `process::exit`, `let...else`)
- **Largest file refactored** — `validate_modern_cross_spring.rs` 949 → 865 LOC via
  macro-based bench extraction and provenance summary condensation
- **bytemuck aligned** — metalForge/forge upgraded 1.14 → 1.21 (workspace-consistent)
- **Hardcoded paths eliminated** — `dump_neuralspring_scenarios` uses
  `NEURALSPRING_SCENARIO_DIR` env var with fallback
- **ring C dep ELIMINATED** — reqwest removed from playGround via Tower Atomic
  pattern (Songbird IPC for all HTTP), zero C deps in entire workspace
- **Tower Atomic HTTP client** — new `songbird_http` module routes all external HTTP
  through Songbird (BearDog + Songbird = Pure Rust HTTPS), capability-based discovery
- **Kokkos benchmark provenance** — placeholder status explicitly documented in code
  and runtime output
- **1128 library tests pass**, zero clippy (pedantic+nursery), zero fmt diffs

## Changes

### 1. Binary Lint Suppression Evolution (5 files)

Evolved blanket `#![expect(clippy::pedantic, clippy::nursery, ...)]` to targeted suppressions:

| Binary | Before | After |
|--------|--------|-------|
| `neuralspring_primal/main.rs` | pedantic, nursery, expect_used, too_many_lines, cast_possible_truncation, similar_names | `similar_names` only |
| `validate_alphafold2_evoformer.rs` | pedantic, cast_possible_truncation, similar_names, too_many_lines | `cast_possible_truncation`, `similar_names` with reasons |
| `validate_multi_head_esn.rs` | pedantic | (none — clean under pedantic) |
| `validate_gpu_ode_batch.rs` | pedantic, expect_used | `cast_precision_loss` only |
| `validate_training_monitor.rs` | pedantic, nursery | `cast_precision_loss` only |

All 5 files now compile with `-W clippy::pedantic -W clippy::nursery -D warnings`.

### 2. Primal Binary Smart Refactoring

`neuralspring_primal/main.rs`:
- Extracted `push_petaltongue_scenario()` — petalTongue integration in own function
- Extracted `spawn_lifecycle_tasks()` — heartbeat + signal handlers
- Extracted `accept_loop()` — connection accept loop with concurrency semaphore
- Eliminated `std::env::set_var` (deprecated in Rust 2024) — pass CLI family ID via value
- SIGTERM handler: `expect()` → `let Ok(...) else { return; }`

Primal sub-modules also cleaned:
- `discovery.rs`: `continue` in match → `if let Ok`, `match resp` → `map_or_else`
- `folding.rs`: `match` → `let...else`, `p_usize` cast annotated
- `spectral.rs`: `branches_sharing_code` fixed, `params_usize` cast annotated

### 3. Error Handling Evolution

| File | Before | After |
|------|--------|-------|
| `dump_neuralspring_scenarios.rs` | `expect()`, `panic!()` | `process::exit(1)` with eprintln |
| `validate_gpu_ode_batch.rs` | `.expect("ODE trace must be non-empty")` | `let Some(...) else { h.check_bool(.., false); h.finish() }` |
| `neuralspring_primal/main.rs` | `.expect("SIGTERM handler...")` | `let Ok(mut sig) = ... else { return; }` |

### 4. Large File Refactoring

`validate_modern_cross_spring.rs` (949 → 865 LOC):
- Extracted `bench_pair()` helper for GPU/CPU timing pairs
- Extracted `print_bench_row()` for formatted output
- Created `bench_row!` macro eliminating 6 repetitive BenchResult constructions
- Condensed `report_provenance_summary()` from 44 lines → 9 lines (detail in docs)

`validate_gpu_pure_workload_all.rs` (935 → 942 LOC, post-fmt):
- Extracted `check_gpu_f32_mean()` helper for common readback→mean→check pattern
- Applied across Hamming, L2, Jaccard pairwise ops (3 domains consolidated)
- Eliminated duplicate readback match blocks

### 5. Tower Atomic Evolution — Zero C Dependencies

**reqwest + ring completely removed** from the workspace dependency tree.

Evolved playGround HTTP from direct `reqwest` (which pulled `ring`, a C assembly crate)
to the Tower Atomic pattern:

| Before | After |
|--------|-------|
| `reqwest` + `rustls-tls` → `ring` (C assembly) | `songbird_http` → IPC → Songbird primal |
| Compile-time HTTP coupling | Runtime capability discovery |
| playGround-specific TLS stack | Ecosystem-wide shared TLS (BearDog + Songbird) |

New modules:
- `playGround/src/songbird_http.rs` — Tower Atomic HTTP client via Songbird IPC
  - Discovers `http.request` capability at runtime
  - `get()`, `get_json()`, `download_to_file()` methods
  - Zero compile-time HTTP deps, zero C deps
- `playGround/src/hf_hub.rs` — rewritten to use `SongbirdHttp` instead of `reqwest`
  - Same public API (`model_info`, `download_file`, `download_model`)
  - HTTP routed through Songbird at runtime

**Dependency verification**: `cargo tree -p neuralspring-playground | grep -E "ring|reqwest"` = empty

### 6. Dependency & Config Fixes

- `metalForge/forge/Cargo.toml`: bytemuck `1.14` → `1.21` (matches root + playGround)
- `playGround/Cargo.toml`: reqwest **removed** entirely (was only HTTP dep)
- `dump_neuralspring_scenarios.rs`: `NEURALSPRING_SCENARIO_DIR` env var with fallback

### 7. Benchmark Provenance

- `bench_kokkos_parity.rs`: Explicit `## Provenance` section documenting placeholder status
- Runtime output: `⚠ PROVENANCE: Kokkos baselines are PLACEHOLDER` warning

## Quality Gate

| Check | Result |
|-------|--------|
| `cargo test --lib` | 1128 pass, 0 fail |
| `cargo test -p neuralspring-playground` | 2 pass, 11 ignored (require daemons) |
| `cargo clippy --lib (pedantic+nursery)` | 0 warnings |
| `cargo clippy` (5 evolved bins, pedantic+nursery, -D warnings) | 0 warnings |
| `cargo clippy` (neuralspring_primal, pedantic+nursery, -D warnings) | 0 warnings |
| `cargo clippy -p neuralspring-playground (pedantic+nursery, -D warnings)` | 0 warnings |
| `cargo fmt --check` | 0 diffs |
| `#![forbid(unsafe_code)]` | enforced (lib + playGround) |
| Max file LOC | 942 (validate_gpu_pure_workload_all.rs) |
| Blanket `clippy::pedantic` in binaries | 0 remaining |
| `expect()`/`unwrap()`/`panic!()` in production | evolved where touched |
| C dependencies (ring, openssl, native-tls) | **0** — fully eliminated |
| reqwest in workspace | **removed** — Tower Atomic replaces |

## Remaining Work (Next Session)

1. **`validate_barracuda_tensor.rs`** (918 LOC) — well-structured but could benefit from
   `validate_tensor_unary`/`validate_tensor_binary` extraction patterns
2. **Kokkos real baselines**: Run Kokkos-CUDA and barraCuda WGSL on same GPU for
   matched hardware comparison
3. **PolyBench/Rodinia**: Add GPU benchmark suites for broader industry parity
4. **Centralize remaining magic numbers**: Deep sweep of all test modules
5. **data.hf_fetch**: Register HuggingFace capability in NestGate for ecosystem-wide
   model access via `data.*` IPC methods
