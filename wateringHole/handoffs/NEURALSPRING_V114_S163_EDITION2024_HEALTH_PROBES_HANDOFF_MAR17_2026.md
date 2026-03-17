# neuralSpring V114 — Session 163: Edition 2024, Health Probes, IPC Resilience

**Date**: March 17, 2026
**From**: neuralSpring Session 163, V114
**To**: barraCuda / toadStool teams, sibling springs
**License**: AGPL-3.0-or-later
**Covers**: V113–V114, Session 163
**Supersedes**: V113 S162 absorption handoff

## Summary

Session 163 upgraded neuralSpring to Rust Edition 2024, added health probes (`health.liveness`, `health.readiness`), introduced generic IPC resilience primitives (`RetryPolicy`, `CircuitBreaker`), added 6 proptest invariants, enriched `DispatchOutcome`, and hardened `deny.toml`.

## Session 163 Changes in Detail

### Edition 2024 Upgrade

- **8 files** with `gen` keyword renames (Rust 2024 reserved keyword)
- **~15** let chain refactors
- Closure pattern fixes for Edition 2024 compatibility

### Health Probes

| File | Change |
|------|--------|
| `src/config.rs` | `ALL_CAPABILITIES` includes `health.liveness`, `health.readiness` |
| `src/niche.rs` | Capabilities + dependencies + cost estimates for health domain |
| `src/bin/neuralspring_primal/main.rs` | Dispatch routing for health methods |
| `src/bin/neuralspring_primal/handlers.rs` | `handle_liveness()`, `handle_readiness()` implementation |
| `playGround/src/mcp_tools.rs` | Tool definitions for health domain, 14→16 MCP tools |

### ipc_resilience.rs

- **RetryPolicy**: Configurable exponential backoff (`initial_delay`, `max_delay`, `multiplier`, `max_retries`)
- **CircuitBreaker**: Closed/Open/HalfOpen with threshold, cooldown, epoch-based timing
- **8 unit tests** for both primitives

### Proptest

6 property tests in `src/primitives.rs`:

1. `softmax_sums_to_one` — softmax outputs sum to 1.0
2. `softmax_nonnegative` — softmax outputs non-negative
3. `shannon_entropy_nonnegative` — entropy non-negative for valid distributions
4. `relu_idempotent` — relu(relu(x)) == relu(x)
5. `relu_identity_for_positive` — relu preserves non-negative inputs
6. `rk4_energy_bounded` — harmonic oscillator energy conservation

### DispatchOutcome (playGround/src/ipc_client.rs)

3 new methods:

- `classify_response(response)` — structured RPC error classification
- `is_method_not_found()` — graceful capability degradation
- `from_ipc_error(err)` — classify typed `IpcError` into `DispatchOutcome`

### deny.toml

- `unknown-git = "deny"`
- Advisory `db-path` and `db-urls` for advisory database
- `allow-git = []` (explicit empty)

### Tolerance Provenance

6 constants with validation citations in `src/tolerances/mod.rs` (EXACT_F64, CROSS_LANGUAGE, ZERO_DETECTION, NORM_PPF_TAIL, BENCHMARK_GLOBAL_MIN, BENCHMARK_CROSS_PYTHON).

## Files Modified (Key Files)

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Edition 2024 |
| `src/config.rs` | ALL_CAPABILITIES + health probes |
| `src/niche.rs` | Health capabilities, dependencies, cost estimates |
| `src/ipc_resilience.rs` | **NEW** — RetryPolicy, CircuitBreaker |
| `src/primitives.rs` | 6 proptest invariants |
| `src/tolerances/mod.rs` | Provenance doc comments |
| `src/bin/neuralspring_primal/main.rs` | Health dispatch |
| `src/bin/neuralspring_primal/handlers.rs` | handle_liveness, handle_readiness |
| `playGround/src/ipc_client.rs` | DispatchOutcome enrichment |
| `playGround/src/mcp_tools.rs` | Health tools, 16 total |
| `deny.toml` | unknown-git, advisory DB, allow-git |
| ~8 files | `gen` keyword renames, let chains |

## Test Results

| Metric | Value |
|--------|-------|
| Tests passed | 1295 |
| Lib tests | 1152 |
| playGround tests | 70 |
| Forge tests | 73 |
| Clippy | Clean |
| fmt | Clean |
| Regressions | 0 |
| Warnings | 0 |
| unsafe blocks | 0 |

---
AGPL-3.0-or-later
