# neuralSpring V114 → barraCuda / toadStool Evolution Handoff

**Date**: March 17, 2026
**From**: neuralSpring Session 163, V114
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V113–V114, Session 163
**Supersedes**: V113 bC/tS handoff

## Executive Summary

- **Rust Edition 2024** across all 3 workspace crates
- **Health probes**: `health.liveness` + `health.readiness` IPC methods
- **IPC resilience**: generic `RetryPolicy` + `CircuitBreaker` (ready for absorption)
- **Property-based testing**: 6 proptest invariants for core primitives
- **MCP tools**: 14→16 (health domain)
- **deny.toml hardened**: unknown-git=deny, advisory DB, allow-git=[]
- **Test counts**: 1295 (1152 lib + 70 playGround + 73 forge)
- Zero regressions, zero warnings, zero unsafe

## 1. Patterns for Absorption

What barraCuda/toadStool should consider absorbing:

### RetryPolicy

Generic exponential backoff (configurable `initial_delay`, `max_delay`, `multiplier`, `max_retries`). Currently in `src/ipc_resilience.rs`. Would benefit all primals doing IPC.

### CircuitBreaker

Closed/Open/HalfOpen pattern with threshold, cooldown, epoch-based timing. Protects against cascading failures.

### Health Probes

`health.liveness` and `health.readiness` — standard IPC health check pattern. Every primal should expose these.

### DispatchOutcome Enrichment

- `classify_response()` for structured RPC error classification
- `is_method_not_found()` for graceful capability degradation

### Proptest Invariants

Mathematical invariants (softmax sums to 1, entropy non-negative, relu idempotent) as a pattern for upstream validation.

### Edition 2024 Patterns

Let chains, reserved keyword handling, pattern ergonomics.

## 2. Still-Relevant Items from V113

- blake3 `pure` feature request (zero C deps)
- Variance semantics documentation (population vs sample)
- `enable f64;` PTXAS regression workaround
- safe_cast, resilient_call, parse_capability_list, discover_primal
- xoshiro128ss.wgsl absorption into barracuda::ops::prng

## 3. Known Workarounds

| Workaround | Status |
|------------|--------|
| A×Bᵀ matmul pattern | Defense-in-depth for S-15 regression |
| `needs_pow_f64_workaround()` flag | Driver-specific guard |
| MHA head split still local | S-03b upstream wrapper |
| `enable f64;` PTXAS Ada Lovelace workaround | pipeline_cache.rs |

## 4. What neuralSpring Needs from barraCuda/toadStool

- nn::Layer / nn::Optimizer for training loops (Tier B blocker)
- Autograd reverse-mode AD
- Flash attention / fused LayerNorm+GELU kernels
- `xoshiro128ss` GPU PRNG upstream
- Size-based f32/f64 routing (PairwiseHammingGpu regression on small sizes)

---
AGPL-3.0-or-later
