# neuralSpring V116 → barraCuda / toadStool Ecosystem Absorption Handoff

**Date**: March 17, 2026
**From**: neuralSpring Session 165, V116
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V115–V116, Session 165
**Supersedes**: V115 bC/tS handoff

## Executive Summary

- **`mul_add()` FMA precision sweep**: 14 sites across 10 library modules evolved to fused multiply-add
- **IPC property testing**: 8 new invariant tests (3 library + 5 playground) for resilience primitives
- **Ecosystem leverage guide**: New `specs/ECOSYSTEM_LEVERAGE_GUIDE.md` documenting all absorption sources
- Zero regressions, zero warnings, zero unsafe, 28 property tests passing

## Part 1 — mul_add() FMA Precision

### What Changed

14 `a * b + c` patterns in production library code replaced with `f64::mul_add()`:

| Module | Sites | Context |
|--------|-------|---------|
| `glucose_prediction` | 3 | Ridge regression accumulation (H^TH, H^Ty), prediction dot product |
| `swarm_robotics` | 2 | Hidden→output layer in `neural_forward` and `neural_forward_max_score` |
| `loss_landscape` | 1 | Metropolis parameter perturbation |
| `pangenome_selection` | 1 | Jaccard intersection accumulation |
| `coral_forge/confidence` | 1 | Softmax expected value computation |
| `coral_forge/pairformer` | 2 | Triangle multiplicative outgoing and incoming gating |
| `coral_forge/structure/ipa` | 1 | IPA point distance squared accumulation |
| `spectral_commutativity` | 1 | Matrix multiplication inner loop |
| `pinn` | 1 | Burgers equation residual (u·u_x + u_t) |

### Why This Matters for barraCuda

- FMA gives single-rounding (IEEE 754 `fusedMultiplyAdd`) vs double-rounding (separate `* then +`)
- Maximum precision benefit in accumulation loops (ridge regression, Gram matrix, dot products)
- barraCuda's GPU kernels already use FMA in WGSL — this aligns CPU validation references

### Recommendation for barraCuda

Sweep `barracuda/crates/barracuda/src/` for remaining `a * b + c` patterns in CPU reference code.
The `stats::`, `linalg::`, and `numerical::` modules are primary candidates. `clippy::suboptimal_flops`
catches some of these but not all patterns.

## Part 2 — IPC Proptest Invariants

### Library Property Tests (src/property_tests.rs)

3 new tests:

1. **`retry_policy_delay_never_exceeds_max`**: 4 configurations × 100 attempts each — verifies delay is always ≤ max_delay
2. **`circuit_breaker_state_machine_sweep`**: 50 random thresholds (1–5) — verifies Closed→Open after threshold failures, reset on success
3. **`circuit_breaker_rapid_cycle_no_panic`**: 1000 random success/failure calls — verifies no panics under rapid state changes

### Playground IPC Tests (playGround/src/ipc_client.rs)

5 new tests:

1. **`parse_capability_list_never_panics_on_arbitrary_json`**: 24 fuzz values (null, bool, nested, malformed)
2. **`parse_capability_list_flat_roundtrip_preserves_strings`**: Round-trip preservation
3. **`dispatch_outcome_classify_never_panics`**: 9 fuzz values
4. **`extract_rpc_error_never_panics`**: 9 fuzz values
5. **`ipc_error_is_recoverable_contract`**: Only Connect/Timeout recoverable

### Recommendation for toadStool

Adopt the proptest pattern for your IPC protocol parsing. Key invariant:
`parse_capability_list` must never panic regardless of input — defensive parsing
is critical for discovery probes against unknown primals.

## Part 3 — Already-Implemented Items

Three absorption candidates from the cross-ecosystem review were already present:

| Item | Status | Evidence |
|------|--------|----------|
| OnceLock GPU probe caching | Already in `gpu.rs::tests::SHARED_GPU` | `OnceLock<Option<StdArc<Gpu>>>` since S162 |
| FAMILY_ID-aware discovery | Already in `ipc_client.rs::discover_socket` | `{name}-{family_id}.sock` pattern since S162 |
| GemmF64 TransA/TransB | Intentionally local | `mat_mul_transpose` operates on n=4..8 matrices; dispatch overhead dominates |

## Part 4 — Leverage Documentation

New `specs/ECOSYSTEM_LEVERAGE_GUIDE.md` documents:

- **Absorption map**: What neuralSpring uses from barraCuda (45+ submodules, 80+ functions, 128+ import files)
- **Sibling spring patterns**: 13 patterns absorbed from 6 springs (hotSpring, wetSpring, groundSpring, airSpring, healthSpring, ludoSpring)
- **Composition interface**: 16 registered capabilities, IPC protocol, self-knowledge constants
- **Evolution readiness**: What stays local (by design) vs. what should migrate upstream

---

## Metrics

| Metric | V115 | V116 |
|--------|------|------|
| Property tests | 25 | 28 (+3) |
| Playground IPC tests | 18 | 23 (+5) |
| FMA sites (mul_add) | existing in 3 modules | +14 sites in 10 modules |
| Clippy warnings | 0 | 0 |
| Unsafe code | 0 | 0 |

---

*AGPL-3.0-or-later — neuralSpring V116 handoff*
