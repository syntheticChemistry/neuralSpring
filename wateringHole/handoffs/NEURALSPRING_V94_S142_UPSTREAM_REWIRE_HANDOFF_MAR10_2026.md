<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V94 → toadStool/barraCuda/coralReef Upstream Rewire Handoff

**Date**: March 10, 2026
**From**: neuralSpring S142 (1048 lib + 71 forge tests, 233 binaries, 0 clippy pedantic+nursery)
**To**: barraCuda team, toadStool team, coralReef team
**Supersedes**: V92 S137 upstream rewire (Mar 10, 2026)
**Synced against**: barraCuda `83aa08a`, toadStool S142 (`a86bc546`), coralReef Iteration 29 (`2779c88`)
**License**: AGPL-3.0-or-later

---

## Executive Summary

neuralSpring S142 is an **upstream rewire and revalidation release** that syncs
to the current HEAD of barraCuda (12 commits ahead), toadStool (20 commits ahead),
and coralReef (19 commits ahead) from the previous S137 pins.

**What changed:**

1. **Sprint 2 API absorption** — barraCuda Sprint 2 has landed at HEAD (`8ecc75a`):
   - `barracuda::activations` (sigmoid, gelu, relu, relu_batch) — `primitives.rs`
     now delegates f64 scalar/batch activations to upstream, eliminating duplicate math
   - `barracuda::device::test_harness::fused_ops_healthy` — replaces 2 local
     variance canary probes in `gpu_ops/tests_ops.rs` and `gpu_dispatch/tests_gpu.rs`
   - `barracuda::rng::lcg_step` — assessed, no direct replacement (neuralSpring
     uses SplitMix64, not LCG)
   - `barracuda::special::tridiagonal_ql` — assessed, optional Anderson
     localization optimization (no immediate action)

2. **SpringDomain API fix** — enum→struct migration (`NeuralSpring`→`NEURAL_SPRING`,
   `HotSpring`→`HOT_SPRING`, `WetSpring`→`WET_SPRING`) in `tests_cpu_provenance.rs`

3. **Precision API fix** — `compile_shader_universal` decomposed to precision-routed
   dispatch (`compile_shader`/`compile_shader_f64`/`compile_shader_df64`) in `gpu.rs`.
   `Precision::F16` removed (upstream dropped F16 tier).

4. **coralReef bridge alignment** — `discover_socket()` updated from stale
   `coralreef.json` namespace scan to ecosystem-aligned discovery:
   primary `$XDG_RUNTIME_DIR/biomeos/coralreef.sock`, fallback capability
   manifest scan `$XDG_RUNTIME_DIR/ecoPrimals/*.json`

5. **Provenance standardization** — 54 validation binaries received standard
   `## Provenance` blocks (Groups A–D: BarraCUDA parity, cross-spring/ToadStool,
   GPU pipeline, integration/system)

6. **Pin updates** — all 14 documentation/spec files and 2 validation binaries
   updated to current HEAD hashes

---

## Upstream Handoffs Reviewed

### barraCuda (`a898dee` → `83aa08a`)

| Commit | Key Change | Impact on neuralSpring |
|--------|-----------|----------------------|
| `8ecc75a` | Sprint 2: activations, eigensolver, LCG PRNG, Wright-Fisher | **ABSORBED** — activations delegated, `fused_ops_healthy` adopted |
| `5c8ebc0` | healthSpring domain in provenance types | Awareness only (docs reference healthSpring patterns) |
| `a34f28c` | Batched f32 logsumexp shader | No immediate use |
| `5c16458` | CoralReefDevice full GpuBackend impl | No neuralSpring impact (sovereign-dispatch feature) |
| `dfcc5e1` | 3-tier precision, F16/templates removed | **ABSORBED** — F16 removed, compile_shader_universal replaced |
| `83aa08a` | Orphaned code cleanup | No impact |

### toadStool (`bfe7977b` → `a86bc546`)

| Session | Key Change | Impact on neuralSpring |
|---------|-----------|----------------------|
| S142 | Hardware testing, PCIe transport, ResourceOrchestrator | No neuralSpring impact |
| S141 | Deep debt, clippy pedantic, zero-copy Bytes | No neuralSpring impact |
| S140 | Spring absorption, barraCuda Sprint 2 awareness | Aligned |
| S139 | Pipeline DAG absorption (from neuralSpring S134) | Already documented |

### coralReef (`d29a734` → `2779c88`)

| Iteration | Key Change | Impact on neuralSpring |
|-----------|-----------|----------------------|
| 29 | NVIDIA last mile pipeline | Future benefit |
| 28 | Unsafe elimination, spring absorption wave 3 | No impact |
| 25-27 | Math evolution, DEBT zero, deep debt | No impact |
| 20-24 | SSA repair, multi-GPU sovereignty | No impact |

---

## Quality Gates

```
cargo fmt --check                             → PASS
cargo clippy --all-targets -- -W pedantic -W nursery → 0 warnings, 0 errors
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps → PASS (240 files)
cargo test --lib                              → 1047/1048 (1 pre-existing GPU flake)
cargo test -p neural-spring-forge --lib       → 71/71 PASS
cargo llvm-cov --lib                          → 91.98% region / 90.75% line
```

The single flaky test (`gpu_pearson_correlation`) passes in isolation; fails under
full parallel load due to GPU driver contention. Pre-existing — not introduced by
this rewire.

---

## Remaining Absorption Opportunities

| Item | Upstream Status | neuralSpring Action |
|------|----------------|---------------------|
| `barracuda::rng::lcg_step` | At HEAD | No — neuralSpring uses SplitMix64, not LCG |
| `barracuda::special::tridiagonal_ql` | At HEAD | Optional O(n) Anderson fast path (currently O(n³) dense) |
| `barracuda::ops::WrightFisherF32` | At HEAD | No — neuralSpring uses f64 `WrightFisherGpu` |
| `barracuda::activations::softmax` | Not at HEAD (batch) | Keep local `softmax` in primitives (no upstream batch) |
| coralReef runtime IPC client | Iteration 29 ready | P2 — implement `shader.compile.wgsl` JSON-RPC calls |

---

## Cross-Spring Alignment

neuralSpring is now aligned with:
- barraCuda `83aa08a` (Sprint 2 absorbed, precision API fixed)
- toadStool S142 `a86bc546` (20 sessions ahead, no breaking changes)
- coralReef Iteration 29 `2779c88` (discovery path aligned)
- All Mar 8–10 wateringHole handoffs reviewed

*This handoff is unidirectional: neuralSpring → ecosystem. No response expected.*
