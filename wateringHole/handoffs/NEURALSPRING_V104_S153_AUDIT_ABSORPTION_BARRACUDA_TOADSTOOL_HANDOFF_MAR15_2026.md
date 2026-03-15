# neuralSpring V104: barraCuda + toadStool Absorption Handoff

**Date:** 2026-03-15
**From:** neuralSpring V104 / Session 153
**To:** barraCuda team, toadStool team, coralReef team
**License:** AGPL-3.0-or-later
**Covers:** Comprehensive ecosystem audit + deep debt execution
**Supersedes:** V103 S152 Deep Debt Execution Handoff (Mar 15, 2026)
**Pins:** barraCuda v0.3.5 at `0649cd0`, toadStool S146+, coralReef Phase 10

---

## Executive Summary

Session 153 ran a full 11-dimension ecosystem audit against wateringHole standards,
then executed all findings. The codebase is now at zero warnings, zero unsafe, zero
fmt diffs, with all capabilities unified, all tolerances centralized, and all
validation binaries on the hotSpring harness pattern.

### Verification

| Check | Result |
|-------|--------|
| `cargo fmt --all --check` | PASS (0 diffs) |
| `cargo clippy --workspace -- -W pedantic -W nursery` | PASS (0 warnings) |
| `cargo doc --workspace --no-deps` | PASS (0 warnings) |
| `cargo test --workspace` | 1290 passed, 0 failed |
| `unsafe` blocks | 0 (all 3 crates `forbid(unsafe_code)`) |
| Files > 1000 LOC | 0 (max: 949) |
| C dependencies | 0 (reqwest uses `rustls-tls`) |

---

## Part 1: Absorption Candidates for barraCuda

### Tier 1: Direct Extraction (low effort, high value)

| Item | Location | Rationale |
|------|----------|-----------|
| `ValidationHarness` standardization | `src/validation/mod.rs` | 6 springs now implement the same `check_abs`/`check_rel`/`check_bool` + `finish()` → `exit(0\|1)` pattern. Extract to `barracuda::validation` as shared infrastructure. |
| Tolerance hierarchy | `src/tolerances/` (780 LOC) | 100+ named constants with scientific justification. Cross-spring standard candidate — wetSpring has 180+, groundSpring 120+. Dedup and centralize. |
| `primitives::sigmoid`/`gelu`/`softmax`/`relu` | `src/primitives.rs` | CPU reference implementations used by 4+ springs. These are already in `barracuda::nn` but springs keep local copies for cross-validation. Document the "CPU reference" pattern so springs know to use `barracuda::nn` and only keep local for parity testing. |

### Tier 2: Module Promotion (medium effort)

| Item | Location | Rationale |
|------|----------|-----------|
| `TensorSession` patterns doc | playGround `inference/transformer.rs` | 7-45x speedup demonstrated via hot dispatch. Main library still uses per-op cold dispatch. Document fused-pipeline patterns for HMM chains, ODE loops, attention pipelines. |
| `expected_source()` provenance linkage | `src/provenance/mod.rs` | Maps experiment labels to reference constant locations. Pattern could be standardized across springs. |
| Bench harness: `bench_median()` + `bench_once()` | `src/bin/bench_*.rs` | Duplicated in 18 bench binaries. Extract to `barracuda::bench` or shared crate. |

### Tier 3: Evolution Targets (design discussion needed)

| Item | Discussion |
|------|------------|
| `barracuda::ops::bio::MhaForwardF64` | HEAD_SPLIT + HEAD_CONCAT local shaders exist because upstream MHA hangs on some workloads. When barraCuda MHA stabilizes, neuralSpring can absorb. 2 local shaders retire. |
| Cross-spring tolerance registry | Each spring has its own `tolerances/` module. A shared `barracuda::tolerances::registry` with named constants + categories would eliminate 500+ LOC of duplication across 6 springs. |
| `graph_laplacian` typed return | `barracuda::linalg::graph_laplacian` returns flat `Vec<f64>`. Springs reconstruct 2D layout manually. Return a matrix type or (eigenvalues, eigenvectors) tuple. |

---

## Part 2: Absorption Candidates for toadStool

### metalForge → toadStool

| Module | Purpose | Absorption Path |
|--------|---------|-----------------|
| `forge/probe.rs` | GPU/CPU discovery (wgpu + `/proc`) | Merge into toadStool device discovery |
| `forge/inventory.rs` | Unified substrate view | Align with toadStool device inventory |
| `forge/dispatch.rs` | Capability-based workload routing | Map to toadStool `compute_dispatch` |
| `forge/mixed.rs` | Transfer cost model for mixed substrates | Feed toadStool scheduling heuristics |
| `forge/pcie_bridge.rs` | PCIe P2P detection | Merge into toadStool hardware topology |
| `forge/graph.rs` | NUCLEUS pipeline DAG executor | Candidate for biomeOS pipeline framework |

### WGSL Shader Absorption Status

| Category | Count | Status |
|----------|-------|--------|
| Absorbed upstream | 24 | Tier A — local copies for validation only |
| Adapt existing | 8 | Tier B — binding/param changes needed |
| New (AlphaFold) | 9 | Tier C — Evoformer/structure, no upstream equiv |
| Truly local | 4 | HEAD_SPLIT, HEAD_CONCAT, XOSHIRO128SS, SWARM_NN_SCORES |

---

## Part 3: Cross-Spring Patterns Learned

1. **Tolerance centralization saves debugging time** — neuralSpring had 9 binaries
   with inline `1e-6` / `1e-10` that the audit caught. Named constants with
   scientific justification prevent tolerance creep. Recommend ecosystem-wide
   standard in `barracuda::tolerances`.

2. **`ALL_CAPABILITIES` single source of truth** — capability lists were duplicated
   between primal binary and MCP tools. Unified into `config.rs` with `pub use`
   re-exports. Other springs may have the same pattern.

3. **playGround lints should match workspace** — playGround had 9 clippy warnings
   because it lacked `[lints]` in its `Cargo.toml`. All workspace members should
   inherit the same lint configuration.

4. **`forbid(unsafe_code)` in Cargo.toml vs lib.rs** — forge had it only in
   `Cargo.toml` but not in `lib.rs`. Both should be set for defense in depth.

5. **Kokkos benchmark gap** — `bench_kokkos_parity.rs` exists but uses placeholder
   data. The referenced `BARRACUDA_KOKKOS_GPU_BENCHMARK_RESULTS_MAR04_2026.md` is
   missing from wateringHole. Either generate real Kokkos comparison data or
   retract the placeholder.

---

## Part 4: Benchmark Parity Status

| Benchmark | Status | Notes |
|-----------|--------|-------|
| Python → barraCuda CPU | COMPLETE | 15 domains, 38.6x geomean speedup |
| cuBLAS SGEMM/DGEMM | COMPLETE | via `bench_industry_gpu_parity` |
| cuDNN ops | COMPLETE | softmax, layernorm, GELU, conv2d, sigmoid |
| cuFFT | COMPLETE | FFT/RFFT |
| FlashAttention | MEASURED | ~30x slower (expected: decomposed vs fused) |
| Kokkos | PLACEHOLDER | Harness exists, no real comparison data |
| Polybench | NOT STARTED | Future barraCuda work |

---

## Part 5: Primitive Consumption (v0.3.5)

neuralSpring consumes 46 upstream barraCuda delegations across:

- `barracuda::dispatch::*` (matmul, softmax, gelu, mean, variance, l2_distance, hmm_forward)
- `barracuda::ops::bio::*` (HmmBatchForwardF64, PairwiseL2Gpu, MultiObjFitnessGpu, etc.)
- `barracuda::ops::BatchedEighGpu`, `barracuda::ops::BatchIprGpu`
- `barracuda::stats::correlation::*` (pearson, variance)
- `barracuda::special::*` (chi_squared_statistic)
- `barracuda::tensor::Tensor` (matmul, log_softmax, layer_norm, sigmoid)

---

## Part 6: What's Left

| Item | Priority | Owner |
|------|----------|-------|
| TensorSession adoption in main library | P1 | neuralSpring |
| MHA stabilization (retire HEAD_SPLIT/CONCAT) | P2 | barraCuda |
| Kokkos benchmark real data | P2 | barraCuda/toadStool |
| Cross-spring tolerance registry | P2 | wateringHole RFC |
| `graph_laplacian` typed return | P3 | barraCuda |
| `bench_median()`/`bench_once()` extraction | P3 | barracuda::bench |
