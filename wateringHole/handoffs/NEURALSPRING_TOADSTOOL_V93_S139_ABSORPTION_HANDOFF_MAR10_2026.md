<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/BarraCUDA V93 Handoff

| Field | Value |
|-------|-------|
| **Date** | 2026-03-10 |
| **From** | neuralSpring S139 |
| **To** | ToadStool, BarraCUDA, coralReef |
| **Supersedes** | V92 (S134) |
| **Pins** | ToadStool S139+, BarraCUDA v0.3.3 (`a898dee`), coralReef Iteration 10 (`d29a734`) |

## Executive Summary

neuralSpring S139 completes a full codebase audit (deep debt, linting,
documentation, evolution readiness) on top of the S134–S138 buildout
(visualization, streaming I/O, search pipeline, Kokkos parity, industry
coverage). The Spring is clean and stable: 1048 lib + 71 forge + 9
integration tests, 233 binaries, 220/220 validate\_all, 92% line coverage,
0 clippy warnings (pedantic+nursery, `--all-features`), 0 doc warnings.

This handoff documents what ToadStool/BarraCUDA/coralReef should know for
continued absorption and evolution.

---

## Part 1: Metrics Snapshot

| Metric | Value |
|--------|-------|
| Library tests | 1048 |
| Forge tests | 71 |
| Integration tests | 9 |
| Validation binaries | 233 |
| validate\_all | 220/220 PASS |
| Line coverage (llvm-cov) | 92% |
| Clippy warnings | 0 (pedantic + nursery, `--all-features`) |
| Doc warnings | 0 |
| Named tolerances | 80+ (centralized `tolerances/` with justifications) |
| Upstream rewires | 46 |
| WGSL shaders (metalForge) | 42 |
| CPU→GPU dispatch ops | 47 (~97%) |
| `#[allow(` in production | 0 (all `#[expect(` with reasons) |
| Files > 1000 LOC | 0 |
| Unsafe code | 0 (`#![forbid(unsafe_code)]`) |
| TODO/FIXME/MOCK/STUB | 0 |
| SPDX headers | 100% |
| Python baselines | 331/331 PASS |

## Part 2: What Changed Since V92 (S134)

### S135 — petalTongue Visualization Evolution

- 7 new domain scenario builders (HMM, game theory, WDM, glucose,
  immunological, population, loss landscape).
- All 8 `DataChannel` types exercised.
- `TrainingVisualizer` live streaming to petalTongue.
- `full_study()` 12-track combiner.
- `neuralspring_live_dashboard` binary.
- `scripts/visualize.sh` for offline/live/render/ecosystem modes.
- 56/56 petalTongue validation checks.

### S136 — Deep Audit + Evolution

- `PetalTonguePushClient::headless()` — socket hardcoding eliminated.
- `Gpu::read_buffer_u32` — upstream parity.
- `validate_gpu_pure_workload_all` refactored (976→940 LOC).
- Industry GPU parity gap documented.
- Kokkos/Polybench/cuBLAS gap formally requested.

### S137 — Upstream Rewire + Deep Debt

- Hardcoded `256` → `WORKGROUP_SIZE_1D` (15 sites).
- 7 WGSL shaders updated to "absorbed upstream" status.
- `gpu_or_exit()` async helper eliminates 5-line GPU init boilerplate.
- Duplicate `max_abs_diff` eliminated.
- Full audit: zero unsafe, zero TODOs, zero mocks, zero hardcoded paths.

### S138 — Industry Gap Closure

- Streaming FASTA parser (`streaming/fasta.rs`, 16 tests).
- CPU-reference BLAST pipeline (`search/kmer_index`, `search/seed_extend`, 19 tests).
- `bench_kokkos_parity` GPU benchmark harness (9 ops × production scale).
- `INDUSTRY_TOOL_GAP_ANALYSIS.md`, `BLAST_LIKE_SEARCH_SCOPE.md`, `MSA_PIPELINE_SCOPE.md`.

### S139 — Visualization Evolution + Deep Debt + Full Audit

- 4 new petalTongue scenario builders (search results, streaming I/O quality,
  Kokkos parity, industry coverage). 16 total tracks.
- `neuralspring_ecosystem_dashboard` binary.
- `config.rs` centralizes primal identity, env var names, petalTongue
  domain/theme (zero scattered magic strings).
- Named constants: `LINE_BUF_CAPACITY`, `VCF_LINE_BUF_CAPACITY`,
  `StreamSession::BACKPRESSURE_THRESHOLD`.
- Streaming FASTQ + VCF parsers (zero-copy `BufRead`, 35 tests).
- Full clippy audit (pedantic+nursery): 75 warnings resolved — `unwrap_used`
  in test modules gated with `#[expect(`, `float_cmp` replaced with epsilon
  checks, `needless_collect` refactored, `too_many_lines` addressed with
  targeted extractions.
- `cargo doc --no-deps -D warnings`: 0 warnings.

## Part 3: BarraCUDA Primitive Consumption Inventory

neuralSpring consumes 45+ BarraCUDA submodules across 128+ files:

| BarraCUDA Module | Usage Count | Domains |
|------------------|-------------|---------|
| `barracuda::tensor` | 90+ ops | All GPU validators |
| `barracuda::linalg` | `eigh_f64`, `solve_f64`, `cholesky_f64`, `lu_*`, `svd_*`, `tridiag` | Spectral, Anderson, baseCamp |
| `barracuda::stats` | `variance`, `pearson_correlation`, `covariance`, `norm_cdf` | 13 papers |
| `barracuda::special` | `gamma`, `erf`, `bessel`, `legendre`, `hermite`, `laguerre` | NIST DLMF checks |
| `barracuda::optimize` | `nelder_mead`, `bisect`, `brent` | Isotherm, fitting |
| `barracuda::nn` | `SimpleMlp` | WDM surrogates (nW-01, nW-02) |
| `barracuda::numerical` | `rk45_solve` | Regulatory, signal, game theory |
| `barracuda::shaders::provenance` | 22 shaders | Cross-spring edges |
| `barracuda::nautilus` | Brain, drift monitor | baseCamp nS-05 |
| `barracuda::shaders::precision` | CPU add/mul/fma/dot/sum | 12 exact-f64 checks |

## Part 4: Absorption-Ready Shaders

These metalForge WGSL shaders are candidates for upstream absorption:

| Shader | Domain | Status | Absorption Priority |
|--------|--------|--------|---------------------|
| `xoshiro128ss.wgsl` | GPU PRNG | Validated (5/5) | P1 — enables stochastic GPU |
| `logsumexp_reduce.wgsl` | HMM/softmax | Validated (5/5) | P1 — numerical stability |
| `swarm_nn_scores.wgsl` | Swarm robotics | Validated (9/9) | P2 |
| `stencil_cooperation.wgsl` | QS spatial | Validated (3/3) | P2 |
| `rk45_adaptive.wgsl` | ODE integration | Validated (6/6) | P2 |
| `wright_fisher_step.wgsl` | Population genetics | Validated (4/4) | P2 |

15 coralForge df64 shaders are already integrated via `compile_shader_df64`.

## Part 5: Evolution Gaps (For ToadStool/BarraCUDA Team)

### GPU Training Infrastructure

neuralSpring validates **inference** on GPU but **training** remains CPU-only:

- **Autograd**: No GPU autograd — backpropagation is CPU.
- **`nn::Layer` trait**: No composable GPU layer abstraction.
- **Optimizers**: SGD/Adam/AdaGrad are CPU-only.

These are the primary gaps blocking full GPU training pipelines.

### Benchmark Gaps

- **Python parity benchmarks**: neuralSpring has 15-domain CPU Rust-vs-Python
  benchmarks (38.6× geomean). Need equivalent for BarraCUDA CPU-vs-Python.
- **Kokkos/cuBLAS parity**: `bench_kokkos_parity` harness exists (9 ops ×
  production scale) but reports neuralSpring-only. Need upstream Kokkos
  comparison data.

### Small-Matrix CPU Implementations (Intentional)

Two functions intentionally remain local CPU rather than BarraCUDA dispatch:

- `information_flow::mat_mul_transpose` — n=4-8 matrices, GPU dispatch
  overhead exceeds computation.
- `glucose_prediction::solve_symmetric` — n≤73 Cholesky, same reasoning.

Both are documented with justification comments in source.

## Part 6: Quality Gates (Reproducible)

```bash
cargo fmt --check                                          # PASS
cargo clippy --all-targets --all-features -- -D warnings   # 0 warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps             # 0 warnings
cargo test --lib                                           # 1048/1048 PASS
cargo test --test integration                              # 9/9 PASS
cargo run --release --bin validate_all                     # 220/220 PASS
```

---

*This handoff is unidirectional: neuralSpring → ecosystem. No response expected.*
*License: AGPL-3.0-or-later*
