# Changelog

All notable changes to neuralSpring are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Session 88 (February 27, 2026)

Next session.

## [0.5.2] — 2026-02-27 (Session 88: df64 Core Streaming — Sovereign Folding)

### Changed

- All 15 sovereign folding WGSL shaders evolved to hotSpring/ToadStool df64 core
  streaming pattern: f64 buffer I/O → df64 compute on FP32 cores → f64 output.
  Three-zone architecture: `df64_from_f64` at load, `df64_*` arithmetic and
  transcendentals for compute, `df64_to_f64` at store.
- `src/gpu.rs`: Added `create_buffer_f64()`, `upload_f64()`, and
  `compile_shader_f64_hybrid()` (prepends `df64_core.wgsl` +
  `df64_transcendentals.wgsl` then calls `compile_shader_f64`).
- `validate_sovereign_folding_gpu`: Rewritten for f64 I/O with two-tier
  tolerance: `GPU_DF64_TOL = 1e-6` (arithmetic), `GPU_DF64_TRANS_TOL = 5e-4`
  (transcendental). `Fp64Strategy::Hybrid` auto-detected on RTX 4070.
- `specs/PAPER_REVIEW_QUEUE.md`: Updated shader table with new precision
  tiers and observed max-diff values. Added precision hierarchy documentation
  (fp16 → bf16 → f32 → df64/fp48 → f64).

### Validation

- `cargo fmt --check`: PASS
- `cargo clippy --workspace -- -D warnings`: 0 warnings
- `cargo test --workspace`: 675/675 PASS
- `validate_all`: **158/158 PASS** (was 156)
- `validate_sovereign_folding_gpu`: **37/37 PASS** (df64 core streaming)

### Precision Results (RTX 4070, Fp64Strategy::Hybrid)

| Tier | Operations | Tolerance | Observed |
|------|-----------|-----------|----------|
| Arithmetic | dot products, matmul, accumulate, `sqrt_df64` | 1e-6 | 3.6e-8 to 5.6e-7 |
| Transcendental | `exp_df64`, `tanh_df64` (Horner degree-6) | 5e-4 | 1.7e-4 to 3.4e-4 |

## [0.5.1] — 2026-02-26 (Session 87: WDM Queue Closed — nW-03, nW-05)

### Added

- `src/wdm_sqw.rs`: LSTM reservoir S(q,ω) peak predictor module
- `src/wdm_esn.rs`: ESN WDM regime classifier module
- `control/wdm/sqw_peak_predictor.py`: nW-03 Python baseline (LSTM on MD time series, R²=0.98)
- `control/wdm/esn_regime_classifier.py`: nW-05 Python baseline (ESN classifier, 96.5% accuracy)
- `src/bin/validate_wdm_sqw.rs`: 27/27 PASS — loaded, finite, positive, deterministic, monotonic
- `src/bin/validate_wdm_esn.rs`: 39/39 PASS — label parity, score parity, physics constraints
- 2 new baselines in `check_drift.sh` (31 total)

### Changed

- `validate_all.rs`: 156 binaries (was 154)
- WDM surrogate queue fully closed: nW-01 through nW-05 all complete
- 623 lib tests (was 611), 40 modules (was 38), 172 binaries (was 170)

### Validation

- `cargo fmt --check`: PASS
- `cargo clippy --workspace -- -D warnings`: 0 warnings
- `cargo doc --workspace --no-deps`: 0 warnings
- `cargo test --workspace`: 623/623 + 43 + 9 = 675 PASS
- `validate_wdm_sqw`: 27/27 PASS
- `validate_wdm_esn`: 39/39 PASS

## [0.5.0] — 2026-02-26 (Session 86: WDM Buildout + V50 Handoff)

### Added

- `src/wdm_transport.rs`: New module for nW-01 Stanton-Murillo transport
  surrogate (MLP 3→H→3, log-space normalization, diffusivity/viscosity/thermal
  conductivity prediction).
- `src/bin/validate_wdm_transport.rs`: 30 checks (loaded, finite, positive,
  deterministic, monotonic per coefficient).
- `src/bin/validate_wdm_transfer.rs`: 6 checks (classical R² > 0.85, transfer
  R² > 0.40, determinism, Python baseline transfer advantage).
- `validate_wdm_eos` and `validate_barracuda_wdm_eos` wired into `Cargo.toml`
  and `validate_all.rs` (154 total binaries).
- `wdm/transport_surrogate.py` and `wdm/transfer_classical_to_wdm.py` added to
  `check_drift.sh` (29 baselines total).
- V50 handoff: WDM buildout learnings, SimpleMLP absorption target, cross-language
  RNG divergence documented.
- Experiment 054: Session 86 WDM surrogate buildout.

### Changed

- README.md: Updated all counts (611 lib, 170 binaries, 223 Py, 2350+ total),
  added WDM surrogates section, updated directory structure.
- All root docs (CONTROL_EXPERIMENT_STATUS, EVOLUTION_READINESS, baseCamp
  extensions, PAPER_REVIEW_QUEUE) updated to Session 86 numbers.
- V49 handoff archived.

### Validated

- 611/611 lib, 663/663 total tests, 0 clippy warnings, 154/154 validators PASS.

## [0.4.2] — 2026-02-26 (Session 85: Doc Sweep + V49 Handoff)

### Changed

- All stale test counts fixed across 20+ documents: 580→604 lib, 163→166
  binaries, 107→129+ tolerances, V43→V48 handoff refs.
- baseCamp sub-theses (sub01–sub05) extended through S85.
- `waters.md`: Fixed `quorum_sensing.rs` → `signal_integration.rs`.
- `BARRACUDA_EVOLUTION.md`: PcieBridge placeholder replaced with real content.
- Five-spring provenance documented in `CROSS_SPRING_SHADER_LINEAGE.md`.
- Hamming 20.85× regression flagged in BARRACUDA_USAGE + V49 handoff.

### Added

- V49 handoff: cross-spring evolution learnings, recommendations for ToadStool.
- Experiment 053: Session 85 doc sweep + handoff.

### Validated

- 604/604 lib, 0 clippy warnings, 150/150 GPU validators PASS.

## [0.4.1] — 2026-02-26 (Session 84: Cross-Spring Benchmark + Lineage)

### Added

- `bench_cross_spring_evolution`: 5 new S68 API benchmarks (fit_quadratic,
  fit_exponential, fit_all, spearman_correlation, rawr_mean) + GPU dispatch
  provenance benchmarks (variance, pearson, shannon, matmul via Dispatcher).
  28/28 PASS with full five-spring provenance annotations.
- `CROSS_SPRING_SHADER_LINEAGE.md`: Expanded from 3 Springs to 5 Springs
  (added airSpring, groundSpring). Full provenance map with ~700 WGSL shaders
  across all Springs.

### Validated

- 604/604 lib, 0 clippy warnings, 150/150 GPU validators, 28/28 bench PASS.
- Full benchmark suite: dispatch tiers, evolution tiers, upstream vs local,
  GPU kernels, barracuda tensor, basecamp parity, rewire evolution.

## [0.4.0] — 2026-02-26 (Session 83: ToadStool S68 Universal Precision Sync)

### Fixed

- 5 shader imports broken by ToadStool S68 universal precision evolution:
  `WGSL_PAIRWISE_JACCARD`, `WGSL_SPATIAL_PAYOFF`, `WGSL_PAIRWISE_HAMMING`
  (privatized → local copies), `WGSL_LOCUS_VARIANCE` (renamed → f64 const),
  `rk4_parallel.wgsl` (renamed → local f32 copy).
- 2 validator binaries rewired: `validate_gpu_pipeline_swarm` and
  `validate_gpu_logsumexp` now use forge shader constants.

### Changed

- ToadStool HEAD updated from `17932267` (S65) to `f0feb226` (S68) across
  14 active files.
- API gap #3 (variance_ddof) closed upstream — documented in BARRACUDA_USAGE.

### Validated

- 604/604 lib, 43/43 forge, 0 clippy warnings, 150/150 GPU validators PASS.

## [0.3.0] — 2026-02-26 (Session 82: Titan V Pure Rust Pipeline Validation)

### Fixed

- `batched_eigh_nak_optimized_f64.wgsl`: replaced `fma(f64)` calls (not valid
  WGSL per spec) with `a * b + c` — Sovereign Compiler re-fuses into
  `OpFMulAdd` at IR level. Zero performance regression.
- Explicit f64 typing for bare float literals in `select()` and division
  contexts — prevents abstract-float-to-f32 coercion causing type mismatches.

### Validated

- 384/384 GPU checks PASS on NVIDIA TITAN V (NVK GV100, Volta SM70,
  full-rate FP64) — 33 validation binaries across all domains.
- RTX 4070 regression: zero regressions after shader fix.
- Library tests: 604/604 PASS.

## [0.2.0] — 2026-02-26 (Session 81: Deep Debt Evolution)

### Added

- 25 new named tolerance constants centralizing all previously inline magic
  numbers across validation binaries (`LEVEL_SPACING_GOE_SLACK`,
  `SPECTRAL_IPR_COMPARISON_SLACK`, `NUMERICAL_DISTINCTNESS`,
  `FST_IDENTICAL_POP_TOL`, `FST_ESTIMATOR_AGREEMENT`,
  `GAME_DEFECTION_UPPER`, `GAME_QS_COOPERATION_MIN`, `GAME_QS_VARIANCE_MAX`,
  `RELATIVE_ERROR_FLOOR`, `ODE_STEADY_STATE_SLACK`, `QUANT_Q8_GEMV_ERROR`,
  `QUANT_Q4_GEMV_ERROR`, `QUANT_SIGN_AGREEMENT`, `GATE_DISORDER_COMPARISON`,
  `SPECTRAL_RADIUS_SWEEP_SLACK`, `GPU_COMMUTATOR_NEAR_ZERO_F64`,
  `GPU_COMMUTATOR_RESIDUAL_F64`, plus 8 hardware dispatch constants).
- Tolerance registry categories: `training_quantized`, `hardware`.
- Cross-platform `probe.rs`: `#[cfg(target_os = "linux")]` gating for
  `/proc/cpuinfo` and `/proc/meminfo` reads with platform-agnostic fallbacks.
- PyTorch deterministic seeding (`torch.manual_seed(42)`,
  `torch.cuda.manual_seed_all(42)`, `cudnn.deterministic = True`) in 7
  Python training scripts for full baseline reproducibility.

### Changed

- `weight_spectral::spectral_entropy` now delegates to
  `barracuda::stats::shannon_from_frequencies` — eliminates last duplicate
  math between neuralSpring and barracuda.
- ~50 inline magic-number tolerances across 17+ validation binaries replaced
  with named constants from the `tolerances` module.
- `wdm_surrogate` test uses idiomatic `.expect_err()` instead of
  `.err().expect()`.

### Fixed

- Clippy `doc_markdown` lint in `validation.rs` doc comment.
- Clippy `err_expect` lint in `wdm_surrogate.rs` test.
- `PCIe` properly backticked in tolerance doc comments for `doc_markdown`.
- f32/f64 type mismatch in `validate_gpu_stateful_pipeline.rs` and
  `validate_gpu_rk4.rs` steady-state checks.

## [0.1.0] — 2026-02-25

### Summary

Initial release: 206/206 Python PASS, 2040+ Rust+GPU PASS, 604 lib tests,
166 validation binaries, 93.5% coverage.  All 17 ToadStool shortcomings
resolved.  AGPL-3.0-or-later.

- Phase 0: surrogates, transformer, metrics, LeNet, transfer, isomorphic,
  LSTM, quantized, sequence.
- Phase 0+: scholarly reproduction (Iram 2020, Liu 2014, Bruger 2018, etc.).
- Phase 0++: 25 papers across evolution, phylogenetics, game theory, spectral
  theory, population genetics.
- baseCamp: 5 biophysical AI interpretability modules (weight spectral,
  information flow, loss landscape, neural PGM, agent coordination).
- metalForge: GPU dispatch, substrate discovery, workload tracking, BarraCUDA
  bridge, sovereign folding shaders.
