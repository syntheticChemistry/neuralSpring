# Changelog

All notable changes to neuralSpring are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Session 98 (March 1, 2026)

### Session 98 — coralForge nF-03 AlphaFold3 GPU Tier Closure (March 1, 2026)

**New validators:**
- `validate_alphafold3_diffusion_gpu` (14/14): Forward diffusion, DDPM/DDIM reverse, SE(3) equivariance, pair FFN — all via BarraCUDA Tensor on RTX 4070
- `validate_alphafold3_pairformer_gpu` (12/12): Timestep conditioning, TriMul outgoing/incoming, triangle attention QKV, FFN, full block via matmul_ref

**Expanded validators:**
- `validate_gpu_pure_wdm_coral` (22→24): +AF3 diffusion forward (mean readback), PF FFN (Frobenius), PF TriMul (Frobenius)
- `bench_cross_spring_evolution` (33→40): +7 AF3 CPU throughput benchmarks (cosine schedule, forward diffusion, DDPM, DDIM, SE(3), FFN, sinusoidal embedding)

**Cross-spring provenance:**
- hotSpring: df64 precision shaders enable fp48 accuracy on consumer FP32 cores
- wetSpring: bio-domain scheduling patterns inform diffusion noise schedules

**Quality metrics:** 211 binaries, 199/199 validate_all (197 PASS + 2 pre-existing), 685 lib tests, 3490+ checks, 0 clippy warnings.

### Session 97d — ToadStool S70+++ Cross-Spring Evolution Validation (February 28, 2026)

- **New validator**: `validate_toadstool_s70_evolution` (27/27 PASS) — exercises all five springs' contributions absorbed into BarraCUDA S70+++. groundSpring: Kimura fixation, error threshold, jackknife. airSpring: FAO-56 ET₀, Hargreaves, crop coefficient, soil water balance. wetSpring: `chao1_classic` (u64) vs `chao1` (f64). neuralSpring: `matmul_ref` non-consuming proof (bit-identical to consuming), `SimpleMlp` forward+JSON round-trip. S70+++ throughput benchmark with provenance table.
- **Expanded `bench_cross_spring_evolution`**: S70+++ section — Kimura, jackknife, fao56_et0, chao1_classic, SimpleMlp benchmarks with provenance annotations. Updated summary to S97d.
- **Updated cross-spring provenance**: `validate_modern_cross_spring` and `bench_cross_spring_evolution` summaries refreshed with S70+++ absorption details and S97d session tags.
- **Quality**: `cargo fmt` clean, `cargo clippy --all-targets` 0 warnings, `cargo test --lib` 685 PASS, `validate_all` 197/197 (195 PASS + 2 pre-existing wright_fisher WGSL parse).
- **Metrics**: 209 binaries, 3450+ checks. 46 upstream rewires + 6 shader sources.

### Session 97c — nF-03 bC Tier Closure + CPU↔GPU Domain Parity + metalForge NUCLEUS (February 28, 2026)

- **nF-03 BarraCUDA CPU tier closure**: `validate_barracuda_alphafold3` (13/13 PASS) — proves BarraCUDA CPU math matches neuralSpring for AF3 diffusion, Pairformer, and confidence head primitives. Closes BarraCUDA CPU 2/3 → 3/3 for coralForge.
- **WDM+coralForge CPU↔GPU domain parity**: `validate_wdm_coral_parity` (39/39 PASS) — proves BarraCUDA CPU and GPU produce bit-identical results for domain-level WDM surrogate and coralForge compositions through the Dispatcher. Covers MLP, EOS, LSTM, ESN spectral radius, Evoformer attention, triangle multiply, pLDDT, layer norm, SE(3).
- **metalForge NUCLEUS atomics**: `validate_metalforge_wdm_coral` (41/41 PASS) — validates mixed-hardware routing (Tower discovery, Node compute dispatch, Nest provenance) and PCIe bypass cost modeling for WDM and coralForge workloads.
- **ToadStool pin bump**: `e96576ee` (S68+) → `1dd7e338` (S70+++) — absorbs 13 commits including cross-spring absorption (7 DF64 ML shaders, SimpleMlp, matmul_ref, SymmetrizeGpu, LaplacianGpu, stats::evolution/jackknife/hydrology), ComputeDispatch migration, chrono elimination, unsafe reduction 47→45, dead code cleanup. Pin updated in 20+ doc/source files.
- **matmul_ref rewire**: 2 sites (validate_barracuda_wdm_esn.rs, bench_barracuda_tensor.rs) now use non-consuming `matmul_ref` instead of `clone().matmul`, eliminating unnecessary GPU buffer copies.
- **Quality**: `cargo fmt` clean, `cargo clippy --all-targets` 0 warnings, `cargo test --lib` 685 PASS, `validate_all` 196/196 (194 PASS + 2 pre-existing wright_fisher WGSL parse). Fully re-validated against new ToadStool pin.
- **Metrics**: 208 binaries, 3420+ checks. V64 handoff crafted with ToadStool absorption review. All root docs updated.

### Session 94 — coralForge Rename + Deep Debt Resolution (February 28, 2026)

- **coralForge**: Renamed `sovereign_folding/` + `structure_module/` → unified `coral_forge/` with `structure/` submodule. Updated 25+ source files, 3 validation binaries, Cargo.toml, control scripts, specs, docs. RPC capability names (`science.structure_module`) stable for protocol compatibility.
- **Magic number elimination**: 5 new domain-specific tolerance constants (`FISHER_EPS`, `BURGERS_IC_GUARD`, `DP_EQUALITY_EPS`, `SINGLETON_FREQ_EPS`, `PHENOTYPE_TIE_EPS`) in `tolerances/mod.rs`. Zero inline magic numbers remain in production code.
- **expect() → require!**: Evolved 24 `expect()` calls in `validate_coral_forge_gpu`, `validate_coral_forge_gpu_pipeline`, and `validate_barracuda_alphafold2` to graceful `require!(h, ...)` error recording.
- **Cast safety**: `cpu_fallback.rs` activator indices now bounds-checked via `safe_idx()`.
- **Provenance docs**: All 34 `BaselineProvenance` constants documented with `///` comments.
- **Dependency analysis**: All 12 external deps are pure Rust, zero C/C++ wrappers, documented in EVOLUTION_READINESS.md.
- **Metrics**: 208 binaries, 685 lib tests, 9 integration, 139+ named tolerances, 0 clippy pedantic warnings, 0 doc warnings. All quality gates green.

### Session 93 — Deep Debt Evolution + nF-03 Phase C Confidence Heads (February 28, 2026)

- **Deep debt evolution**: `dispatch_ops.rs` (842→7 domain files), `gpu_ops/mod.rs` (668→38+tests_ops). Iterator evolution across 6 core modules. Self-identification→`env!("CARGO_PKG_NAME")`. `.unwrap()`→`.expect()`.
- **nF-03 Phase C: Confidence Heads**: pLDDT, PAE, pDE, ranking score — Py 19/19, Rs 16/16, 7 new unit tests. New `coral_forge/confidence.rs` module.
- **Metrics**: 201 binaries, 685 lib tests, **189/189 validate_all**, 39 Python drift baselines. 5 clippy warnings (all pre-existing pedantic).

### Session 92 — nF-03 AlphaFold3 Phase A+B (February 27, 2026)

- **Diffusion primitives**: cosine/linear schedules, forward diffusion, DDPM/DDIM reverse, SE(3)-equivariant noise — Py 29/29, Rs 26/26.
- **Pairformer block**: sinusoidal embedding, conditioning, triangle ops + FFN — Py 14/14, Rs 13/13.
- **Metrics**: 196 binaries, 680 lib tests, 184/184 validate_all. 38 Python drift baselines.

### Session 88+ — BarraCUDA CPU Parity & GPU Portability Benchmarks (February 27, 2026)

- **`validate_barracuda_cpu_bench`** (25/25 PASS): Cross-language benchmark proving BarraCUDA CPU is pure math and 83.6× faster than Python/NumPy (geometric mean across 11 paper domains). Fastest: multi-objective fitness 1104×, NK fitness 820×, pairwise L2 314×. One domain (commutator 64×64) is 0.4× because NumPy delegates to BLAS — documented and expected.
- **`bench_portability_tiers`** (9/9 PASS): CPU→GPU portability proof across 7 domains. Proves same math produces identical results at every tier: Python → BarraCUDA CPU → BarraCUDA GPU. ToadStool unidirectional streaming pattern validated (upload → compute → scalar readback).
- Total: **175 binaries**, **174/175 validate_all** (1 pre-existing WDM damping assertion), **668 lib tests**, **3034+ checks**.

### Changed (ToadStool `1dd7e338` sync)

- **`compile_shader_f64_hybrid` rewired**: Now delegates to upstream
  `WgpuDevice::compile_shader_df64()` instead of manually prepending DF64
  core/transcendentals from `barracuda::ops::lattice::su3` constants.
  Upstream method provides ILP optimizer + Sovereign compiler pipeline.
- **ToadStool pin updated**: `f0feb226` → `1dd7e338` (3 new commits:
  CPU feature-gate fix, root docs cleanup, GPU device-lost resilience).
  Pin updated across 17 documentation files.
- **Previously-missing APIs confirmed upstream**: `LogSumExp` (wired S51),
  `PairwiseDistance` (wired via PairwiseL2Gpu), `BatchedEighGpu` (wired
  for eigensolver). All 3 items from V55 "Not Yet Used" list now resolved.
- ToadStool universal precision pipeline: `compile_shader_universal(source,
  precision)` with F16/F32/F64/DF64 variants. 703 WGSL shaders, all f64
  canonical. Zero f32-only shaders remain upstream.

### Added

- **GPU tier: Exp-050** (`validate_barracuda_training_trajectory`): 9/9 — eigensolve → IPR
  → variance on GPU for training trajectory spectral analysis.
- **GPU tier: Exp-052** (`validate_barracuda_hessian_eigen`): 10/10 — Hessian eigensolve
  → spectral diagnostics on GPU for loss landscape analysis.
- **GPU tier: Exp-053** (`validate_barracuda_anderson_multiagent`): 11/11 — Laplacian →
  disordered eigensolve → IPR + pairwise L2 on GPU for multi-agent coordination.
- **bench_modern_rewire**: New binary (23/23 PASS) validating modern typed-op rewires.
- **Modern rewires** (S88+): pairwise_l2_matrix_gpu→PairwiseL2Gpu,
  geographic_distance_matrix_gpu→PairwiseL2Gpu, disorder_sweep_gpu IPR→BatchIprGpu.
- **Pipeline + metalForge** (`validate_publication_gpu_pipeline`): 13/13 — BatchIprGpu
  pure GPU pipeline, Dispatcher CPU↔GPU parity, metalForge mixed-hardware routing.
- **Exp-050** (training trajectory spectral analysis): Py 11/11 + Rs 12/12 PASS.
- **Exp-052** (Hessian eigenanalysis): Py 8/8 + Rs 14/14 PASS.
- **Exp-053** (Anderson multi-agent QS): Py 11/11 + Rs 18/18 PASS.
- ToadStool/BarraCUDA absorption handoff V54: barracuda evolution audit, debt
  reduction, control matrix verification, absorption targets refreshed.
- Root docs audit: README, CHANGELOG, CONTROL_EXPERIMENT_STATUS, baseCamp,
  experiments/ journal, wateringHole handoffs, specs/ all updated.
- **biomeOS integration**: `neuralspring_primal` JSON-RPC server binary
  (feature-gated `--features primal`). 7 science capabilities registered in
  biomeOS capability registry. `neuralspring_spectral_pipeline.toml` graph for
  biomeOS orchestration. `validate_biomeos_spectral`: 29/29 PASS.
- **biomeOS SDK**: `PrimalCapability::science()` added to `biomeos-types`.
  `providers_for_capability()` updated to include `neuralspring` for science.
- **Publication mixed-hardware** (`validate_publication_mixed_hardware`): 43/43 — 
  Exp-050/052/053 extended to metalForge mixed-hardware tier. NPU→GPU PCIe bridge,
  GPU→CPU fallback, substrate cost model routing, NUCLEUS atomic transfer hierarchy.
- **NUCLEUS compute dispatch** (`validate_nucleus_compute_dispatch`): 39/39 —
  Tower discovery (CPU+GPU substrate inventory), Node eigensolve/Anderson/Hessian
  compute dispatch, Nest provenance (mean/variance/Frobenius parity), mixed atomic
  coordination, PCIe bypass validation.
- **ToadStool spectral absorption** (`validate_toadstool_spectral_absorption`): 294/294 —
  CPU correctness (eigh trace/eigenvector norms, Anderson localization ratio, Hamiltonian
  symmetry), GPU dispatch parity (8×8/16×16/24×24 + stats), batch scaling, mixed substrate
  routing (large→GPU, small→CPU, realtime→NPU).
- **Phase 4 WGSL shader validation** (`validate_gpu_shader_phase4`): 22/22 — Direct
  metalForge shader dispatch for HMM backward (log-domain), HMM Viterbi decoding,
  matrix correlation (Pearson of N×N upper triangle), linear regression (OLS normal
  equations). All shaders validated against CPU references via `gpu_shader_validation`
  infrastructure. ToadStool absorption targets for `barracuda::ops::bio::hmm_*` and
  `barracuda::stats::*_gpu`.
- **ToadStool streaming spectral pipeline** (`validate_streaming_spectral_pipeline`):
  28/28 — Demonstrates unidirectional streaming pattern: batch eigensolve → BatchIprGpu
  → variance/mean aggregation with minimal CPU round-trips. Anderson disorder sweep
  across 6 W values (0.5→16) shows localization transition on GPU (IPR 0.09→0.79).
  Dispatcher pipeline parity at 1.6e-14 (machine ε). This is the structural proof
  that ToadStool's unidirectional streaming will preserve scientific conclusions.

### Changed

- **WDM SQW JSON fix**: `wdm_sqw.rs` loader now accepts both `spec_mean`/`spec_std`
  and `series_mean`/`series_std` field names. Feature strategy auto-detected from
  `w_out` dimensions (32-dim h_last vs 96-dim pooled). 0/1 → 26/27 PASS.
- **Debt reduction**: 18 `unwrap_or_else(|e| panic!(...))` sites evolved to
  idiomatic `.expect()` across WDM tests (`wdm_sqw`, `wdm_esn`, `wdm_transport`,
  `wdm_surrogate`) and `validate_basecamp_gpu.rs`. 3 bare `.unwrap()` in
  `bench_cross_spring_evolution.rs` replaced with descriptive `.expect()`.
- **Iterator idioms**: 11 manual loop sites evolved to `chunks_exact`, `flat_map`,
  `zip`, `recip` patterns in `basecamp.rs` (4 sites: belief propagation, MLP
  signal, pairwise L2, adjacency) and `coral_forge.rs` (7 sites: layer_norm,
  softmax, SDPA scores, attention apply, triangle mul ×2, outer product mean).
- **Module-level `#[allow(clippy::expect_used)]`**: Added to WDM test modules and
  basecamp GPU validation binary; redundant per-test allows removed.
- `whitePaper/baseCamp/extensions.md`: Session range extended through S88+.
- `specs/PAPER_REVIEW_QUEUE.md`: Control matrix verified for open data ×
  BarraCUDA CPU × BarraCUDA GPU × metalForge hardware tiers.
- `specs/BARRACUDA_USAGE.md`: Absorption inventory refreshed.

### Validation

- `cargo fmt --check`: PASS
- `cargo clippy --all-targets`: 0 warnings
- `cargo test --workspace`: PASS
- `validate_all`: **174/175 PASS** (175 binaries, 1 pre-existing WDM damping assertion)
- `validate_biomeos_spectral`: **29/29 PASS** (biomeOS primal integration, feature-gated)
- `validate_gpu_shader_phase4`: **22/22 PASS** (Phase 4 WGSL direct shader dispatch)
- `validate_streaming_spectral_pipeline`: **28/28 PASS** (ToadStool streaming proof)
- Publication experiments: full GPU progression (Py → Rs → GPU → Pipeline → metalForge)
- Documentation sweep: all counts aligned (3034+ checks, 175 binaries, 668 lib tests)

## [0.5.2] — 2026-02-27 (Session 88: df64 Core Streaming — coralForge)

### Changed

- All 15 coralForge WGSL shaders evolved to hotSpring/ToadStool df64 core
  streaming pattern: f64 buffer I/O → df64 compute on FP32 cores → f64 output.
  Three-zone architecture: `df64_from_f64` at load, `df64_*` arithmetic and
  transcendentals for compute, `df64_to_f64` at store.
- `src/gpu.rs`: Added `create_buffer_f64()`, `upload_f64()`, and
  `compile_shader_f64_hybrid()` (prepends `df64_core.wgsl` +
  `df64_transcendentals.wgsl` then calls `compile_shader_f64`).
- `validate_coral_forge_gpu`: Rewritten for f64 I/O with two-tier
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
- `validate_coral_forge_gpu`: **37/37 PASS** (df64 core streaming)

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

- ToadStool HEAD updated from `17932267` (S65) to `1dd7e338` (S70+++) across
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
  bridge, coralForge shaders.
