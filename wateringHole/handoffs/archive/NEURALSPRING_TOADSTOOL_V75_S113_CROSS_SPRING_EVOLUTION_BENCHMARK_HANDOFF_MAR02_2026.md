<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/BarraCUDA Handoff V75 — Cross-Spring S86 Evolution Benchmark

**Date**: March 2, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 113 — Cross-spring evolution benchmark, modern validation, provenance tracking
**Supersedes**: V74 (S86 Rewire + Nautilus Absorption)
**ToadStool HEAD**: `2fee1969` (S86)

---

## Executive Summary

- **Cross-spring evolution validated**: `validate_modern_cross_spring` 57 → **68/68 PASS**
- **Cross-spring benchmark expanded**: `bench_cross_spring_modern` 10 → **14/14 PASS**
- **6-spring provenance tracked**: hotSpring, wetSpring, airSpring, groundSpring, bingoCube, neuralSpring
- **All 5 S81 hydrology methods exercised**: thornthwaite, hamon, makkink, turc, hargreaves ET₀
- **Nautilus brain/drift/bridge benchmarked** via barracuda::nautilus absorption
- **692+ WGSL shaders** in ToadStool S86; 15+ neuralSpring shaders absorbed upstream
- **208/208 validate_all PASS**, 861 lib tests, 0 clippy, 0 fmt

---

## Part 1: Cross-Spring Provenance Map

neuralSpring's validation now explicitly traces every validated operation back through the ToadStool absorption chain to its originating Spring.

| Source Spring | → ToadStool/BarraCUDA | → neuralSpring Usage |
|--------------|----------------------|---------------------|
| **hotSpring** | DF64 core, Fp64Strategy, split_workgroups, lattice QCD, universal precision, brain arch (S80 nautilus) | Dispatcher GPU path, eigh/eigensolve, compile_shader_universal, gelu/sigmoid/softmax_df64, NautilusBrain observations |
| **wetSpring** | diversity (Shannon, Bray-Curtis, Simpson, chao1), NMF, HMM, ODE bio, ridge | alpha_diversity, HMM chains, FST, chao1_classic |
| **airSpring** | regression, hydrology (RMSE, R², NSE, MAE), moving_window, spearman, Thornthwaite/Hamon/Makkink/Turc ET₀ (S81) | fit_linear/quad/exp, fao56_et0, crop_kc, soil_water_balance, thornthwaite/hamon/makkink/turc_et0 |
| **groundSpring** | bootstrap (rawr_mean), multinomial, MC propagation, evolution, jackknife | bootstrap_ci, norm_cdf/pdf/ppf, kimura, jackknife |
| **bingoCube** | NautilusBrain, DriftMonitor, EvolutionConfig, NautilusShell (absorbed S80) | ABSORBED into barracuda::nautilus — dep removed in nS S112 |
| **neuralSpring** | batch_fitness, pairwise ops, eigh, swarm_nn, matmul_ref, SimpleMlp, fused_chi²_f64, fused_kl_divergence_f64, SpectralNautilusBridge | Dispatcher (47 ops), graph_laplacian, effective_rank, WDM MLP, nautilus_bridge roundtrip |

---

## Part 2: What S113 Validated (New Checks)

### 2.1 validate_modern_cross_spring (+11 checks → 68/68)

| Check | Provenance Chain | Result |
|-------|-----------------|--------|
| nautilus brain creation | hotSpring brain arch → bingoCube → TS S80 | PASS |
| nautilus observe (QCD) | hotSpring lattice QCD → nS spectral bridge | PASS |
| DriftMonitor N_e·s | hotSpring brain → bingoCube → TS S80 | PASS (exact f64) |
| SpectralNautilusBridge train | nS → TS nautilus absorption | PASS |
| SpectralNautilusBridge predict | nS → hS → bC → TS → nS roundtrip | PASS |
| thornthwaite_et0 | airSpring → TS S81 hydrology | PASS |
| thornthwaite_heat_index | airSpring → TS S81 hydrology | PASS |
| hamon_et0 | airSpring Tier A → TS S81 | PASS |
| makkink_et0 | airSpring Tier A → TS S81 | PASS |
| turc_et0 | airSpring Tier A → TS S81 | PASS |
| ComputeDispatch 144 ops | S80-S86 evolution | PASS |

### 2.2 bench_cross_spring_modern (+4 checks → 14/14)

| Benchmark | Timing | Provenance |
|-----------|--------|-----------|
| 5 hydrology ET₀ methods | sub-µs each | airSpring → TS S81 |
| NautilusBrain::new | 8.6 µs | hotSpring → bingoCube → TS S80 |
| NautilusBrain::observe | 7.3 µs | hotSpring QCD → TS S80 |
| NautilusShell::from_seed | 7.4 µs | TS evolutionary |
| DriftMonitor 20-gen cycle | 0.5 µs | bingoCube → TS S80 |
| SpectralNautilusBridge roundtrip | 950 ms | nS → TS absorption (ESN inside) |

---

## Part 3: What neuralSpring Contributes Back to ToadStool

### 3.1 Shaders Absorbed Upstream (15+ neuralSpring → barracuda)

| Shader | BarraCUDA Location | Session |
|--------|-------------------|---------|
| matmul_cpu_tiled, matmul_gpu_evolved | shaders/math/ | nS S-02 |
| eigh_f64 (Householder+QR) | ops::linalg::eigh_f64 | nS S-12 |
| pairwise_hamming, pairwise_jaccard | shaders/math/ | nS S-25 |
| locus_variance | shaders/bio/ | nS S-25 |
| spatial_payoff | shaders/math/ | nS S-25 |
| batch_ipr | shaders/spectral/ | nS S-25 |
| hmm_forward_log, batch_fitness_eval | shaders/ml/ | nS S-25 |
| rk4_parallel | shaders/numerical/ | nS S-25 |
| pairwise_l2, multi_obj_fitness | shaders/math/, shaders/bio/ | nS S-42 |
| hill_gate, swarm_nn_forward | shaders/bio/ | nS S-42 |
| fused_chi_squared_f64 | shaders/special/ | nS V24 |
| fused_kl_divergence_f64 | shaders/special/ | nS V24 |

### 3.2 Credited in BarraCUDA Source

| File | Credit |
|------|--------|
| barracuda/src/shaders/special/fused_chi_squared_f64.wgsl | `// neuralSpring V24` |
| barracuda/src/shaders/special/fused_kl_divergence_f64.wgsl | `// neuralSpring V24` |
| barracuda/src/ops/bio/mod.rs | `neuralSpring metalForge (Feb 2026)` |
| barracuda/src/ops/logsumexp.rs | `(neuralSpring)` |
| barracuda/src/spectral/stats.rs | `neuralSpring V69 handoff` |
| barracuda/src/nautilus/spectral_bridge.rs | `neuralSpring S102 nautilus_bridge.rs` |

---

## Part 4: Rewire Opportunities Identified

| Priority | Item | Files | Recommendation |
|----------|------|-------|---------------|
| Low | `primitives::rk4_step` | regulatory_network.rs, signal_integration.rs | Local single-step RK4 for fixed-step ODE integration. barracuda has adaptive `rk45_solve` — different algorithm. Keep local for fixed-step cases. |
| Low | `cpu_fallback::variance` | gpu_dispatch/cpu_fallback.rs | Fallback when GPU unavailable. Could use `barracuda::stats::variance_ddof(data, 0)` for population variance. |
| Low | `jacobi_eigh` naming | anderson_localization.rs | Name is misleading (uses Householder, not Jacobi). Already delegates to barracuda. |
| Future | BatchedEncoder adoption | new code | SpectralNautilusBridge (950ms) is a GPU acceleration candidate via BatchedEncoder |
| Future | barracuda::pde | new code | Richards PDE for hydrology/ET₀ if soil moisture modeling is added |
| Future | barracuda::multi_gpu | new code | Multi-GPU interconnect for large tensor operations |

---

## Part 5: Lessons Learned for ToadStool Evolution

1. **Backward compatibility matters**: S79→S86 was nearly seamless. Only the nautilus DriftMonitor API broke — and `is_drifting()` replacing `consecutive_drift` is a cleaner API. Keep this pattern.

2. **Cross-spring provenance is valuable**: When shaders credit their origin (`// neuralSpring V24`), downstream consumers can trace the evolution chain. Consider standardizing provenance comments in all WGSL.

3. **Sub-microsecond hydrology is excellent**: The 5 ET₀ methods in barracuda::stats::hydrology are fast enough for real-time pre-processing. The whole airSpring absorption path works well.

4. **Nautilus brain operations are fast**: NautilusBrain (8.6µs), observe (7.3µs), DriftMonitor (0.5µs) are well-optimized. The SpectralNautilusBridge bottleneck (950ms) is in the ESN training — a BatchedEncoder GPU path could help.

5. **Shader count matters less than coverage**: 692+ WGSL shaders, but the key metric is that 144 ComputeDispatch ops cover the critical paths. Quality over quantity.

6. **Paper queue is fully covered**: All 25 Phase 0++ papers have Python control + barracuda CPU + GPU validator + metalForge coverage. The validation pyramid is complete for the current paper set.

---

## Part 6: Paper Queue Validation Coverage

| Tier | Papers | Coverage |
|------|--------|----------|
| Phase 0++ (011-025) | 15 papers | 15/15 full stack (Py + bC CPU + GPU + metalForge) |
| Phase 0/0+ (001-010) | 10 studies | 10/10 Python + bC; 2 analytical-only (no GPU needed) |
| baseCamp (B-01..B-21) | 6 sub-theses | 6/6 bC + GPU + metalForge dispatch |
| WDM surrogates (nW-01..05) | 5 papers | 5/5 full stack |
| coralForge (nF-01..03) | 3 papers | 3/3 full stack |

---

## Appendix: Validation State

| Metric | V74 (S112) | V75 (S113) |
|--------|-----------|-----------|
| ToadStool HEAD | `2fee1969` (S86) | `2fee1969` (S86) |
| validate_all | 208/208 | **208/208** |
| validate_modern_cross_spring | 57/57 | **68/68** |
| bench_cross_spring_modern | 10/10 | **14/14** |
| lib tests | 861 | 861 |
| clippy warnings | 0 | 0 |
| Cross-spring provenance | partial | **6 springs tracked** |
| Hydrology ET₀ methods validated | 0 | **5** |
| neuralSpring→ToadStool shaders | documented | **15+ tracked, 6 credited** |
