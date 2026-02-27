# neuralSpring → ToadStool/BarraCUDA Handoff V57

## Modern Rewire & Cross-Spring GPU Evolution

| Field | Value |
|-------|-------|
| Date | February 27, 2026 |
| Session | S88+ (Experiment 064) |
| ToadStool pin | `e96576ee` |
| neuralSpring | 173 binaries, 172/173 PASS (1 pre-existing WDM) |
| Lib tests | 668/668 PASS |
| Total checks | 3,000+ |
| Previous handoff | V56 (ToadStool `e96576ee` upstream sync) |

---

## What Changed (V56 → V57)

### 3 High-Impact GPU Rewires

| Local (pre-V57) | Upstream (post-V57) | Impact |
|-----------------|---------------------|--------|
| `pairwise_l2_matrix_gpu` — O(n²) loop calling `l2_distance_gpu` per pair | `PairwiseL2Gpu::dispatch` — single GPU dispatch | **Eliminates n*(n-1)/2 round-trips** |
| `geographic_distance_matrix_gpu` — O(n²) loop calling `l2_distance_gpu` per pair | `PairwiseL2Gpu` via `pairwise_l2_matrix_gpu` | **Same benefit, 2D→flat→expand** |
| `disorder_sweep_gpu` — CPU loop computing Σ|ψ|⁴ per eigenvector | `BatchIprGpu::dispatch` on GPU | **IPR stays GPU-resident after eigensolve** |

### New Benchmark Binary

- `bench_modern_rewire` — 23/23 PASS
- Benchmarks all 3 rewired functions + modern ToadStool APIs
- Tracks cross-spring provenance for every op

---

## Cross-Spring Shader Evolution Provenance

```
hotSpring (precision physics)    → DF64 core, eigensolve, Welford variance, logsumexp
wetSpring (bioinformatics)       → Shannon, Simpson, HMM, diversity fusion, Bray-Curtis
neuralSpring (ML/neuroevolution) → pairwise L2, IPR, batch fitness, swarm NN, MHA
airSpring (atmospheric)          → RMSE, R², NSE, fit_linear, moving_window
groundSpring (hydrology)         → multinomial sampling, MC propagation, ET₀
```

### How Springs Benefit From Each Other

| Flow | Example |
|------|---------|
| hotSpring → neuralSpring | Jacobi eigensolve (HFB) → Anderson localization analysis |
| wetSpring → neuralSpring | Shannon/Simpson diversity → eco_dynamics module |
| neuralSpring → hotSpring | Pairwise L2 (novelty search) → nuclear wavefunction distance |
| neuralSpring → wetSpring | Batch fitness → genomic selection pressure |
| hotSpring precision → all | DF64 core + Welford variance → all springs get f64 accuracy |
| wetSpring bio → all | Diversity fusion → any spring needing Shannon+Simpson+Pielou |

---

## Benchmarked APIs (RTX 4070, release)

| API | Size | µs/iter | Provenance |
|-----|------|---------|------------|
| `PairwiseL2Gpu` (rewired) | 200×50 | ~1,800 | nS MODES → ToadStool S52 |
| `geographic_distance_gpu` (rewired) | 100 coords | ~2,000 | nS MetaPop → ToadStool S52 |
| `disorder_sweep_gpu` (rewired) | 20×16 | ~13,000 | nS Anderson → ToadStool S52+S56 |
| `LogSumExp` f64 | 10,000 | ~7,000 | hS HMM → ToadStool S64 |
| `PairwiseDistance` L2 | 5,000×32 | ~500 | nS MODES → ToadStool S52 |
| `BatchedEighGpu` | 50×24 | ~21,000 | hS HFB → ToadStool S56 |
| `BatchIprGpu` | 500×128 | ~2,800 | nS Anderson → ToadStool S52 |
| `DiversityFusionGpu` | 64×200 | ~3,500 | wS diversity → ToadStool S64 |
| `Dispatcher::variance` f64 | 50k | ~11,000 | hS Welford → ToadStool S62 |
| `Dispatcher::pearson` f64 | 50k | ~9,600 | wS+hS → ToadStool S64 |
| `Dispatcher::shannon` f64 | 50k | ~3,400 | wS fused → ToadStool S64 |
| `Dispatcher::mat_mul` | 200×200 | ~1,800 | nS → ToadStool |

---

## What ToadStool Should Absorb Next

1. **Nothing immediate** — all 3 rewires already use upstream APIs
2. **Phase 4 WGSL shaders** (22/22 PASS, Exp-062) — HMM backward/Viterbi, matrix correlation, linear regression via direct metalForge dispatch, candidates for typed wrapper absorption
3. **Sovereign folding df64 shaders** (15 shaders) — Evoformer primitives (layer_norm, GELU, sigmoid, SDPA×3, triangle×3, backbone, IPA, MSA, OPM, torsion) using `compile_shader_df64`

---

## Validation Matrix

| Category | Checks | Status |
|----------|--------|--------|
| Lib tests | 668 | 668/668 PASS |
| validate_all binaries | 173 | 172/173 PASS |
| Rewire bench | 23 | 23/23 PASS |
| Cross-spring evolution | 52 | 52/52 PASS |
| Upstream vs local | 10 | 10/10 PASS |
| Pre-existing WDM failure | 1 | Known (γ damping) |

---

## Files Changed

- `src/gpu_ops/bio.rs` — `pairwise_l2_matrix_gpu` rewired to `PairwiseL2Gpu`
- `src/gpu_ops/population.rs` — `geographic_distance_matrix_gpu` rewired via above
- `src/gpu_ops/eigensolver.rs` — `disorder_sweep_gpu` IPR rewired to `BatchIprGpu`
- `src/bin/bench_modern_rewire.rs` — new benchmark binary (23/23)
- `src/bin/validate_all.rs` — added `bench_modern_rewire` (173 total)
- `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` — S88+ rewire section added
