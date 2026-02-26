# neuralSpring → ToadStool/BarraCUDA Handoff V46: Deep Debt Evolution

**Date:** February 26, 2026
**From:** neuralSpring (ecoPrimals)
**To:** ToadStool / BarraCUDA team
**Supersedes:** V45 (Session 80 debt audit)
**Type:** Deep debt evolution, tolerance centralization, barracuda rewire
**License:** AGPL-3.0-or-later

## Executive Summary

Session 81 performed a comprehensive deep-debt sweep of the entire neuralSpring
codebase. All inline magic numbers across 21 validation binaries are now replaced
with 129+ named tolerance constants (25 new). The last duplicate math
(`spectral_entropy`) was rewired to `barracuda::stats::shannon_from_frequencies`.
Platform-specific `/proc` reads are gated behind `#[cfg(target_os = "linux")]`.
Seven Python training scripts gained full PyTorch seeding for deterministic
baselines. All quality gates pass: `cargo fmt`, `cargo clippy -D warnings`
(pedantic+nursery), `cargo test` (604+9+9), `cargo doc`. Zero TODO, zero
unsafe, zero inline magic numbers.

37 files changed, +334 / -51 lines.

---

## Part 1: What Changed in Session 81

### 1.1 Tolerance Centralization (25 new constants)

| Category | New Constants |
|----------|---------------|
| Spectral analysis | `LEVEL_SPACING_GOE_SLACK`, `SPECTRAL_IPR_COMPARISON_SLACK`, `NUMERICAL_DISTINCTNESS`, `GATE_DISORDER_COMPARISON`, `SPECTRAL_RADIUS_SWEEP_SLACK` |
| Population genetics | `FST_IDENTICAL_POP_TOL`, `FST_ESTIMATOR_AGREEMENT` |
| Game theory | `GAME_DEFECTION_UPPER`, `GAME_QS_COOPERATION_MIN`, `GAME_QS_VARIANCE_MAX` |
| Numerical guards | `RELATIVE_ERROR_FLOOR`, `ODE_STEADY_STATE_SLACK` |
| Quantization | `QUANT_Q8_GEMV_ERROR`, `QUANT_Q4_GEMV_ERROR`, `QUANT_SIGN_AGREEMENT` |
| GPU commutator | `GPU_COMMUTATOR_NEAR_ZERO_F64`, `GPU_COMMUTATOR_RESIDUAL_F64` |
| Hardware dispatch | `BRIDGE_COST_MIN_US`, `BRIDGE_COST_MAX_US`, `BRIDGE_CHAIN_OVERHEAD_MAX`, `BRIDGE_PROBE_MIN_US`, `TRANSFER_1MB_MIN_US`, `TRANSFER_1MB_MAX_US`, `DISPATCH_COST_RATIO_MIN`, `DISPATCH_COST_RATIO_MAX` |

Registry categories expanded: `training_quantized`, `hardware`.

### 1.2 Binaries Updated (21 files)

| Binary | Change |
|--------|--------|
| `validate_weight_spectral` | 4 magic numbers → named constants |
| `validate_information_flow` | 2 magic numbers → named constants |
| `validate_cross_spring_evolution` | 2 FST thresholds → named constants |
| `validate_barracuda_game` | 3 game-theory thresholds → named constants |
| `validate_barracuda_wdm_eos` | 2 division guards → `RELATIVE_ERROR_FLOOR` |
| `validate_quantized` | 3 quantization thresholds → named constants |
| `validate_gpu_stateful_pipeline` | `0.5` → `ODE_STEADY_STATE_SLACK as f32` |
| `validate_gpu_rk4` | Same f32 cast fix |
| `validate_barracuda_gpu_spectral` | `1e-12` → `GPU_COMMUTATOR_NEAR_ZERO_F64` |
| `validate_barracuda_spectral` | `0.5` → `GPU_COMMUTATOR_RESIDUAL_F64` |
| `validate_barracuda_gpu_{signal,regulatory,introgression}` | `1e-5_f32` → `GPU_BOUNDS_SLACK_F32 as f32` |
| `validate_barracuda_gpu_{meta_pop,eco}` | `-1e-6` → `VARIANCE_FLOOR` |
| `validate_barracuda_gpu_modes` | `1e-5` → `GPU_BOUNDS_SLACK_F32` |
| `validate_{regulatory_network,agent_coordination,barracuda_lenet}` | `1e-10` → `RELATIVE_ERROR_FLOOR` |
| `validate_{cross_system_dispatch,compute_dispatch,metalforge_pcie}` | Hardware cost literals → named constants |
| `validate_barracuda_pangenome` | `16.92` → `CHI2_CRITICAL_DF9_P05` |

### 1.3 BarraCUDA Rewire

`weight_spectral::spectral_entropy` → `barracuda::stats::shannon_from_frequencies`.
This was the last duplicate math. **39 functions + 6 shader sources** now
rewired to upstream barracuda.

### 1.4 Cross-Platform Probe

`metalForge/forge/src/probe.rs`: `/proc/cpuinfo` and `/proc/meminfo` reads
gated behind `#[cfg(target_os = "linux")]`. Non-Linux fallback returns safe
defaults. No more hard dependency on Linux filesystem.

### 1.5 Python Baseline Determinism

Seven PyTorch training scripts gained:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Scripts: surrogate, quantized, transfer, sequence, lstm_weather, pinn, deeponet.

### 1.6 Clippy Fixes

- `validation.rs:477`: backticked `check_abs` in doc comment
- `wdm_surrogate.rs:374`: `.err().expect()` → `.expect_err()` with allow

---

## Part 2: BarraCUDA Primitive Usage (updated)

### Usage by category

| Category | Import Sites | Key Modules |
|----------|:------------:|-------------|
| Device/GPU | 50+ | `WgpuDevice`, `GpuDriverProfile`, `Fp64Strategy` |
| Stats | 15+ | `r_squared`, `rmse`, `mae`, `nse`, `shannon_from_frequencies`, `hill`, `pearson_correlation`, `variance`, `fit_linear`, `empirical_spectral_density`, `marchenko_pastur_bounds` |
| Linalg | 10+ | `eigh_f64`, `eigh_householder_qr`, `graph_laplacian`, `belief_propagation_chain`, `effective_rank` |
| Numerical | 5+ | `rk45_solve`, `numerical_hessian` |
| Special | 5+ | `chi_squared_statistic`, `gamma`, `erf`, `bessel` |
| Tensor | 40+ | Full Tensor API (matmul, relu, softmax, etc.) |
| Ops (bio) | 30+ | `BatchFitnessGpu`, `PairwiseHammingGpu`, `HmmBatchForwardF64`, `GillespieGpu`, 15+ more |
| Ops (reduce) | 10+ | `FusedMapReduceF64`, `VarianceReduceF64`, `CorrelationF64`, `SumReduceF64` |
| Ops (FFT) | 4 | `Fft1D`, `Ifft1D`, `Fft1DF64`, `Rfft` |
| Dispatch | 10+ | `matmul_dispatch`, `softmax_dispatch`, `gelu_dispatch`, `l2_distance_dispatch` |
| Pipeline | 3 | `ReduceScalarPipeline`, `StatefulPipeline`, `KernelDispatch` |

**Total**: 90+ import sites across 150+ .rs files, 16+ barracuda submodules.

### Intentional CPU references (not duplicates)

| Location | Function | Why kept |
|----------|----------|----------|
| `primitives::sigmoid` | CPU scalar sigmoid | No barracuda CPU scalar equivalent |
| `primitives::rk4_step` | Fixed-step RK4 | Complementary to adaptive `rk45_solve` |
| `transformer::{softmax, gelu}` | CPU reference | Independent Python baseline validation |
| `spectral_commutativity::{mat_mul, frobenius_norm}` | CPU reference | GPU validation reference |
| `cpu_fallback::variance` | Population (÷N) | Intentional convention difference vs barracuda sample (÷(N-1)) |

---

## Part 3: Evolution Recommendations for ToadStool

### 3.1 High Priority — Absorption Candidates

| Item | What neuralSpring Has | What ToadStool Should Absorb |
|------|----------------------|------------------------------|
| `validate_tensor_unary` | GPU tensor op validation harness | Move to `barracuda::validation` |
| `validate_tensor_reduction` | GPU scalar reduction harness | Move to `barracuda::validation` |
| `SimpleMLP` pattern | JSON weights → layer forward | `barracuda::nn::SimpleMLP` |
| 9 sovereign folding shaders | `layer_norm_f64`, `gelu_f64`, `sigmoid_f64`, `sdpa_scores_f64`, etc. | Absorb into `barracuda::ops` |
| Tolerance derivation pattern | Every constant has `// Derivation:` doc | Adopt across springs |

### 3.2 Medium Priority — API Gaps

| Gap | Current Workaround | Proposed Upstream |
|-----|-------------------|-------------------|
| `variance(data, ddof)` | Two separate functions | Single API with ddof parameter |
| Fused MLP dispatch | N encoder submissions per forward | `TensorSession::fused_mlp` |
| `stats::hill` with amplitude | Callers wrap `amplitude * hill(x, k, n)` | `hill(x, k, n, amplitude)` |
| f64 SDPA pipeline | 3 shader dispatches | Single pipeline submission |
| `harness.check_abs_result()` | Manual unwrap + check | Result-aware check method |

### 3.3 Low Priority

| Gap | Notes |
|-----|-------|
| `l2_distance_cpu` shortcut | `l2_distance_dispatch(a, b, None)` works but verbose |
| Cross-spring tolerance registry | Convention for named tolerances across springs |

---

## Part 4: Validation Results

### Quality Gates (Session 81)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | 0 warnings (pedantic+nursery) |
| `cargo test` | 604 lib + 9 integration + 9 doc-test PASS |
| `cargo doc --no-deps` | clean (166 pages) |
| Python baselines | 206/206 PASS (last validated 2026-02-21) |
| Coverage | 93.5% (above 90% gate) |

### Three-Tier Validation Matrix

| Paper/Domain | CPU (Rust) | BarraCUDA CPU | GPU Tensor | GPU Shader | Cross-Dispatch | Pipeline |
|:------------|:----------:|:-------------:|:----------:|:----------:|:--------------:|:--------:|
| 011 CD Evolution | 11/11 | 7/7 | — | batch_fitness | 8/8 | ecology |
| 012 MODES | 9/9 | 7/7 | — | pairwise_l2 | 8/8 | modes |
| 013 Eco Dynamics | 7/7 | 6/6 | 6/6 | batch_fitness | 8/8 | ecology |
| 014 Directed Evo | 8/8 | 7/7 | — | multi_obj | 8/8 | directed |
| 015 Swarm | 11/11 | 10/10 | 6/6 | swarm_nn | 8/8 | fitness |
| 016 HMM | 10/10 | 14/14 | 5/5 | hmm_forward | 8/8 | hmm |
| 017 SATé | 8/8 | 6/6 | — | pairwise_hamming | 8/8 | genomics |
| 018 Introgression | 8/8 | 11/11 | 5/5 | hmm_forward | 12/12 | hmm |
| 019 Game Theory | 8/8 | 5/5 | 6/6 | spatial_payoff | — | eco |
| 020 Regulatory | 7/7 | 6/6 | 5/5 | rk4 | — | signal |
| 021 Signal | 8/8 | 14/14 | 6/6 | hill_gate | — | signal |
| 022 Spectral | 8/8 | 10/10 | 10/10 | batch_ipr | — | spectral |
| 023 Anderson | 8/8 | 7/7 | 7/7 | batch_ipr | — | spectral |
| 024 Pangenome | 8/8 | 12/12 | — | pairwise_jaccard | 8/8 | genomics |
| 025 Meta-pop | 8/8 | 12/12 | 5/5 | locus_variance | 12/12 | meta_pop |
| PINN (001) | 8/8 | 14/14 | — | — | — | — |
| DeepONet (002) | 7/7 | 9/9 | — | — | — | — |
| LeNet-5 (003) | 5/5 | 5/5 | — | — | — | — |
| LSTM (004) | 5/5 | 6/6 | — | — | — | — |
| Quantized (005) | 6/6 | 15/15 | — | — | — | — |
| baseCamp (5) | 114/114 | — | 14/14 | — | 16/16 | — |
| **Totals** | **2040+** | **272+** | **98+** | **108+** | **49+** | **77+** |

### Hardware Coverage

| Hardware | Driver | Tests | Status |
|----------|--------|:-----:|:------:|
| RTX 4070 (Ada) | Vulkan 1.3 | 604 lib + 166 binaries | PASS |
| TITAN V (Volta/NVK) | NVK | GPU dispatch subset | PASS (bit-identical) |
| llvmpipe (software) | Vulkan 1.0 | CPU fallback | PASS |

---

## Part 5: Data Provenance

All validation datasets are from public repositories with documented accession:

| Domain | Source | License |
|--------|--------|---------|
| Weather (ERA5) | ECMWF Copernicus | Open |
| MNIST | LeCun et al. | Public domain |
| Open-Meteo | Open-Meteo API | CC-BY 4.0 |
| NK Landscapes | Synthetic (seed=42) | — |
| WDM EOS | MESA/OPAL tables | Academic open |

See `specs/DATA_PROVENANCE.md` for full inventory.

---

## Part 6: Cross-Spring Lessons

### What neuralSpring validated for barracuda

1. **Tolerance pattern**: 129+ named constants with `// Derivation:` docs — proved
   the pattern scales. Recommend adoption across springs.
2. **`ValidationHarness`**: 166 binaries use it. Mature enough for `barracuda::validation`.
3. **Dispatch cost model**: `metalForge` hardware probe → `Dispatcher::mixed_dispatch()`
   validates CPU↔GPU cost decisions. Hardware tolerance constants now formalize
   the bounds.
4. **PyTorch seeding**: `torch.manual_seed` + `cudnn.deterministic` is required for
   bitwise reproducibility. All springs using PyTorch should adopt.
5. **Cross-platform probing**: `/proc` reads gated behind `#[cfg(target_os)]` — pattern
   for all `metalForge` substrate probes.

### What neuralSpring still needs from barracuda

1. `SimpleMLP` for WDM EOS surrogate (JSON → forward pass)
2. `variance(data, ddof)` for the population-vs-sample convention split
3. Fused MLP dispatch for encoder efficiency
4. 9 sovereign folding shaders absorbed

---

## Part 7: Verification Commands

```bash
# Full quality gate
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo doc --no-deps

# Python baselines
pip install -r control/requirements.txt && bash scripts/run_all_baselines.sh

# Drift detection
bash control/check_drift.sh

# All validation binaries
cargo run --release --bin validate_all
```

---

## Part 8: Next Steps

### For ToadStool

- [ ] Absorb `validate_tensor_unary` / `validate_tensor_reduction`
- [ ] Implement `barracuda::nn::SimpleMLP` (JSON weights + forward)
- [ ] Add `variance(data, ddof)` parameter
- [ ] Absorb 9 sovereign folding f64 shaders
- [ ] Review tolerance derivation pattern for cross-spring adoption

### For neuralSpring

- [x] Zero inline magic numbers (25 new named constants)
- [x] Zero duplicate math (`spectral_entropy` rewired)
- [x] Cross-platform probe (Linux-gated)
- [x] Full PyTorch seeding (7 scripts)
- [x] CHANGELOG.md created
- [ ] Monitor approaching-1000-line files (max 911, all compliant)
- [ ] Paper queue: confirm WDM surrogate controls at all 3 tiers

---

*Generated: February 26, 2026 | Session 81 | 37 files changed, +334/-51 lines*
