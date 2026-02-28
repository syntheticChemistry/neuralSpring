# neuralSpring → ToadStool/BarraCUDA Handoff V63 — WDM + AlphaFold3 GPU Tensor Validators + Drift Resolution

**Date**: February 28, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 95 — WDM GPU Tensor validators, AlphaFold3 confidence GPU validator, Python drift resolution, full quality sweep
**Supersedes**: V62 (coralForge Rename + Deep Debt Resolution)

---

## Executive Summary

- **4 new BarraCUDA GPU Tensor validators** proving math portability for WDM surrogates (nW-01 transport, nW-03 S(q,ω), nW-05 ESN) and AlphaFold3 confidence heads (nF-03 Phase C: pLDDT, PAE, pDE)
- **201 binaries**, **189/189 validate_all**, **685 lib tests**, **3200+ checks**
- **39/39 Python drift baselines** — zero drift (fixed isomorphic catalog shader names + 4 path resolution fixes)
- **BarraCUDA Tensor API validated** for: `matmul`, `add`, `relu`, `tanh`, `sigmoid`, `argmax_dim` across MLP, ESN, LSTM, and confidence head architectures
- All quality gates green: `cargo fmt` PASS, clippy pedantic+nursery 0 warnings, `cargo doc` 0 warnings, 685 lib + 9 doc-tests PASS

---

## Part 1: WDM Surrogate GPU Tensor Validators

Three new validators prove that BarraCUDA's f32 Tensor API reproduces the CPU f64 WDM surrogate math within documented tolerances.

### validate_barracuda_wdm_transport (nW-01)

GPU MLP forward pass for WDM transport surrogate. Architecture: input normalization → 3-layer MLP (matmul + bias add + ReLU) → output denormalization.

| Tensor Op | Usage | Count |
|-----------|-------|-------|
| `Tensor::from_data` | Weight/bias/input upload | 7 |
| `Tensor::matmul` | Dense layer projections | 3 |
| `Tensor::add` | Bias addition | 3 |
| `Tensor::relu` | Hidden activations | 2 |
| `Tensor::to_vec` | GPU→CPU readback | 1 |

**Tolerance**: `ML_MLP_F32` (1e-2) — f32 GPU vs f64 CPU comparison across matmul chain.

### validate_barracuda_wdm_esn (nW-05)

GPU Echo State Network regime classifier. Architecture: input normalization → 2-step reservoir recurrence (matmul + tanh) → linear readout → argmax.

| Tensor Op | Usage | Count |
|-----------|-------|-------|
| `Tensor::matmul` | Reservoir + readout projections | 5 (with `.clone()` for reuse) |
| `Tensor::add` | Bias addition | 3 |
| `Tensor::tanh` | Reservoir nonlinearity | 2 |
| `Tensor::argmax_dim` | Classification label | 1 |

**Tolerance**: `TENSOR_TRANSCENDENTAL_F32` (1e-3) — accounts for f32 tanh precision.

**Learning for ToadStool**: `Tensor::matmul` consumes `self` (move semantics). ESN recurrence requires `.clone()` before reuse of intermediate tensors. Consider whether a non-consuming `matmul_ref` variant would reduce allocations in recurrent architectures.

### validate_barracuda_wdm_sqw (nW-03)

GPU LSTM reservoir for S(q,ω) peak prediction. Architecture: input normalization → LSTM unroll (4 gates per timestep) → mean/std pooling → linear readout.

| Tensor Op | Usage | Count per Step |
|-----------|-------|----------------|
| `Tensor::matmul` | Gate projections (W_ih, W_hh) | 2 |
| `Tensor::add` | Gate bias | 1 |
| `Tensor::to_vec` | Per-step readback for sigmoid/tanh | 1 |

**Design**: `LstmGpuWeights` struct groups weight/bias slices to satisfy clippy's 7-argument limit. Sigmoid and tanh activations are computed CPU-side after reading back gate pre-activations — BarraCUDA's `sigmoid()` and `tanh()` operate on full tensors, but LSTM gates require per-element splitting (input/forget/cell/output) which doesn't map cleanly to the current Tensor API.

**toadStool action**: Consider `Tensor::lstm_cell` or `Tensor::split` operations for efficient GPU-resident LSTM unrolling without per-step CPU readback. This would eliminate the N×(matmul→readback→sigmoid→upload) round-trip pattern.

---

## Part 2: AlphaFold3 Confidence GPU Validator (nF-03 Phase C)

### validate_barracuda_alphafold3_confidence_gpu

GPU-accelerated confidence head validation for pLDDT, PAE, and pDE.

| Head | GPU Ops | CPU Ops | Why Split |
|------|---------|---------|-----------|
| pLDDT | matmul + add + sigmoid | — | Fully GPU-accelerable |
| PAE | matmul + add (logits) | softmax_rows + expected_distance | `Tensor::softmax()` is global, not row-wise |
| pDE | matmul + add (logits) | softmax_rows + expected_distance | Same as PAE |

**Tolerances**:
- pLDDT: `TENSOR_TRANSCENDENTAL_F32` (1e-3) — sigmoid precision
- PAE/pDE: `ML_MLP_F32 * 2.0` (2e-2) — matmul chain + CPU-side softmax accumulation

**Determinism**: pLDDT is tested for GPU determinism (allocate→compute twice, compare). Confirmed deterministic on RTX 4070.

**toadStool action**: A `Tensor::softmax_dim(axis)` operation (row-wise softmax along a specified dimension) would allow full GPU-resident PAE/pDE computation. Currently `Tensor::softmax()` reduces globally. This is the primary gap blocking full GPU promotion of AlphaFold3 confidence heads.

---

## Part 3: Python Drift Resolution

### Isomorphic Catalog Fix

`control/isomorphic/isomorphic_catalog.py` reported 20% BarraCUDA coverage (failing 70% threshold) because shader name mappings were outdated.

| Before | After |
|--------|-------|
| `attention.wgsl` | `scaled_dot_product_attention_f64.wgsl` |
| `gemv_q4.wgsl` | `gemv_q4_f64.wgsl` |
| `gemv_q8.wgsl` | `gemv_q8_f64.wgsl` |
| ... (20+ entries) | Full `_f64` suffix + accurate shader stems |

Coverage: 20% → 100% after shader name resolution.

### Path Resolution Fixes

4 control scripts hardcoded output paths relative to project root, but `check_drift.sh` runs from `control/`:

| Script | Fix |
|--------|-----|
| `alphafold3_confidence.py` | `Path(__file__).parent / "confidence_baselines.json"` |
| `training_trajectory.py` | `Path(__file__).parent / "baseline_values.json"` |
| `hessian_eigenanalysis.py` | `Path(__file__).parent / "baseline_values.json"` |
| `anderson_multiagent.py` | `Path(__file__).parent / "baseline_values.json"` |

All 39 baselines now pass in both direct invocation and `check_drift.sh` contexts.

---

## Part 4: BarraCUDA API Gaps Identified This Session

| Gap | Impact | Suggested API | Priority |
|-----|--------|---------------|----------|
| `Tensor::softmax_dim(axis)` | PAE/pDE confidence heads stuck on CPU | Row-wise softmax along specified axis | P2 |
| `Tensor::lstm_cell` or `Tensor::split` | LSTM unroll requires per-step CPU readback | GPU-resident gate splitting | P3 |
| `Tensor::matmul` consuming self | ESN/LSTM recurrence needs `.clone()` | Non-consuming `matmul_ref` or `Tensor: Clone` impl | P3 |

---

## Part 5: What neuralSpring Learned (Relevant to ToadStool Evolution)

1. **MLP GPU promotion is straightforward**: The `matmul → add → relu` pattern maps 1:1 to the Tensor API. No API gaps. WDM transport validator proves this for a real scientific workload.

2. **Recurrent models hit the move-semantics wall**: ESN and LSTM both require reusing intermediate tensors across timesteps. The current API forces `.clone()` which allocates. For production recurrent networks, a borrow-based matmul or explicit in-place operations would eliminate these allocations.

3. **LSTM gate splitting is the real bottleneck**: The 4-gate LSTM cell (input/forget/cell/output) requires splitting a concatenated projection output. Without `Tensor::split` or `Tensor::chunk`, this forces a CPU round-trip per timestep. For the WDM S(q,ω) predictor with 10 timesteps, that's 10 readback cycles.

4. **Confidence heads expose the softmax dimensionality gap**: `Tensor::softmax()` is global (reduces entire tensor). PAE/pDE need row-wise softmax (each residue pair independently). This is the same gap that would affect any attention-like head with explicit probability distributions.

5. **f32 GPU vs f64 CPU tolerances are well-understood**: MLP chains: 1e-2. Transcendental (tanh/sigmoid): 1e-3. These are stable across all validators and consistent with the tolerance registry.

---

## Part 6: Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy -- -W clippy::pedantic -W clippy::nursery` | **0 warnings** |
| `cargo doc --no-deps` | **0 warnings** (202 pages) |
| `cargo test` | **685 lib + 9 doc-tests PASS** |
| `check_drift.sh` | **39/39 PASS** (0 drift) |
| `validate_all` entries | **189 binaries** |
| Zero unsafe | Confirmed |
| Zero production mocks | Confirmed |
| SPDX headers | 100% coverage |

---

## Part 7: Current BarraCUDA Tensor API Usage Across All GPU Validators

| Tensor Op | Validators Using It | Domain |
|-----------|-------------------|--------|
| `from_data` | All 4 new + 15 existing | Universal |
| `matmul` | All 4 new + 12 existing | Dense projections |
| `add` | All 4 new + 8 existing | Bias terms |
| `relu` | WDM transport + EOS + existing | MLP activations |
| `tanh` | WDM ESN + SQW + existing | Reservoir / LSTM |
| `sigmoid` | AlphaFold3 confidence + existing | pLDDT, gate activations |
| `argmax_dim` | WDM ESN | Classification |
| `to_vec` | All 4 new + 15 existing | GPU→CPU readback |
| `transpose` | 7 existing | Matrix operations |
| `softmax` | 2 existing (global only) | Attention normalization |

---

## Superseded Handoffs

All in `wateringHole/handoffs/archive/`:

- V62: coralForge Rename + Deep Debt Resolution (Feb 28, S94)
- V61: Deep Debt + Confidence Heads (Feb 28, S92-93)
- V60: Dispatch Parity + Mixed-Hardware (Feb 27, S89)

---

*neuralSpring V63 — Session 95, February 28, 2026. WDM + AlphaFold3 GPU Tensor validators built, Python drift resolved, all quality gates green.*
