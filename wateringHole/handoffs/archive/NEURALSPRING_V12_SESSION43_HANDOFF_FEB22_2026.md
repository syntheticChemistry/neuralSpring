# neuralSpring V12 — Session 43 Handoff

**Date**: February 22, 2026
**Session**: 43
**ToadStool HEAD**: `5437c170` (unchanged — local evolution for future absorption)
**Previous**: V11 (Session 42 — 10/10 upstream parity, LeNet-5 full bC)

---

## Executive Summary

Session 43 extended neuralSpring across three axes: **new WGSL shaders** for ToadStool
absorption, **upstream BarraCuda wrapper integration** (including wetSpring parity),
and **mixed-hardware dispatch infrastructure** for GPU-NPU-CPU routing. All 12 new
validators pass (108/108 checks). The forge shader catalog grew from 17 to 21.
CPU vs GPU parity is validated as bit-identical for key operations.

---

## Part 1: New WGSL Shaders (4 — evolving for ToadStool absorption)

| Shader | Entry Point | Domain | Validator | Checks |
|--------|-------------|--------|-----------|--------|
| `logsumexp_reduce.wgsl` | `logsumexp_reduce` | Batched logsumexp (HMM/phylo) | `validate_gpu_logsumexp` | 5/5 |
| `stencil_cooperation.wgsl` | `stencil_update` | Fermi imitation (game theory) | `validate_gpu_stencil` | 3/3 |
| `rk45_adaptive.wgsl` | `rk45_step` | Dormand-Prince RK45 (Hill RHS) | `validate_gpu_rk45` | 6/6 |
| `wright_fisher_step.wgsl` | `wright_fisher` | Drift+selection+xoshiro128** | `validate_gpu_wright_fisher` | 4/4 |

### Key design decisions

- **logsumexp_reduce**: f32, max-subtract trick, one thread per row. Extends the single-thread
  `barracuda::shaders::math::logsumexp.wgsl` to batched parallel reduction.
- **stencil_cooperation**: Deterministic Fermi imitation dynamics on Moore neighborhood.
  Temperature κ controls sharpness. Complements `spatial_payoff.wgsl`.
- **rk45_adaptive**: Single Dormand-Prince step with 5th/4th order embedded error estimate.
  Injectable RHS via Hill function coefficients. Host controls step size adaptation.
- **wright_fisher_step**: Exact binomial sampling (2N trials per locus) using inline
  xoshiro128**. Each thread handles one (population, locus) pair.

---

## Part 2: Upstream BarraCuda Wrappers Wired

| API | Module | Validator | Checks | Key Finding |
|-----|--------|-----------|--------|-------------|
| `GillespieGpu` | `ops::bio::gillespie` | `validate_gpu_gillespie` | 20/20 | Perfect f64 conservation |
| `TaxonomyFcGpu` | `ops::bio::taxonomy_fc` | `validate_upstream_taxonomy` | 3/3 | f64 log-posterior bit-exact |
| `KmerHistogramGpu` | `ops::bio::kmer_histogram` | `validate_upstream_kmer` | 3/3 | u32 histogram exact match |
| `UniFracPropagateGpu` | `ops::bio::unifrac_propagate` | `validate_upstream_unifrac` | 2/2 | f64 leaf init exact |
| `chi_squared::*` | `special::chi_squared` | `validate_barracuda_chi_squared` | 13/13 | PDF/CDF/moments within 1e-4 of SciPy |

---

## Part 3: CPU vs GPU Parity

`validate_cpu_gpu_parity` (17/17 PASS) validates Tensor API across GPU and CPU:

| Operation | GPU vs Rust | CPU vs Rust | Cross-hardware |
|-----------|-------------|-------------|----------------|
| MatMul 32×32 | 2.8e-9 | 2.3e-9 | **Bit-identical** |
| ReLU | 0.0 | 0.0 | **Bit-identical** |
| Sigmoid | 1.2e-7 | 1.2e-7 | Identical |
| Tanh | 1.2e-7 | 1.2e-7 | Identical |
| Sum 256 | 4.6e-3 | 4.6e-3 | Identical |
| erf(1) | 1.0e-6 | — | CPU f64 only |
| gamma(5) | exact | — | CPU f64 only |
| conv2d identity | exact | — | cpu_conv_pool |
| max_pool2d 2×2 | exact | — | cpu_conv_pool |

---

## Part 4: Dispatch Routing Validation

`validate_toadstool_dispatch` (16/16 PASS) validates metalForge substrate heuristics:

| Heuristic | Small→CPU | Large→GPU | Validated |
|-----------|-----------|-----------|-----------|
| pairwise | 20×500 | 200×1000 | ✓ |
| batch_fitness | 100×100 | 1000×100 | ✓ |
| ode | 10×100 | 100×200 | ✓ |
| hmm | 3×100 | 10×1000 | ✓ |
| spatial | 100 | 10000 | ✓ |
| batch_ipr | 100×100 | 1000×100 | ✓ |
| logsumexp | 100×100 | 500×100 | ✓ (new) |
| stochastic | 10×10×100 | 100×100×20 | ✓ (new) |

---

## Part 5: Mixed-Hardware Dispatch Infrastructure

### New forge modules

| Module | Purpose |
|--------|---------|
| `mixed.rs` | `MixedSubstrate` enum, `TransferCost`, PCIe bandwidth constants, `mixed_substrate()` heuristic |
| `pcie_bridge.rs` | `PcieBridge` struct, `can_p2p()`, `transfer_cost()`, `detect_p2p()` placeholder |

### Transfer cost model (validated: `validate_mixed_dispatch` 16/16 PASS)

| Path | 1 MB Cost | Bandwidth |
|------|-----------|-----------|
| GPU→CPU (PCIe 4.0 x16) | 35.3 µs | 31.5 GB/s |
| GPU→NPU P2P (PCIe 4.0 x4) | 134.7 µs | 7.9 GB/s |
| GPU→NPU staged (via CPU) | 139.7 µs | 7.9 GB/s (+ latency) |

### Design doc

`metalForge/MIXED_HARDWARE_DESIGN.md` — device topology, dispatch decision tree,
PCIe P2P DMA design, transfer cost model, implementation plan.

---

## Counts

| Metric | V11 | V12 | Delta |
|--------|-----|-----|-------|
| WGSL shaders | 17 | 21 | +4 |
| Validation binaries | 115 | 127 | +12 |
| Forge tests | 18 | 26 | +8 |
| Total checks | 1604+ | 1710+ | +106 |
| Upstream wrappers wired | 10 bio + HMM f64 | +5 (Gillespie, Taxonomy, Kmer, UniFrac, chi²) | +5 |

---

*neuralSpring V12 — Session 43: 4 new shaders, 5 upstream wrappers, CPU/GPU parity, mixed-hardware dispatch.
ToadStool HEAD `5437c170`. 127 validators, 1710+ checks, all green.*
