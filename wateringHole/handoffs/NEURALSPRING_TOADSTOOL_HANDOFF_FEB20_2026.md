# neuralSpring → ToadStool: Phase 0++ Complete — 23 Papers, New Primitives, GPU Promotion Map

**Date:** 2026-02-20
**From:** neuralSpring (ML / isomorphic learning / scholarly reproduction Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** `NEURALSPRING_TOADSTOOL_HANDOFF_FEB19_2026.md` (11 shortcomings — all still pending)

---

## Executive Summary

neuralSpring has completed the entire paper review queue: **23 experiments**
across 5 scientific disciplines (ML, evolutionary computation, phylogenetics,
game theory/regulatory biology, spectral theory). Every experiment has both
a Python baseline and a Rust validation binary. Grand total: **599/599 PASS**
(190 Python + 167 Rust native + 242 BarraCUDA).

The Phase 0++ buildout (13 new papers since Feb 19) introduces **7 new
algorithmic patterns** that map directly to BarraCUDA GPU primitives. This
handoff catalogs what was built, what the ToadStool team can absorb, and the
priority order for GPU promotion.

**What changed since Feb 19:**
- 7 new Python experiments (58 new Python checks)
- 7 new Rust library modules (7 new crate modules, ~2000 lines)
- 7 new Rust validation binaries (57 new native checks)
- Deterministic Xoshiro256** PRNG for reproducible stochastic algorithms
- New algorithmic patterns: ODE integration, eigendecomposition, distance matrices, spatial cooperation

**The 11 BarraCUDA shortcomings from Feb 19 remain unresolved.** They are
documented in `specs/TOADSTOOL_HANDOFF.md` and the recommended absorption
order is unchanged. This handoff focuses on **new content** only.

---

## 1. New Modules and Their BarraCUDA Relevance

### 1.1 Evolutionary Computation (Dolson — Papers 011–015)

| Module | Paper | Core Algorithm | BarraCUDA Primitive |
|--------|-------|----------------|---------------------|
| `counterdiabatic.rs` | 011 | NK landscape + Wright-Fisher + CD schedule | `gemm_f64` (fitness eval), `softmax.wgsl` (Boltzmann) |
| `modes.rs` | 012 | Change/novelty/complexity/ecology metrics | `reduce_sum`, `elementwise` |
| `eco_dynamics.rs` | 013 | Multi-niche EA, Shannon diversity | batch GEMM (fitness), `reduce_sum` |
| `directed_evolution.rs` | 014 | 5 selection algorithms (lexicase, tournament, etc.) | batch GEMM, `reduce_max`, `argmax` |
| `swarm_robotics.rs` | 015 | Heterogeneous controller evolution | batch GEMM (3 controller types), `reduce_sum` |

**Key insight for BarraCUDA**: All 5 Dolson papers evaluate fitness as a
matrix-vector product or a small GEMM. A **batch fitness evaluation kernel**
— dispatching population × genome × niche fitness in one GPU pass — would
accelerate all of them. The population sizes are 100–500, genome lengths
8–50, and niche counts 1–5. These are small enough for shared-memory
tiling but large enough to benefit from GPU parallelism when the outer
evolutionary loop runs 100–500 generations.

### 1.2 Phylogenetics & Alignment (Liu — Papers 016–018)

| Module | Paper | Core Algorithm | BarraCUDA Primitive |
|--------|-------|----------------|---------------------|
| `hmm.rs` | 016 | Forward/backward/Viterbi/posterior | `gemm_f64` chain (T × α), log-domain numerics |
| `sate_alignment.rs` | 017 | Pairwise distance + neighbor-joining + alignment | `gemm_f64` (N×N distance matrix), `reduce_min` |
| `introgression.rs` | 018 | PhyloNet-HMM + likelihood ratio test | `gemm_f64` chain (reuses `hmm.rs`), log-sum-exp |

**Key insight for BarraCUDA**: The HMM forward algorithm is a **matrix chain
multiplication** in log-domain: `log(α_t) = log(A^T · α_{t-1}) + log(B · o_t)`.
For T=500 loci and N=2 states, this is 500 sequential 2×2 matmuls. The
bottleneck is latency (sequential), not throughput. A **batched log-domain
GEMM chain** primitive — or better, a fused HMM forward kernel — would
directly port Papers 016–018 to GPU.

The pairwise distance matrix (Paper 017) is an O(N²) computation that maps
directly to a GPU kernel: each thread computes the Hamming/Jukes-Cantor
distance between one pair of sequences.

### 1.3 Game Theory & Regulatory Biology (Waters — Papers 019–021)

| Module | Paper | Core Algorithm | BarraCUDA Primitive |
|--------|-------|----------------|---------------------|
| `game_theory.rs` | 019 | Prisoner's Dilemma, Snowdrift, replicator dynamics, QS | `gemm_f64` (payoff), `softmax.wgsl` (replicator) |
| `regulatory_network.rs` | 020 | GRN ODE with Hill activation/repression | `elementwise` (Hill fn), RK4 integration |
| `signal_integration.rs` | 021 | Two-input Hill function (biological AND gate) | `elementwise` (multiplicative attention) |

**Key insight for BarraCUDA**: Papers 020–021 both use **ODE integration
with Hill function kinetics**. The ODE right-hand side is purely elementwise
(Hill activation/repression + linear degradation). A **GPU-parallel RK4
integrator** for systems of ODEs would accelerate both papers, and would
also benefit any future multi-agent simulation where each agent has internal
dynamics (e.g., primal metabolic networks).

The spatial cooperation model (Paper 019) uses a **1D stencil** to compute
neighborhood averages. This maps to a GPU stencil convolution kernel.

### 1.4 Spectral Theory (Kachkovskiy — Papers 022–023)

| Module | Paper | Core Algorithm | BarraCUDA Primitive |
|--------|-------|----------------|---------------------|
| `spectral_commutativity.rs` | 022 | Commutator, distance to normal, Frobenius norm | `gemm_f64`, `reduce_sum` |
| `anderson_localization.rs` | 023 | Tridiagonal Hamiltonian, Jacobi eigensolver, IPR | `tridiag`, `eigh_f64`, `reduce_sum` |

**Key insight for BarraCUDA**: Paper 023 uses a **symmetric tridiagonal
eigensolver** (Jacobi iteration). BarraCUDA already has `tridiag` (Thomas
algorithm for tridiagonal systems) and `eigh_f64` (general symmetric
eigenvalue). A **specialized tridiagonal eigensolver** (Householder reduction
→ bisection → inverse iteration) would be faster for the Aubry-André model's
N×N matrices where N=50–200. Paper 022's `gemm_f64` for commutator
computation ([A,B] = AB - BA) is directly served by existing primitives.

---

## 2. New Algorithmic Patterns for GPU Promotion

| Pattern | Papers | Computation | Proposed BarraCUDA Primitive |
|---------|--------|-------------|------------------------------|
| **Batch fitness eval** | 011–015 | Population × genome fitness | `batch_gemv` or `batch_gemm` with population dim |
| **HMM forward chain** | 016–018 | Sequential log-domain matmul | `hmm_forward_log.wgsl` (fused log-sum-exp chain) |
| **Pairwise distance** | 017 | O(N²) Hamming/JC distance | `pairwise_distance.wgsl` (one thread per pair) |
| **ODE integration (RK4)** | 020–021 | Parallel multi-system ODE | `rk4_batch.wgsl` (elementwise RHS + 4 stages) |
| **Spatial stencil** | 019 | 1D/2D neighborhood average | `stencil_1d.wgsl` (or reuse conv1d) |
| **Tridiag eigensolver** | 022–023 | Symmetric tridiag eigenvalues | `tridiag_eigh.wgsl` (bisection + inverse iteration) |
| **Replicator dynamics** | 019 | Softmax-like frequency update | Existing `softmax.wgsl` + `elementwise` |

### Recommended Absorption Order

1. **Batch GEMM/GEMV** — serves all 5 Dolson papers plus any future EA work.
   Already partially available via matmul; needs a population dimension.
2. **Pairwise distance kernel** — simple, high-value, directly maps to GPU.
   One thread per pair, no synchronization needed.
3. **GPU-parallel RK4** — serves Papers 020–021 and any future ODE work.
   Elementwise RHS + 4 fused Euler steps per kernel launch.
4. **Fused HMM forward** — serves Papers 016–018 and any future HMM work.
   Log-domain matmul chain with scaling, minimizes kernel launches.
5. **Tridiagonal eigensolver** — serves Papers 022–023. Specialized algorithm
   is much faster than general `eigh_f64` for tridiagonal structure.
6. **Spatial stencil** — serves Paper 019. Could reuse existing conv1d.

---

## 3. Deterministic PRNG for Stochastic Algorithms

The Phase 0++ papers required stochastic algorithms (mutation, sampling,
random initialization). We implemented a deterministic **Xoshiro256\*\***
PRNG in `src/rng.rs` (189 lines) providing:

- Uniform f64 in [0, 1)
- Normal distribution (Box-Muller transform)
- Uniform usize in [0, n)
- Categorical sampling (weighted)
- Multinomial sampling
- Choose k distinct indices
- Random permutation
- Bernoulli mask

This is seeded from a SplitMix64 state initializer for reproducibility.
**All stochastic algorithms use seed=42** for deterministic validation.

The PRNG is pure Rust with no external dependencies and no `unsafe` code.
If BarraCUDA needs a GPU-side PRNG for parallel population evaluation,
Xoshiro256** has excellent parallelization properties (each thread can
use `jump()` to get independent streams from a single seed).

---

## 4. Validation Infrastructure Additions

### New Tolerances

| Constant | Value | Justification |
|----------|-------|---------------|
| `CD_COMPARABLE_DIST` | 0.01 | L1 distance in 32-dim simplex |
| `ADIABATIC_KL_GAP` | 0.05 | KL nats for Fisher information discretization |
| `HMM_POSTERIOR_SUM` | 1e-8 | Forward-backward scaling over T≤5000 |
| `QS_VARIANCE_MAX` | 0.05 | Late-stage cooperation stability |

### New Provenance Records

7 new `BaselineProvenance` constants in `src/provenance.rs`, one per paper
(015, 017, 018, 020, 021, 022, 023). Each traces to a specific Python script,
git commit, and seed.

### CI Updates

7 new validation steps in `.github/workflows/rust.yml` under the
`validate-native` job. Total CI validation: 26 binaries.

---

## 5. BarraCUDA Usage Inventory

Complete catalog of how neuralSpring uses BarraCUDA, organized by usage pattern:

### Direct `barracuda::*` Calls (10 validation binaries, 242 checks)

| BarraCUDA Domain | Checks | Key Functions |
|------------------|--------|---------------|
| `stats` | 13 | `variance`, `std_dev`, `pearson_correlation`, `covariance`, `norm_cdf` |
| `linalg` | 17 | `solve_f64`, `lu_det`, `eigh_f64`, `cholesky_f64`, `tridiag_solve` |
| `linalg_ext` | 17 | `svd_f64`, `lu_inverse`, `gen_eigh_f64` |
| `special` | 26 | `gamma`, `erf`, `bessel_j`, `legendre_p`, `hermite_h`, `laguerre_l` |
| `optimize` | 10 | `nelder_mead`, `bisect`, `brent` |
| `precision` | 12 | f64 add, mul, fma, dot, Kahan sum |
| `Tensor API` | 84 | 84 ops (activations, losses, reductions, evolved) |
| `Tensor f64` | 35 | GPU f64 reductions and fused maps |
| `quantized` | 15 | Q4/Q8 dequant, quantized GEMV |
| `ML inference` | 13 | MLP + Transformer end-to-end |

### Locally Evolved Ops (7 modules in `src/evolved/`)

| Module | Workaround | Retires When |
|--------|------------|--------------|
| `layer_norm.rs` | GPU-resident (no readback) | `Tensor::from_buffer` is `pub` |
| `log_softmax.rs` | GPU-resident (no readback) | `Tensor::from_buffer` is `pub` |
| `mha.rs` | Decomposed MHA (avoids z-dispatch bug) | MHA z-dimension fix |
| `fused_pipeline.rs` | Single-encoder dispatch + shader cache | `TensorSession` extension |
| `fused_mlp.rs` | Fused 9-pass MLP | `TensorSession` MLP support |
| `fused_transformer.rs` | Fused 18-pass transformer | `TensorSession` transformer support |
| `matmul_*.wgsl` (2) | Double-buffered CPU + GPU matmul | Upstream kernel router |

### WGSL Shaders Ready for Upstream

| Shader | Purpose | Location |
|--------|---------|----------|
| `HEAD_SPLIT_WGSL` | `[seq, d_model]` → `[n_heads, seq, d_head]` | `fused_pipeline.rs` inline |
| `HEAD_CONCAT_WGSL` | Reverse of head-split | `fused_pipeline.rs` inline |
| `BATCHED_ATTENTION_WGSL` | Fused QK^T/√d → softmax → ·V | `fused_pipeline.rs` inline |
| `matmul_cpu_tiled.wgsl` | 32×32 double-buffered, 8×4 micro-kernel | `evolved/` |
| `matmul_gpu_evolved.wgsl` | 32×32 double-buffered, 2×2 micro-kernel | `evolved/` |

---

## 6. Unresolved Items from Feb 19 (11 Issues)

All 11 issues from the previous handoff remain pending. Summary:

| Priority | Issue | Impact |
|----------|-------|--------|
| **Critical** | Per-op command submission | 46–78× penalty |
| **Critical** | Naive matmul (no tiling) | CPU 3× slower than Python |
| **High** | `Tensor::from_buffer` `pub(crate)` | Forces 2 round-trips |
| **High** | MHA z-dimension dispatch bug | Correctness (tokens 1–7 zeroed) |
| **Medium** | `layer_norm` round-trip | 5× penalty |
| **Medium** | `log_softmax` round-trip | 5× penalty |
| **Medium** | `science_limits()` CPU failure | Blocks CPU validation |
| **Medium** | Softmax pooled buffer corruption | Correctness |
| **Low** | `leaky_relu` Params mismatch | wgpu panic |
| **Low** | `elu` Params mismatch | wgpu panic |

Full details with code locations and suggested fixes: `specs/TOADSTOOL_HANDOFF.md`.

---

## 7. Learnings Relevant to ToadStool Evolution

### 7.1 Deterministic Stochastic Validation

For stochastic algorithms (EA, Monte Carlo, ODE with noise), **bit-for-bit
reproducibility across Python and Rust is not achievable** due to different
PRNG implementations (Python: PCG64, Rust: Xoshiro256**). The validation
strategy is **qualitative property checking**: verify inequalities,
monotonicity, stability, and relationships rather than exact numerical match.

This means the validation harness needs `check_bool` (property holds?),
`check_lower` (value above threshold?), and `check_upper` (value below
threshold?) in addition to `check_abs` (numerical match within tolerance).

### 7.2 Cross-Domain Primitive Convergence

The 23-paper catalog confirms the isomorphism thesis with concrete evidence:
**GEMM appears in 18/23 experiments**, `reduce_sum` in 20/23, and
`elementwise` operations in 23/23. The long tail of domain-specific
primitives (ODE integration, eigendecomposition, HMM chain) is small:
only 4 truly new patterns emerged from 13 papers.

### 7.3 Unidirectional Streaming Opportunity

Several Phase 0++ patterns are naturally suited to **ToadStool's
unidirectional streaming** model:

- **EA generation loop**: Upload population → GPU fitness eval → download
  fitness → CPU selection/mutation → repeat. The fitness evaluation is
  entirely GPU-side with no intermediate readback.
- **HMM forward**: Upload observation sequence → sequential GPU matmul
  chain → download final log-likelihood. Each step writes to the same
  GPU buffer (ping-pong).
- **ODE integration**: Upload initial state → GPU RK4 steps → download
  final state. The entire trajectory stays GPU-resident.

In all cases, the dispatch overhead dominates the compute. **Reducing
round-trips from O(T) to O(1)** — exactly what ToadStool's streaming
architecture provides — would unlock the GPU advantage at these scales.

### 7.4 The Phase 0++ Modules Are Tier A (Direct Port)

All 13 new Rust modules are **pure math** with no framework dependencies,
no PyTorch training loops, and no real data requirements. They use only:
- Arithmetic (add, mul, div, sqrt, exp, ln)
- Matrix operations (GEMM, transpose, elementwise)
- Reductions (sum, mean, max, argmax)
- Sorting (partial sort for selection algorithms)

This makes them ideal **Tier A** candidates for direct BarraCUDA CPU port,
followed by GPU promotion. No adaptation layer needed.

---

## 8. Reproduction Commands

```bash
# Full Python baselines (190/190 PASS, ~10 min)
bash scripts/run_all_baselines.sh

# Full Rust validation (409/409 PASS, ~10 sec)
make validate

# Just the new Phase 0++ validators
cargo run --release --bin validate_swarm_robotics
cargo run --release --bin validate_sate_alignment
cargo run --release --bin validate_introgression
cargo run --release --bin validate_regulatory_network
cargo run --release --bin validate_signal_integration
cargo run --release --bin validate_spectral_commutativity
cargo run --release --bin validate_anderson_localization

# All quality gates
make check
```

---

*neuralSpring Phase 0++ complete. 23 papers, 599/599 PASS. 7 new algorithmic
patterns identified for GPU promotion. 11 prior shortcomings remain pending.
Deterministic PRNG infrastructure in place. All modules Tier A — ready for
direct BarraCUDA CPU port and GPU acceleration via unidirectional streaming.*
