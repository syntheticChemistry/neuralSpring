# neuralSpring — baseCamp: Per-Faculty Research Briefings & Cross-Domain Extensions

**Last Updated**: April 20, 2026 (Session S185)
**Status**: **27 papers** (full queue complete) + 6 baseCamp sub-theses + WDM surrogates + coralForge (nF-01/02/03) + 5 novel compositions + playGround (Squirrel MCP + Model Lab + compute triangle), **4,900+ checks**, ~97% GPU promotion, 269 binaries, 520+ `.rs` files, **1,234 lib + 73 forge + 80 playGround tests**, 68 modules, **228+** named tolerances, 14 proptest invariants. Zero clippy (pedantic+nursery+cast deny, workspace-wide), zero fmt, zero doc warnings, zero unsafe, zero `#[allow()]`, zero mocks in production. **Python→Rust→Primal→guideStone validation stack** — 6 composition validators + `IpcMathClient` + `neuralspring_guidestone` v0.3.0 (via `primalspring::composition` API), proto-nucleate aligned to upstream `downstream_manifest.toml` (7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES`), primal speaks full deployment standard (30 capabilities: health triad, identity, MCP, Squirrel routing, Tower discovery). **guideStone Level 3**: 29/29 bare ALL PASS (P1-P5 certified). `is_skip_error` unified skip classification. guideStone standard v1.2.0. BLAKE3 CHECKSUMS (15 files) via `primalspring::checksums`. primalSpring v0.9.17. genomeBin v5.1 (46 binaries, 6 target triples). Stadial `deny.toml` bans enforced. `rust-toolchain.toml`. ecoBin harvest. Edition 2024, MSRV 1.87. **V136 handoff**. barraCuda v0.3.12, toadStool S146+, coralReef Iter 49.

- **S181 (Apr 11, 2026)**: Full composition evolution — 30-capability surface (health.check, identity.get, mcp.tools.list added to ALL_CAPABILITIES), Squirrel inference routing (try_squirrel_route fallback), Tower Atomic startup discovery (BearDog + Songbird probes), Tier 3 `validate_composition_evolution` validator, `composed` feature gate, ToadStool discovery fix, tolerance forensics, deploy graph V131/S181. V131 handoff.
- **S180 (Apr 11, 2026)**: Composition evolution — deployment health triad (`health.check`), T4 discovery (`identity.get`), `mcp.tools.list` on primal, iterative method normalization, 27/27 MCP tool definitions, deploy graph `nest_atomic`, primalSpring graph reconciliation, plasmidBin metadata refresh. V130 handoff.
- **S179 (Apr 11, 2026)**: Composition validation — PRIMAL_GAPS.md reconciled (10 gaps, 4 resolved), deploy graph V129, capability surface expanded to 26, version strings aligned. V129 handoff.
- **S178 (Apr 11, 2026)**: Composition validation phase — three-layer stack (Python→Rust→NUCLEUS). `validate_all` wired with 3 composition validators + exit-2 honest skip. `PRIMAL_GAPS.md` reconciled (inference.* wip, binary naming resolved). Version strings aligned to barraCuda v0.3.11. Deploy graph V128/S178. V128 handoff.
- **S177 (Apr 10, 2026)**: NUCLEUS composition validation — proto-nucleate graph validator, inference chain validator, primal discovery validator, bonding policy (Metallic/InternalNucleus), composition infrastructure (`validation::composition`), `inference.*` capability wiring (niche, config, handlers, rpc_service, MCP tools), ecoBin harvest script, CI composition-validation job. V127 handoff.
- **S176 (Mar 24, 2026)**: Deep audit — clippy zero-warning gate restored, provenance environment centralization (19 literals → 2 constants), IPC resilience wired (RetryPolicy+CircuitBreaker → PetalTongue), GPU module refactor (gpu.rs → gpu/), integration tests 9→12 (49/49 provenance), doc reconciliation. V126 handoff.
- **S175 (Mar 24, 2026)**: Ecosystem absorption — `ValidationSink` (5 sinks, 12 tests), `cast_possible_truncation`/`cast_sign_loss` deny, 4 provenance integrity tests, deploy graph V124/S174, leverage guide refresh.
- **S174 (Mar 24, 2026)**: Deep audit execution — zero `#[allow()]`, tolerance fidelity (all literals centralized, `GPU_MULTI_OBJ_BESSEL_F64`, 4 upstream contract constants), self-knowledge compliance (dead hints removed, origins neutralized, petalTongue gated), 49 Python provenance headers, CONTRIBUTING.md + SECURITY.md.
- **S173 (Mar 24, 2026)**: Deep debt resolution — `thiserror` typed errors, 3 smart module decompositions (nucleus_pipeline, glucose_prediction, immunological_anderson), cargo-deny CI, IPC smoke test, barraCuda feature selection, coral forge shader absorption plan.
- **S172**: Deep evolution & ecosystem absorption — DeviceCapabilities migration (11 files), workspace `[workspace.lints]`, 163 playGround missing-docs resolved, `normalize_method` IPC from barraCuda, 3 validation binaries refactored by responsibility, config centralization (8 env vars), `#[allow]`→`#[expect]` workspace-wide, ~1,385 tests / 0 clippy / 0 fmt / 0 doc. **V122 handoff**
- **S170**: Deep debt execution — `expected_source()` provenance fix (9→49+ mappings), 66 clippy→zero, `ipc_client.rs` 885→448 LOC (`discovery.rs` extracted), `TensorSession`/`StatefulPipeline` wired, 8 new proptests, `head_split`/`head_concat` lean cycle, CI workspace-wide. V119 handoff
- **S167**: Deep audit + ecosystem evolution — `pearson_r` centralized, `primal_names::display`, fossil `#[allow()]`→`#[expect()]`, ecoBin CI, `capability_registry.toml`, upstream `WGSL_MEAN_REDUCE` re-export, L-BFGS path. V118 handoff
- **S166**: Doc evolution — stale counts corrected, full barraCuda Sprint 7 review. V117 handoff
- **S165**: Ecosystem absorption — `mul_add()` FMA sweep (14 sites), IPC proptest, `ECOSYSTEM_LEVERAGE_GUIDE.md`. V116 handoff
- **S163–S164**: Edition 2024, deep debt, health probes, RetryPolicy/CircuitBreaker. V114–V115 handoffs
- **S157–S162**: IPC evolution, cross-ecosystem absorption, Tower Atomic, safe_cast, discover_primal. V108–V113 handoffs
- **S146–S156**: playGround compute triangle, niche architecture, capability discovery. V99–V107 handoffs

## Purpose

Per-faculty validation briefings and cross-domain extension proposals.

The validation chain proves peer-reviewed science runs faithfully through the
sovereign compute stack:

```
Python baseline (peer-reviewed science, documented provenance)
    → Rust validation (spring binary — the "Rust proof", Level 2)
    → BarraCUDA CPU → GPU Tensor → metalForge WGSL → pipeline
    → cross-dispatch → multi-GPU (~97% GPU promotion)
    → Primal composition (IPC round-trip parity against NUCLEUS primals)
    → guideStone (self-validating bare: 29/29 PASS, 5 properties certified)
    → NUCLEUS deployment (plasmidBin ecobins on clean machine) [Level 4+]
```

Each briefing maps this chain per-paper. Extension proposals identify where
neuralSpring's validated primitives can serve larger fields of study,
cross-domain science, and the gen3 baseCamp sub-theses.

## Faculty Summary

| Faculty | Institution | Track | Papers | Checks | Domains |
|---------|------------|-------|--------|--------|---------|
| Emily Dolson | Michigan State | Evolutionary Computation | 5 (011–015) | 50 | NK fitness, MODES, eco dynamics, directed evolution, swarm robotics |
| Kevin Liu | Michigan State | Phylogenetics / HMM | 3 (016–018) | 38 | HMM forward/backward/Viterbi, SATé alignment, introgression detection |
| Chris Waters | Michigan | Microbial Cooperation | 3 (019–021) | 21 | Game theory, regulatory networks, signal integration |
| Ilya Kachkovskiy | Michigan State | Spectral Theory | 2 (022–023) | 16 | Spectral commutativity, Anderson localization |
| R. Anderson / Campbell | Various | Population Genetics | 2 (024–025) | 16 | Pangenome selection, meta-population dynamics |
| Wei Liao (Wang et al.) | Michigan State (BAE/ADREC) | Bioprocess Engineering | 1 (027) | 59 | ESN digestion prediction, methane yield, bC/gT GPU |

**Total**: 15 Phase 0++ papers + Paper 026 (Chuna LSTM glucose) + Paper 027 (Wang/Liao digestion ESN), 200+ Phase 0++ checks (all PASS at 7 tiers).

## Validation Chain

```
Layer 1 — Science Fidelity (Python validates Rust):
  Python baseline (seed=42) → Rust CPU (provenance) → BarraCUDA CPU
    → GPU Tensor (WGSL) → metalForge shaders → GPU pipeline → cross-dispatch
      → gpu_dispatch (~97% pure GPU, Phase C: HMM chains, FST, introgression)
        → CPU↔Python parity (39/39 PASS, 1e-10 cross-language)

Layer 2 — Compute Sovereignty (Rust validates GPU/WGSL):
  Rust reference → barraCuda ops → WGSL shaders → TensorSession pipeline
    → f64/df64 precision dispatch → cross-substrate (RTX 4070 + TITAN V NVK)

Layer 3 — Primal Composition Proof (Rust+Python validate primal IPC):
  downstream_manifest.toml → proto-nucleate = pure primal NUCLEUS (no spring binary)
    → 7 PROTO_NUCLEATE_VALIDATION_CAPABILITIES (tensor.matmul, tensor.create,
        compute.dispatch, inference.complete, inference.embed, stats.mean, crypto.hash)
      → call primals by capability via IPC → compare against Python/Rust baselines
        → 6 composition validators + neuralspring_guidestone (via primalspring::composition)
          → guideStone: bare properties (P1-P5) → discovery + liveness → domain parity
            → additive NUCLEUS (BearDog signing, Songbird discovery)
              → honest skip (exit 2) when primals unavailable

Layer 4 — Deployment (plasmidBin/genomeBin + benchScale validate ecoBin):
  harvest ecoBin → genomeBin v5.1 (46 binaries, 6 target triples) → fetch + smoke
    → benchscale validate ipc → C1-C7 composition probes
      → Level 4: deploy NUCLEUS from plasmidBin, run guideStone externally
```

Edition 2024 (S163): All 3 workspace crates on Rust 2024. Proptest invariants
(softmax, entropy, relu, rk4). `ipc_resilience` (RetryPolicy + CircuitBreaker)
for transient IPC failures. Deployment health triad (S180). Squirrel routing +
Tower Atomic discovery (S181). Tier 3 composition evolution validator (S181).
primalSpring v0.9.17 absorption (S185): `is_skip_error` unified skip
classification, guideStone standard v1.2.0, genomeBin v5.1 for Level 4 path.

## Briefings

| File | Faculty | Papers |
|------|---------|--------|
| [dolson.md](dolson.md) | Emily Dolson (MSU) | 011–015: Evolutionary computation |
| [liu.md](liu.md) | Kevin Liu (MSU) | 016–018: Phylogenetics / HMM |
| [waters.md](waters.md) | Chris Waters (UMich) | 019–021: Microbial cooperation |
| [kachkovskiy.md](kachkovskiy.md) | Ilya Kachkovskiy (MSU) | 022–023: Spectral theory |
| [anderson.md](anderson.md) | R. Anderson / Campbell | 024–025: Population genetics |

## baseCamp Research Program: Biophysical AI Interpretability

neuralSpring's novel research program applies validated physics and biology
primitives to understanding AI systems as physical systems. Six sub-thesis
proposals, each grounded in 3-6 published papers and using existing validated
primitives.

| File | Sub-Thesis | Domain Cross |
|------|-----------|--------------|
| [extensions.md](extensions.md) | **Program overview** — sub-theses 01-05, priority, reading order | All |
| [sub01_weight_hamiltonians.md](sub01_weight_hamiltonians.md) | Weight matrices as Anderson Hamiltonians | Random matrix theory x DL |
| [sub02_information_propagation.md](sub02_information_propagation.md) | Information flow as wave propagation | Statistical physics x RNNs |
| [sub03_loss_landscapes.md](sub03_loss_landscapes.md) | Loss landscapes as energy landscapes | Chemical physics x Optimization |
| [sub04_neural_pgm.md](sub04_neural_pgm.md) | Neural networks as probabilistic graphical models | Bayesian inference x Interpretability |
| [sub05_multiagent_qs.md](sub05_multiagent_qs.md) | Multi-agent AI coordination as quorum sensing | Microbial ecology x Multi-agent AI |
| [sub06_immunological_anderson.md](sub06_immunological_anderson.md) | Anderson localization in immunological signaling | Immunology x condensed matter x drug repurposing |

---

## Infrastructure Summary

### Upstream Rewiring (Sessions 56–79 — ToadStool S66)

baseCamp functions delegate to upstream BarraCUDA. Session 58 rewired
7 core Dispatcher methods to `barracuda::dispatch::domain_ops` and wired in
`GpuDriverProfile`. Sessions 77–79 added WDM surrogates and completed
cross-spring rewiring to ToadStool S66 APIs.

| Local Function | Upstream | Sub-thesis | Session |
|----------------|----------|-----------|---------|
| `graph_laplacian` | `barracuda::linalg::graph` | Sub-05 | S56 |
| `disordered_laplacian` | `barracuda::linalg::graph` | Sub-05 | S56 |
| `belief_propagation_chain` | `barracuda::linalg::graph` | Sub-04 | S56 |
| `numerical_hessian` | `barracuda::numerical` | Sub-03 | S56 |
| `mat_mul` | `barracuda::dispatch::matmul_dispatch` | All | S58 |
| `frobenius_norm` | `barracuda::dispatch::frobenius_norm_dispatch` | Sub-01 | S58 |
| `transpose` | `barracuda::dispatch::transpose_dispatch` | Sub-01 | S58 |
| `softmax` | `barracuda::dispatch::softmax_dispatch` | All | S58 |
| `l2_distance` | `barracuda::dispatch::l2_distance_dispatch` | Sub-02 | S58 |
| `mean` | `barracuda::dispatch::mean_dispatch` | All | S58 |
| `variance` | `barracuda::dispatch::variance_dispatch` | All | S58 |
| `softmax_row_wise` | `Tensor::softmax_dim(1)` | Sub-04 (PGM) | S73 |
| `fst_single_locus` | `barracuda::ops::bio::fst_variance_decomposition` | Pop genetics | S73 |
| `pairwise_fst_full` | upstream per-locus decomposition | Pop genetics | S73 |
| Viterbi argmax | `Tensor::argmax_dim(0)` | Sub-04 (HMM) | S73 |
| `metrics::mae` | `barracuda::stats::mae` | WDM surrogates | S78 |
| `shannon_entropy` | `barracuda::stats::shannon_from_frequencies` | All | S78 |
| `hill_activation` | `barracuda::stats::hill` | Sub-02, Waters | S78 |
| `hill_repression` | `barracuda::stats::hill` (inverted) | Sub-02, Waters | S78 |
| `modes::l2_distance` | `barracuda::dispatch::l2_distance_dispatch` | Sub-05 | S78 |
| `complexity_metric` | `barracuda::stats::fit_linear` | MODES | S78 |
| `MlpLayer` (EOS surrogate) | `barracuda::nn::SimpleMlp` | WDM surrogates | S121 |
| `MlpLayer` (Transport surrogate) | `barracuda::nn::SimpleMlp` | WDM surrogates | S121 |
| `hmm_viterbi_chain_gpu` (per-step f32) | `barracuda::ops::bio::hmm_viterbi` (f64 ComputeDispatch) | HMM/Phylo | S121 |

### Hardware Validation

All baseCamp experiments inherit neuralSpring's validated multi-tier pipeline:

1. **BarraCUDA CPU**: Pure Rust, machine-precision agreement with Python
2. **BarraCUDA GPU**: Tensor API, f32-f64 agreement < 1e-3 across all domains
3. **metalForge mixed**: Same answer on CPU, GPU, NPU — multi-substrate dispatch
4. **df64 core streaming** (S88): f64 buffer I/O → df64 compute on FP32 cores → f64
   output. Achieves ~14-digit (fp48) precision on consumer GPUs. Arithmetic ops:
   3.6e-8 to 5.6e-7 max diff. Transcendental ops: 1.7e-4 to 3.4e-4 max diff.
   `Fp64Strategy::Hybrid` auto-detected on RTX 4070 (1:64 FP64:FP32 ratio).

### Performance Summary

| Metric | Value |
|--------|-------|
| Pure Rust vs Python | 83.6× geomean (11 domains); fastest 1104× (multi-obj) |
| CPU↔Python parity | 39/39 PASS (1e-10 cross-language) |
| Dispatch overhead | ≤1.04× for 9/10 ops (transparent) |
| GPU vs Python | Up to 104x (transformer medium) |
| GPU crossover | ~1.5 ms dispatch overhead |
| Multi-GPU | Bit-identical (RTX 4070 + TITAN V NVK) — **384/384 Titan V (S82)** |
| Fused pipeline | 46-78x over per-op dispatch |
| GPU math coverage | ~97% of production operations (Phase C complete) |

### Open Data

All 27 papers use computationally generated data from published parameters.
No external datasets, no API dependencies, no proprietary sources.
See `specs/DATA_PROVENANCE.md` for full inventory.
Paper 027 (Wang/Liao digestion) is the gateway to real ADREC digester data
and NCBI digester microbiome BioProjects — extension targets for gen3 Paper 16.

### Session 168 — Deep Debt Execution + Ecosystem Handoff (March 18, 2026)

- `expected_source()` provenance fix: 9 → 49+ script mappings (was non-functional)
- 66 clippy warnings → zero (workspace-wide including tests)
- `ipc_client.rs` smart refactor: 885 → 448 LOC (`discovery.rs` extracted, 439 LOC)
- `TensorSession`/`StatefulPipeline` wired to `Dispatcher` for fused GPU pipelines
- 8 new proptests: metrics (R²/RMSE/MAE), spectral (Frobenius/transpose/normal)
- `head_split`/`head_concat` lean cycle completed in absorption manifest
- CI workspace-wide: `cargo clippy`/`cargo test`/`cargo fmt`/`cargo doc`
- **Quality**: 1312 tests (1164+73+75), 0 clippy, 0 fmt, 0 doc warnings, 0 unsafe

### Session 167 — Deep Audit + Ecosystem Evolution (March 18, 2026)

- Comprehensive ecosystem audit against wateringHole standards (15 dimensions)
- `pearson_r` centralized, `primal_names::display`, fossil `#[allow()]`→`#[expect()]`
- ecoBin CI, `capability_registry.toml`, upstream `WGSL_MEAN_REDUCE` re-export
- **Quality**: 1156 lib tests, 0 clippy, 0 fmt, 0 doc warnings, 0 unsafe
