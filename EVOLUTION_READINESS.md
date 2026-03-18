# neuralSpring — Evolution Readiness

**Date**: March 17, 2026 (Session 165 — Ecosystem Absorption: FMA sweep 14 sites, IPC proptest 8 tests, ECOSYSTEM_LEVERAGE_GUIDE.md + V116 Handoff)
**barraCuda**: v0.3.5 at `0649cd0` (`../barraCuda/crates/barracuda`). 719 WGSL shaders, wgpu 28, Sprint 2 APIs (activations, rng, tridiagonal_ql), healthSpring domain, batched logsumexp, CoralReefDevice. `PrecisionRoutingAdvice` with `F64NativeNoSharedMem` Ada Lovelace reclassification, `WORKGROUP_SIZE_1D` constant, cross-spring provenance registry, `BatchedOdeRK45F64`, `mean_variance_to_buffer`, coralReef Phase 10 IPC. Three-tier precision: F32/F64/Df64 (lean 3-tier model, F16+templates removed). Deep debt: typed errors, named constants, `Arc<str>` hot-path, `RwLock` compiler, ring buffer back-off, streaming pipeline completion. **Known issue**: `enable f64;` in WGSL triggers PTXAS silent-zero regression on Ada Lovelace — fix implemented locally in `pipeline_cache.rs`, pending upstream absorption.
**ToadStool**: S146 at `751b3849`. Hardware testing, PCIe transport, ResourceOrchestrator, 19,900+ tests. Absorbed neuralSpring `pipeline_graph` DAG + hotSpring `streaming_dispatch`. Dual-write discovery (canonical + coralReef-compatible). `GpuDevice` enrichment (render_node, driver, arch). `gpu.dispatch` + `shader.compile` + `orchestration` capabilities. Compute triangle unblocked. Deep debt, zero-copy.
**coralReef**: Iteration 49. `coral-glowplug` crate (sovereign PCIe broker), GV100 per-runlist registers, HBM2 training, bar cartography, 1842+ tests. Sovereign shader compiler (WGSL → native GPU binary). NVIDIA SM70-SM89, AMD RDNA2+ (E2E GPU dispatch verified on RDNA2). Three-tier precision architecture: f32 native, f64 DFMA+polynomial lowering, df64 preamble auto-prepend. `Fp64Strategy` in `CompileOptions`. Built-in `df64_preamble.wgsl`. 8 neuralSpring shaders in corpus.
**neuralSpring**: 1152 lib tests + 70 playGround + 73 forge tests, 260 binaries, 48 modules, 0 clippy warnings (pedantic+nursery, -D warnings), 0 fmt diffs. All 3 crate roots `forbid(unsafe_code)`. All files ≤1000 LOC. AGPL-3.0-or-later. **Zero C dependencies** in entire workspace (ecoBin compliant — reqwest/ring eliminated via Tower Atomic). **Rust Edition 2024** (let chains, reserved `gen` keyword, pattern ergonomics). proptest property-based testing (6 invariants).
**S165**: Ecosystem absorption — `mul_add()` FMA sweep (14 sites, 10 modules), IPC proptest invariants (8 new tests: RetryPolicy bounds, CircuitBreaker state machine, parse_capability_list fuzz, DispatchOutcome fuzz, IpcError contract), `specs/ECOSYSTEM_LEVERAGE_GUIDE.md`. V116 handoff.
**S164**: Deep debt evolution — 7 inline tolerances named and wired (domain_guards), `solve_symmetric` → `barracuda::linalg::solve::solve_f64_cpu()`, MSRV `rust-version = "1.87"` pinned across workspace, `/tmp` → `temp_dir()`, `partial_cmp().unwrap()` → `total_cmp()`, `tolerances/training.rs` extracted, test sockets → `niche::NICHE_NAME`. V115 handoff.
**S163**: Edition 2024 evolution — Rust Edition 2024 upgrade, `health.liveness`+`health.readiness` IPC probes, `ipc_resilience.rs` (RetryPolicy + CircuitBreaker), proptest (6 property tests), MCP tools 14→16, tolerance provenance doc comments, deny.toml hardened (unknown-git=deny, advisory DB), DispatchOutcome enriched.
**S162**: Cross-ecosystem absorption execution — 4-format `parse_capability_list()`, `discover_primal()` + `socket_env_var()`, `DispatchOutcome` enum, `resilient_call()` circuit breaker, `safe_cast` module (5 helpers), zero `eprintln!` workspace-wide (1642 → 0), safe GPU casts. V113 handoff.
**S161**: Doc cleanup + structured logging completion — hardcoded `"biomeos.sock"` → config constants (3 files), playGround 28× `eprintln!` → `log::*` (zero remaining), root doc consolidation, archive sweep clean. V112 handoff.
**S160**: IPC evolution — `IpcError` typed enum (healthSpring V31 / rhizoCrypt V13), `extract_rpc_error()` centralized (airSpring V0.8.6), `call_typed()` for structured errors, typed `compute.dispatch` protocol (wetSpring V124). `JsonRpcError::code` i32→i64. V111 handoff.
**S159**: Cross-ecosystem absorption execution — `OrExit<T>` (wetSpring V123), `deny.toml` (groundSpring/healthSpring), primal `eprintln!` → `log::info!/warn!/debug!`, dep audit confirmed zero C. V110 handoff.
**S158**: Cross-ecosystem absorption — `#[allow()]` → `#[expect(reason)]`, stale expectations pruned, `temp-env` for safe env testing, validate_barracuda_tensor 918→875 LOC, hardcoded primal names → constants. V109 handoff.
**S157**: Deep debt + idiomatic Rust — 5 blanket lint suppressions eliminated, primal binary refactored, error handling evolved, validate_modern_cross_spring 949→865 LOC, **reqwest+ring removed** (Tower Atomic via Songbird IPC), zero C deps, bytemuck aligned, Kokkos provenance documented. V108 handoff.
**S156**: Full codebase audit + IPC discovery fixes — 2 critical bugs fixed, typed BiomeOsClient, 3 validators to harness. V107 handoff.
**S155**: Cross-spring absorption — `src/primal_names.rs` (11 primal + 4 domain constants), `control/tolerances.py` (80+ shared Python tolerance constants), deploy graph provenance trio, V106 handoff.
**S154**: Niche deployment architecture — `src/niche.rs` (22 capabilities), `graphs/neuralspring_deploy.toml` (5-phase biomeOS deploy), hardcoded primal names eliminated, V105 handoff.
**S153**: Full ecosystem audit + deep debt — `ALL_CAPABILITIES` unified into `config.rs`, `validate_gpu_eigensolve_pipeline` migrated to `ValidationHarness`, 3 new centralized tolerances, forge `#![forbid(unsafe_code)]`, `BaselineProvenance::expected_source()`, playGround lint alignment, 4 new GPU tests, V104 handoff.
**S152**: Deep debt execution — 15+ tolerance literals centralized, capability-based discovery, shared validation infrastructure, V103 handoff.
**S151**: Deep audit — ecoBin compliance (`reqwest` → `rustls-tls`), capability-based IPC discovery, 12 tolerance centralizations, V102 handoff.
**S150**: Compute triangle integration — typed ToadStool/coralReef IPC clients, hot/cold dispatch benchmarks, `neuralspring_compute_probe`, V101 handoff.
**S149**: HuggingFace Model Lab — `neuralspring_model_lab` binary, safetensors weight loading, GPT-2 transformer forward pass on barraCuda, f16/bf16→f32 conversion.
**S148**: Squirrel MCP adapter — `neuralspring_mcp_adapter` (JSON-RPC MCP server), `neuralspring_interactive` (REPL), 14 MCP tool definitions, biomeOS 5-tier socket discovery.
**S147**: Deep debt execution — zero inline magic numbers in production code (all centralized to `tolerances::` registry). `digester_anderson::shannon_diversity` rewired to `barracuda::stats::shannon_from_frequencies` (zero duplicate math). 6 composition experiment provenance records added. Hardcoded petalTongue discovery strings evolved to `config::` constants. V100 handoff for toadStool/barraCuda. Root doc, baseCamp, and wateringHole updates.
**S146**: Industry GPU parity benchmarks — BarraCUDA WGSL vs cuBLAS/cuDNN/cuFFT/FlashAttention on RTX 4070 (PyTorch/CUDA control scripts). FFT beats cuFFT at sizes 256–16K (0.19–0.85×). GEMM beats cuBLAS at 2048×2048 (0.16×). Deep audit: provenance accuracy fix, tolerance tightening (GPU_SOFTMAX_SUM_F32 0.01→5e-3), visualization refactor (3-file split, all < 1000 LOC), Clippy pedantic clean. 4 Python control scripts + 1 Rust benchmark binary.
**S145**: GPU infra evolution sprint — barraCuda v0.3.5 sync, 5 workload rewires (chi², KL, HMM backward, HMM Viterbi, pairwise L2), NUCLEUS pipeline GPU dispatch (eigensolve + attention_anderson via Dispatcher), composition_pipeline() mixed-hardware substrates, 4 GPU experiment binaries (Exp 103-106).
**S144**: petalTongue composition visualization + NUCLEUS pipeline executor. 5 new scenario builders for composition experiments. `composition_study()` combiner (21 tracks). `composition_pipeline()` 6-stage DAG in metalForge. `nucleus_pipeline.rs` executor implementing Tower→Node→Nest dispatch. `visualize.sh --compositions` mode. 219 files with barracuda imports. V97 handoff.
**S143**: 5 novel composition experiments (Exp 097–101): digester×Anderson, isomorphic reservoir ensemble, WDM ensemble QS, HMM introgression NN, attention Anderson spectral. Axis 2 complete. V96 handoff.
**S138**: Industry gap closure — streaming FASTA parser (16 tests), CPU-reference BLAST pipeline (`search/kmer_index`, `search/seed_extend`, 19 tests), `bench_kokkos_parity` GPU benchmark harness (9 ops × production scale). `INDUSTRY_TOOL_GAP_ANALYSIS.md`, `BLAST_LIKE_SEARCH_SCOPE.md`, `MSA_PIPELINE_SCOPE.md`. V92 handoff written.
**S137**: Upstream rewire + deep debt execution. Reviewed 27+ Mar 8–9 wateringHole handoffs (ToadStool S139, barraCuda Sprint 2 + deep debt + concurrency, coralReef precision architecture, groundSpring V98/V100 sovereign rewire). **Rewires**: hardcoded `256` → `WORKGROUP_SIZE_1D` (library+forge, 15 sites). **Absorption docs**: 7 WGSL shaders updated to "absorbed upstream" status. toadStool S139 `pipeline_graph` absorption acknowledged. **Deep debt**: `gpu_or_exit()` async helper eliminates 5-line GPU init boilerplate (~75 binaries); duplicate `max_abs_diff` in `validate_gpu_promotion` eliminated; 2 largest GPU binaries refactored to use `gpu_or_exit()`. **Full audit**: zero unsafe, zero TODOs, zero mocks in production, zero hardcoded paths, zero non-Rust deps (wgpu is the GPU bridge), all library files < 800 LOC. V92 handoff written. 968 lib + 71 forge tests PASS.
**S136**: Deep audit + evolution — `PetalTonguePushClient::headless()` (socket hardcoding eliminated), `Gpu::read_buffer_u32` (upstream parity), `validate_gpu_pure_workload_all` refactored (976→940 LOC, raw wgpu eliminated), industry GPU parity gap documented (`specs/BENCHMARK_ANALYSIS.md`), Kokkos/Polybench/cuBLAS gap formally requested in V92 handoff. 968 lib tests (+2 headless client). All casts audited (f64→f32 for GPU, usize→f64 no From impl — legitimate).
**S135**: petalTongue visualization evolution — 7 new domain scenario builders (HMM, game theory, WDM, glucose, immunological, population, loss landscape), all 8 DataChannel types exercised, `TrainingVisualizer` live streaming, `full_study()` 12-track combiner, `neuralspring_live_dashboard` binary, `scripts/visualize.sh`, 56/56 petalTongue validation.
**S133**: metalForge PCIe `transfer_buffer_strategy()`, `NpuToGpuP2P` substrate, biomeOS pipeline DAG (`graph.rs`: topological execution, 3 canonical pipelines), petalTongue `StreamSession` + `push_replace` + 64KB IPC. Feature-gated `validate_all`. 220/220 `validate_all` PASS. V91 handoff.
**S121 rewires**: WDM surrogates → `barracuda::nn::SimpleMlp` (~300 LOC eliminated), HMM Viterbi chain → f64 `ComputeDispatch`. 46 total upstream rewires. 80/80 S121 rewire validation + 28/28 cross-spring modern bench.
**Pattern**: Python baseline → Rust validation → BarraCUDA CPU → BarraCUDA GPU Tensor → metalForge WGSL → GPU Pipeline → Cross-dispatch → Mixed-hardware → Multi-GPU → Phase 4 shader validation → ToadStool streaming → NUCLEUS compute dispatch → biomeOS integration → lean on upstream `compile_shader_df64`
**Hardware**: RTX 4070 (Vulkan, proprietary) + TITAN V (NVK GV100, open-source) — **both fully validated (S82)**

---

## Quick Status

41 Rust modules cover all 25 papers + 5 Phase 0/0+ studies + 6 baseCamp sub-theses + 5 WDM surrogates + 3 publication experiments + nF-03 AlphaFold3 Phase C.
233 validation binaries span 9 tiers: Python (Py), Rust native (Rs), BarraCUDA CPU (bC),
GPU Tensor (gT), metalForge WGSL (mF), GPU Pipeline (gP), Cross-dispatch (xD),
Mixed-hardware (mH), and Multi-GPU (mG).

| Category | Count | Status |
|----------|-------|--------|
| Python baselines | 397/397 | **COMPLETE** |
| Rust native validation | 1152 lib + 9 integration + 73 forge tests, 47 modules, 258 binaries | **COMPLETE** |
| BarraCUDA primitives | 272/272 | **COMPLETE** |
| BarraCUDA CPU (bC) | **24/25** papers (96%) | **ALL GREEN** |
| BarraCUDA GPU Tensor (gT) | **23/25** papers (92%) | **ALL GREEN** |
| metalForge WGSL (mF) | 15/25 papers (60%) | **ALL PASS** |
| GPU Pipeline (gP) | 15/25 papers (60%) — S74: +9 domains via `validate_gpu_pure_workload_all` | **ALL PASS** |
| Cross-dispatch (xD) | **15/15** Phase 0++ papers (100%) | **ALL GREEN** |
| Multi-GPU validation | RTX 4070 + TITAN V (NVK) — **384/384 Titan V (S82)** | **Bit-identical** |
| GPU shader validation | 126/126 (21 absorbed WGSL) + 37/37 (15 coralForge df64) | **COMPLETE** |
| GPU pipeline validation | 77/77 | **COMPLETE** |
| ToadStool shortcomings absorbed | **17/17** (S-01..S-17) | **ALL RESOLVED** |
| S-16 (transpose dispatch) | Fixed at `a4996b34` (S39) | **RESOLVED** upstream |
| S-15 (matmul hang) | Fixed at `a4996b34` (S39) | **RESOLVED** upstream |
| S-14 (naive matmul hang) | Naive tier removed at `a4996b34` (S39) | **RESOLVED** upstream |
| S-17 (pow(f64) crash) | Fixed at `c82c23d1` (S58) | **RESOLVED** upstream |
| S-13 (PooledBuffer race) | Deferred return + device poll | **RESOLVED** upstream (Session 39) |
| TS-003 (trig precision) | 7-term Taylor + Cody-Waite | **FIXED** upstream (Session 36) |
| TS-001 (pow_f64 precision) | Extended exp/log polynomials | **FIXED** upstream (Session 36) |
| Shader absorption | 21/21 WGSL shaders absorbed upstream | **S-03b RESOLVED** — ToadStool `0c998992` (matmul + head_split/head_concat) |
| Upstream wrapper validation | **10 bio ops** + f64 HMM + Gillespie + wetSpring trio + chi² | **74/74 PASS** |
| Upstream parity (dual-path) | **10 GPU validators** | **10/10 PASS** (9 bit-identical, 1 Bessel diff 1.95e-3) |
| ReduceScalarPipeline | f64 mean IPR via GPU reduce | **5.55e-17 diff** (machine ε) |
| Spectral theory stack | Lanczos, Anderson, Hofstadter, Lyapunov, eigh×Sturm | **17/17 PASS** (hotSpring lineage) |
| Capability-based dispatch | 12 validators + evolved HMM use `Gpu::dispatch_1d` | **Runtime-validated** (Sessions 40, 42) |
| Upstream vs local benchmark | **10 kernels**, RTX 4070 | **0.72–1.10×** overhead (negligible) |
| LeNet-5 full bC validation | Conv→Pool→FC via `cpu_conv_pool` | **13/13 PASS** (new, Session 42) |
| Session 43: new WGSL shaders | logsumexp, stencil, rk45, wright-fisher (4 shaders, 4 validators) | **18/18 PASS** |
| Session 43: upstream wrappers | GillespieGpu, TaxonomyFcGpu, KmerHistogramGpu, UniFracPropagateGpu, chi² | **41/41 PASS** |
| Session 43: CPU vs GPU parity | Tensor API: MatMul, ReLU, Sigmoid, Tanh, Sum, erf, gamma, conv, pool | **17/17 PASS** |
| Session 43: dispatch routing | metalForge substrate heuristics (8 domains) | **16/16 PASS** |
| Session 43: mixed-hardware | MixedSubstrate, TransferCost, PcieBridge, cost model | **16/16 PASS** |
| Session 44: multi-GPU | RTX 4070 + TITAN V (NVK GV100): 131/131 PASS | **ALL GREEN** |
| Session 45: GPU promotion (Phase A) | `validate_gpu_promotion` 27/27 PASS (RTX 4070 + TITAN V NVK) | **ALL GREEN** |
| Session 46: GPU promotion (Phase B) | `validate_gpu_phase_b` 20/20 PASS (RTX 4070 + TITAN V NVK) | **ALL GREEN** |
| Session 44: stochastic pipelines | WF→reduce + Gillespie→reduce (zero CPU round-trips) | **10/10 PASS** |
| Session 44: Conv2d/MaxPool GPU | `Tensor::conv2d` + `Tensor::maxpool2d` WGSL shaders | **8/8 PASS** |
| Session 44: transformer bC | Full layer: Q/K/V, attention, FFN, residual, softmax | **12/12 PASS** |
| Session 44: BarraCUDA fixes | mean_reduce entry point + chi² expected values | **2 bugs fixed upstream** |
| Session 44→127: benchmarks | Pure Rust vs Python (15 domains, geomean) | **38.6× faster** (honest: includes 2 BLAS-bound) |
| Evolved LOC | ~2,864 fossilized | Documented, bench migration complete |
| gpu_dispatch, gpu_ops | Capability-based GPU/CPU dispatch + 47 promoted ops (now split into 7 domain files), 9 rewired to upstream domain_ops | **240 binaries** |
| `validate_all` (S115) | **220/220 PASS** (RTX 4070, all green) | **ALL GREEN** |
| Session 47: typed op migration | 10 validators rewired raw wgpu → typed BarraCUDA ops | **Cross-spring complete** |
| Session 48: mass typed op rewiring | 28 binaries rewired raw wgpu → typed BarraCUDA ops | **Complete** |
| Session 48: f32→f64 upstream sync | BatchFitnessGpu, LocusVarianceGpu, MultiObjFitnessGpu, WrightFisherGpu, StencilCooperationGpu, SwarmNnGpu | **Data type alignment** |
| Session 48: HillGateGpu f64 | Graceful skip on RTX 4070 (driver limitation) | **f32 path validated** |
| S-03b (MHA projection hangs) | Decomposed into matmul + head_split/head_concat (ToadStool `0c998992`) | **FULLY RESOLVED** upstream |
| Session 47: evolved/hmm_forward_gpu | Retired; HmmBatchForwardF64 (wetSpring) primary | **Fossil** `metalForge/fossils/evolved_hmm_forward_gpu/` |
| Session 54: baseCamp experiment expansion | 5 validators expanded 82→114 checks (nS-103..106, 205, 206, 304, 305, 402, 405, 504, 505) | **114/114 PASS** |
| Session 54: `validate_basecamp_gpu` | Pure GPU workload validation (eigensolve, variance, Pearson, entropy, matmul, chi², L2, KL) | **14/14 PASS** |
| Session 54: `bench_basecamp_parity` | CPU→GPU parity: var 7.77e-16, pearson 6.94e-18, entropy 1.60e-11 | **All sub-epsilon** |
| Session 55: `validate_compute_dispatch` | BarraCUDA CPU vs GPU dispatch parity (routing + variance/Pearson/entropy/chi²/eigh) | **16/16 PASS** |
| Session 55: `Dispatcher::mixed_dispatch()` | metalForge mixed-hardware wiring integrated into `gpu_dispatch` | **Wired** |
| Session 55: `validate_mixed_hardware` | Mixed-hardware dispatch (GPU↔NPU↔CPU routing, PCIe bridge, crossover) | **14/14 PASS** |
| Session 55: doc cleanup | 5 sub-thesis docs fixed (binary refs, check counts), 15 grounding papers → Primitives validated | **Done** |
| `validate_all` | **220/220 PASS** (RTX 4070) | **ALL GREEN** |
| Session 74: pure GPU all-domains | `validate_gpu_pure_workload_all` 10/10 PASS (9 typed GPU ops + determinism) | **ALL GREEN** |
| Session 74: evolution tier bench | `bench_evolution_tiers` 8 domains CPU→GPU portability | **PROVEN** |
| Session 74: cross-system dispatch | `validate_cross_system_dispatch` 46/46 PASS (discovery + heuristics + parity + NPU) | **ALL GREEN** |
| Session 87: WDM surrogates | 5 Python baselines (33/33 PASS) + 6 Rust validators (153/153 PASS incl. GPU) | **ALL GREEN** |
| Session 77: baseCamp GPU pure | `validate_basecamp_gpu_pure` 5/5 sub-theses on GPU | **ALL GREEN** |
| Session 78–79: cross-spring | `validate_cross_spring_evolution` 52/52 PASS, `bench_cross_spring_evolution` 28/28 PASS | **ALL GREEN** |
| Session 87: WDM queue closed | nW-03 S(q,ω) (27/27 Rs), nW-05 ESN regime (39/39 Rs), 5 Py baselines (33/33), 6 Rust validators (153/153) | **ALL GREEN** |
| Grand total checks | **4500+** (397 Py + 4000+ Rust/GPU) | **ALL GREEN** |

---

## Tier A — Shader Absorption Status

### ToadStool Evolution Since Last Sync (Session 104: f97fc2ae)

| Session | Key Changes for neuralSpring |
|---------|----------------------------|
| S39 | Absorb all Spring shaders (7 bio ops, 11 HFB physics, 3 wetSpring WGSL); S-14/S-15/S-16 fixes; `FlatTree`, `sparse_eigh`, `execute_to_buffer` |
| S40 | Richards PDE solver, moving window GPU stats |
| S41 | `cpu_conv_pool` made `pub` (conv2d, max_pool2d, avg_pool2d); 6 f64 shader compile bugs fixed; APIs exposed for Springs |
| S42 | 19 new WGSL shaders (chi_squared_f64, rk45_f64, factorial_f64, cubic_spline_f64, etc.); BarraCUDA → BarraCuda doc rename |
| S68 | Universal precision: ALL 700+ shaders evolved to f64 canonical with LazyLock downcast. Zero f32-only shaders remain. Dual-layer precision (op_preamble + naga IR). `downcast_f64_to_df64()` pipeline |
| S70+ | `Fp64Strategy::Concurrent`, 7 WGSL shaders (gelu/sigmoid/softmax/layernorm DF64, sdpa_df64, brent_f64, seasonal_pipeline), `SymmetrizeGpu`, `LaplacianGpu` |
| S71 | ComputeDispatch batches 2–6: 34+ ops migrated to fluent builder. DF64 gamma/erf transcendentals |
| S78 | libc→rustix, AFIT migration, wildcard narrowing. Pure Rust ecosystem (zero C deps) |
| S79 | **Spring absorption** (neuralSpring V69): MultiHeadEsn, `SpectralAnalysis::from_eigenvalues(gamma)`, spectral_bandwidth/condition_number/classify_spectral_phase. esn_v2 readout shape fix. jackknife/boltzmann bitcast fixes. 5 ComputeDispatch migrations (71→76) |
| f97fc2ae | FFT `fft_1d.rs` ping-pong buffer fix. `ShaderTemplate::for_driver_auto` strips `enable f64;` (unblocks ~30 f64 shaders on naga path). `asin_df64` iterative confirmed |

**New wrapper APIs available** (not yet used by neuralSpring):

| API | Domain | Replaces |
|-----|--------|----------|
| `ops::bio::HillGateGpu` | Signal integration (021) | Local `hill_gate.wgsl` dispatch |
| `ops::bio::MultiObjFitnessGpu` | Directed evolution (014) | Local `multi_obj_fitness.wgsl` dispatch |
| `ops::bio::PairwiseL2Gpu` | MODES (012) | Local `pairwise_l2.wgsl` dispatch |
| `ops::bio::SwarmNnGpu` | Swarm robotics (015) | Local `swarm_nn_forward.wgsl` dispatch |
| `cpu_conv_pool::{conv2d, max_pool2d, avg_pool2d}` | LeNet-5 (Study 003) | Python-only conv2d/pool |

### Absorbed Upstream (Session 42, `5437c170` — generalized variants)

ToadStool absorbed 5 neuralSpring shaders, evolving them into generalized
upstream variants. Local copies retained for validation compatibility.

| Shader | Upstream | Binary | Checks | Key Differences |
|--------|----------|--------|--------|-----------------|
| `pairwise_l2.wgsl` | `barracuda::shaders::math::pairwise_l2` | `validate_gpu_modes` | 15/15 | Closed-form pair decoding vs linear search |
| `multi_obj_fitness.wgsl` | `barracuda::shaders::bio::multi_obj_fitness` | `validate_gpu_directed` | 6/6 | Bessel correction (n-1) vs population (n) |
| `hill_gate.wgsl` | `barracuda::shaders::bio::hill_gate` | `validate_gpu_signal` | 9/9 | Mode 0/1 generalization vs 2D-grid only |
| `swarm_nn_forward.wgsl` | `barracuda::shaders::bio::swarm_nn_forward` | `validate_gpu_swarm` | 9/9 | Generic MLP vs fixed 1→4→5, clamped sigmoid |
| `mean_reduce.wgsl` | `barracuda::shaders::reduce::mean_reduce` | `validate_gpu_pure_workload` | 7/7 | Effectively identical |

### Absorbed Upstream (Pre–Session 39, `77f70b2e` — identical copies)

| Shader | Upstream API | Binary | Checks |
|--------|-------------|--------|--------|
| `hmm_forward_log.wgsl` | `barracuda::ops::bio::hmm` | `validate_gpu_hmm_forward` | 13/13 |
| `batch_fitness_eval.wgsl` | `barracuda::ops::bio::batch_fitness` | `validate_gpu_batch_fitness` | 20/20 |
| `rk4_parallel.wgsl` | `barracuda::ops::rk_stage` | `validate_gpu_rk4` | 8/8 |
| `pairwise_jaccard.wgsl` | `barracuda::ops::bio::pairwise_jaccard` | `validate_gpu_pangenome` | 6/6 |
| `pairwise_hamming.wgsl` | `barracuda::ops::bio::pairwise_hamming` | `validate_gpu_sate` | 5/5 |
| `locus_variance.wgsl` | `barracuda::ops::bio::locus_variance` | `validate_gpu_meta_pop` | 7/7 |
| `spatial_payoff.wgsl` | `barracuda::ops::bio::spatial_payoff` | `validate_gpu_game_theory` | 5/5 |
| `batch_ipr.wgsl` | `barracuda::spectral::batch_ipr` | `validate_gpu_anderson` | 5/5 |

### Still Local (pending absorption)

| Shader | Domain | Binary | Checks | Absorption Target |
|--------|--------|--------|--------|-------------------|
| `xoshiro128ss.wgsl` | Stochastic (PRNG) | `validate_gpu_prng` | 5/5 | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | Swarm (015) | `validate_gpu_pipeline_swarm` | PASS | No upstream equivalent |
| `logsumexp_reduce.wgsl` | HMM/phylo (016–018) | `validate_gpu_logsumexp` | 5/5 | `barracuda::ops::reduce` (batched, Session 43) |
| `stencil_cooperation.wgsl` | Game theory (019) | `validate_gpu_stencil` | 3/3 | `barracuda::ops::stencil` (Session 43) |
| `rk45_adaptive.wgsl` | Regulatory ODE (020–021) | `validate_gpu_rk45` | 6/6 | `barracuda::ops::ode` (Session 43) |
| `wright_fisher_step.wgsl` | PopGen (024–025) | `validate_gpu_wright_fisher` | 4/4 | `barracuda::ops::popgen` (Session 43) |

### WGSL exports (forge crate — single source of truth)

All 21 WGSL shaders are centralized in `metalForge/forge/src/shaders.rs`.
21/21 absorbed upstream (S-03b resolved: head_split/head_concat in `barracuda::ops::mha`).
Library modules re-export for backward compatibility:

| Forge Constant | Library Re-Export |
|----------------|-------------------|
| `forge::shaders::HMM_FORWARD_LOG` | `hmm::WGSL_HMM_FORWARD_LOG` |
| `forge::shaders::PAIRWISE_JACCARD` | `pangenome_selection::WGSL_PAIRWISE_JACCARD` |
| `forge::shaders::LOCUS_VARIANCE` | `meta_population::WGSL_LOCUS_VARIANCE` |
| `forge::shaders::BATCH_FITNESS_EVAL` | `evolved::WGSL_BATCH_FITNESS_EVAL` |
| `forge::shaders::RK4_PARALLEL` | `evolved::WGSL_RK4_PARALLEL` |
| `forge::shaders::MEAN_REDUCE` | `evolved::WGSL_MEAN_REDUCE` |
| `forge::shaders::SPATIAL_PAYOFF` | `game_theory::WGSL_SPATIAL_PAYOFF` |
| `forge::shaders::BATCH_IPR` | `anderson_localization::WGSL_BATCH_IPR` |
| `forge::shaders::PAIRWISE_HAMMING` | `sate_alignment::WGSL_PAIRWISE_HAMMING` |
| `forge::shaders::PAIRWISE_L2` | `modes::WGSL_PAIRWISE_L2` |
| `forge::shaders::MULTI_OBJ_FITNESS` | `directed_evolution::WGSL_MULTI_OBJ_FITNESS` |
| `forge::shaders::SWARM_NN_FORWARD` | `swarm_robotics::WGSL_SWARM_NN_FORWARD` |
| `forge::shaders::HILL_GATE` | `signal_integration::WGSL_HILL_GATE` |
| `forge::shaders::XOSHIRO128SS` | `rng::WGSL_XOSHIRO128SS` |

Binding layouts and dispatch geometry documented in `forge::bindings`.

### Shader binding layouts (for ToadStool absorption)

**hmm_forward_log.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> trans: array<f32>` (N×N transition log-probs)
- `@group(0) @binding(1)` — `var<storage, read> emiss: array<f32>` (N emission log-probs for current obs)
- `@group(0) @binding(2)` — `var<storage, read> prev_alpha: array<f32>` (N forward log-probs at t-1)
- `@group(0) @binding(3)` — `var<storage, read_write> next_alpha: array<f32>` (N forward log-probs at t)
- `@group(0) @binding(4)` — `var<uniform> params: HmmParams` (`{n_states: u32}`)
- Dispatch: `(n_states.div_ceil(256), 1, 1)` — one thread per state

**batch_fitness_eval.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> pop: array<f32>` (pop_size × genome_len)
- `@group(0) @binding(1)` — `var<storage, read> weights: array<f32>` (genome_len)
- `@group(0) @binding(2)` — `var<storage, read_write> fitness: array<f32>` (pop_size)
- `@group(0) @binding(3)` — `var<uniform> params: FitnessParams` (`{pop_size, genome_len}`)
- Dispatch: `(pop_size.div_ceil(256), 1, 1)` — one thread per individual

**rk4_parallel.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read_write> state: array<f32>` (n_systems × 4)
- `@group(0) @binding(1)` — `var<uniform> params: Rk4Params` (`{n_systems, n_steps, dt, ...}`)
- Dispatch: `(n_systems.div_ceil(256), 1, 1)` — one thread per ODE system

**mean_reduce.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> values: array<f32>` (N values)
- `@group(0) @binding(1)` — `var<storage, read_write> result: array<f32>` (1 scalar)
- `@group(0) @binding(2)` — `var<uniform> params: ReduceParams` (`{n: u32}`)
- Dispatch: `(1, 1, 1)` — single workgroup (validation size)

**pairwise_jaccard.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> pa: array<f32>` (n_genes × n_genomes PA matrix)
- `@group(0) @binding(1)` — `var<storage, read_write> distances: array<f32>` (n_pairs)
- `@group(0) @binding(2)` — `var<uniform> params: JaccardParams` (`{n_genomes, n_genes}`)
- Dispatch: `(n_pairs.div_ceil(256), 1, 1)` — one thread per genome pair

**locus_variance.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> allele_freqs: array<f32>` (n_pops × n_loci)
- `@group(0) @binding(1)` — `var<storage, read_write> per_locus_var: array<f32>` (n_loci)
- `@group(0) @binding(2)` — `var<uniform> params: VarianceParams` (`{n_pops, n_loci}`)
- Dispatch: `(n_loci.div_ceil(256), 1, 1)` — one thread per locus

**spatial_payoff.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> grid: array<u32>` (grid_size² strategies)
- `@group(0) @binding(1)` — `var<storage, read_write> fitness: array<f32>` (grid_size²)
- `@group(0) @binding(2)` — `var<uniform> params: PayoffParams` (`{grid_size, b_x1000, c_x1000, _pad}`)
- Dispatch: `(grid_size².div_ceil(256), 1, 1)` — one thread per cell, Moore neighborhood

**batch_ipr.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> eigenvectors: array<f32>` (n_vectors × dim)
- `@group(0) @binding(1)` — `var<storage, read_write> ipr_out: array<f32>` (n_vectors)
- `@group(0) @binding(2)` — `var<uniform> params: IprParams` (`{dim, n_vectors}`)
- Dispatch: `(n_vectors.div_ceil(256), 1, 1)` — one thread per eigenvector

**pairwise_hamming.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> sequences: array<u32>` (n_seqs × seq_len)
- `@group(0) @binding(1)` — `var<storage, read_write> distances: array<f32>` (n_pairs)
- `@group(0) @binding(2)` — `var<uniform> params: HammingParams` (`{n_seqs, seq_len}`)
- Dispatch: `(n_pairs.div_ceil(256), 1, 1)` — one thread per sequence pair

---

## Tier B — Planned (needs design work)

| Shader | Domain | Priority | Status |
|--------|--------|----------|--------|
| `tridiag_eigensolver.wgsl` | Spectral (022–023) | P3 | Pending: Householder → bisection design |
| `pairwise_distance.wgsl` | Alignment (017) | P4 | Pending: O(N²) dispatch geometry |
| ~~`stencil_cooperation.wgsl`~~ | ~~Game theory (019)~~ | ~~P5~~ | **BUILT** (Session 43) — 3/3 PASS |
| ~~`logsumexp_reduce.wgsl`~~ | ~~HMM/phylogenetics~~ | ~~P2~~ | **BUILT** (Session 43) — 5/5 PASS |

---

## Tier C — New BarraCUDA Primitives Suggested

From our 25-paper analysis, these cross-cutting primitives would benefit
multiple Springs:

| Primitive | Use Case | Papers Served | Impact |
|-----------|----------|---------------|--------|
| `linalg::batch_matmul` | HMM forward/backward chain | 016–018 | Eliminate sequential dispatch |
| `ea::batch_fitness` | Population-parallel fitness | 011–015 | One dispatch per generation |
| `numerical::batch_rk45` | Multi-system ODE integration | 020–021 | Parallel biology simulation |
| `linalg::pairwise_distance` | O(N²) distance matrix | 017 | Alignment prerequisite |
| `ea::tournament_select` | GPU-parallel selection | 011–015 | Keep entire EA on GPU |
| `stencil::neighborhood_scan` | Spatial cooperation model | 019 | Reusable for any grid game |

---

## BarraCUDA APIs We Lean On

These are the native BarraCUDA APIs we depend on (via ToadStool absorption
or existing infrastructure):

| API | neuralSpring Use | Validated By |
|-----|------------------|-------------|
| `Tensor::from_data`, `to_vec` | All validation binaries | `validate_barracuda_tensor` (90/90) |
| `Tensor::layer_norm_wgsl` | ML inference validation | `validate_barracuda_tensor` (native, S-08 absorbed) |
| `Tensor::log_softmax_wgsl` | ML inference validation | `validate_barracuda_tensor` (native, S-09 absorbed) |
| `Tensor::leaky_relu_wgsl_with_slope` | Activation validation | `validate_barracuda_tensor` (S-05 absorbed) |
| `Tensor::elu_wgsl` | Activation validation | `validate_barracuda_tensor` (S-06 absorbed) |
| `ops::fft::{Fft1D, Ifft1D}` | f32 FFT validation | `validate_barracuda_fft` (12/12 f32) |
| `ops::fft::Fft1DF64` | f64 FFT (spectral, PPPM) | `validate_barracuda_fft` (8/8 f64, SHADER_F64) |
| `ops::fft::Rfft` | Real-to-complex FFT | `validate_barracuda_fft` (4/4 Rfft) |
| `ops::logsumexp::LogSumExp` | HMM log-domain | `validate_barracuda_logsumexp` (5/5) |
| `staging::StatefulPipeline` | Iterative GPU RK4 | `validate_gpu_stateful_pipeline` (10/10) |
| `dispatch::{dispatch_for, DispatchTarget}` | CPU/GPU parity | `validate_cross_dispatch` (8/8) |
| `WgpuDevice::new_cpu_relaxed` | CPU software adapter | `gpu.rs` (S-10 absorbed) |
| `stats::*`, `linalg::*`, `numerical::*`, `special::*` | 24 paper modules | 24 CPU port binaries (203/203) |

---

## BarraCUDA APIs — New in `5437c170` (Sessions 25–42)

### Now leveraged by neuralSpring

| API | Use | Status |
|-----|-----|--------|
| `ops::linalg::eigh_householder_qr` | `src/eigh.rs` delegates to upstream | **Wired** (S-12 absorbed) |
| `ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | Shader source for HMM GPU forward | **Wired** (shader absorbed) |
| `ops::bio::{PairwiseHammingGpu, PairwiseJaccardGpu, LocusVarianceGpu, SpatialPayoffGpu, BatchFitnessGpu}` | Shader sources re-exported via forge | **Wired** (shaders absorbed) |
| `spectral::BatchIprGpu` | Shader source re-exported via forge | **Wired** (shader absorbed) |
| `ops::rk_stage::WGSL_RK4_PARALLEL` | Shader source re-exported via forge | **Wired** (shader absorbed) |
| S-13 PooledBuffer race fix | Deferred return + device poll — flows via path dep | **Automatic** (Session 39) |
| TS-003 trig precision | 7-term Taylor + Cody-Waite range reduction | **Automatic** (Session 36) |
| TS-001 pow_f64 precision | Extended exp/log polynomials | **Automatic** (Session 36) |
| TS-004 FusedMapReduceF64 fix | Single command encoder for both passes | **Automatic** (Session 36) |

### Validated via BarraCUDA CPU binaries (Tier 3)

These APIs are already validated in dedicated Tier 3 (bC) binaries:

| API | Validated By | Checks |
|-----|-------------|--------|
| `ops::bio::HmmBatchForwardF64` | `validate_barracuda_hmm_f64` | 11/11 PASS |
| `spectral::{anderson_*, hofstadter_*, lanczos}` | `validate_barracuda_spectral_theory` | 17/17 PASS |
| `numerical::rk45_solve` | `validate_barracuda_regulatory`, `validate_barracuda_signal`, `validate_barracuda_game` | 20+ PASS |
| `ops::linalg::eigh_householder_qr` | `validate_eigh_accuracy` | 9/9 PASS |

Local Tier 2 (Rust native) implementations intentionally retained as independent
cross-validation references. Both tiers matching Python proves portability.

### Available for future leverage

| API | Potential Use | Status |
|-----|--------------|--------|
| Native `ops::mha::MultiHeadAttention` | `evolved::mha` thin wrapper | **Wired** (S-03b resolved upstream `0c998992`) |
| `ops::bio::{FelsensteinGpu, SmithWatermanGpu}` | Future paper extensions | Available |
| `ops::bio::GillespieGpu` | Stochastic SSA (Papers 013, 020) | **Wired** (Session 43, 20/20 PASS) |
| `ops::bio::{TaxonomyFcGpu, KmerHistogramGpu, UniFracPropagateGpu}` | wetSpring metagenomics | **Wired** (Session 43, 8/8 PASS) |
| `special::chi_squared::*` | Pangenome selection (Paper 024) | **Wired** (Session 43, 13/13 PASS) |
| `ops::bio::{RfBatchInferenceGpu, TreeInferenceGpu}` | Future ML forest workloads | Available |
| `ops::linalg::{InverseF64, LinSolveF64}` | GPU dense linear algebra | Available |
| `ReduceScalarPipeline::sum_f64` | Fitness aggregation | Available (local mean_reduce validated) |
| `BatchedRK4F64` | CPU-threaded ODE parameter sweeps | Available |
| `WGSL_BATCHED_EIGH_NAK_OPTIMIZED` | GPU-native eigensolve for Anderson | Available |
| `ops::conv2d::Conv2D` | Batched Conv2D — LeNet-5 conv layers | **New** (Session 39) — not yet wired to executor |
| `ops::maxpool2d::MaxPool2D` | MaxPool2D — LeNet-5 pooling | **New** (Session 39) — not yet wired to executor |
| `ops::avgpool2d::AvgPool2D` | AvgPool2D — alternative pooling | **New** (Session 39) — not yet wired to executor |
| `esn_v2::export_weights/import_weights` | GPU-train → NPU-deploy pipeline | **New** (Session 39) |
| `esn_v2::multi_head::MultiHeadEsn` | Multi-head ESN with per-head readout, uncertainty via head disagreement | **New** (S79) — available for multi-regime WDM evolution |
| `spectral::SpectralAnalysis::from_eigenvalues(gamma)` | One-call spectral analysis (bandwidth + cond + phase from eigenvalues + aspect ratio) | **New** (S79) — consumed via `spectral_bandwidth`/`spectral_condition_number` delegates |
| `spectral::classify_spectral_phase` | Spectral phase via MP outlier fraction (Bulk/EdgeOfChaos/Chaotic) | **New** (S79) — different scheme from local `classify_phase` (level spacing ratio) |
| `device::ComputeDispatch` builder | Fluent builder for GPU compute: `.shader()`, `.f64()`, `.storage_read()`, `.dispatch_1d()`, `.submit()` | **New** (S71+) — 76+ upstream ops migrated |

---

## Phase 5b — Full-Stack Validation (23 Domains, ALL GREEN)

BarraCUDA `Tensor` operations validated against CPU f64 references across
23 papers spanning all 7 validation tiers. S-14/S-15/S-16 **RESOLVED** upstream
(`a4996b34` S39). S-17 **RESOLVED** upstream (`c82c23d1` S58).
Validators retain conservative data generation patterns (positive-only data,
`rng.uniform() * 0.5 + 0.5` ensuring all elements ≥ 0.5.

| Validator | Domain | Papers | GPU Ops | Checks | Status |
|-----------|--------|--------|---------|--------|--------|
| `validate_barracuda_gpu_spectral` | Spectral commutativity | 022 | matmul | 10 | **PASS** |
| `validate_barracuda_gpu_eco` | Ecological dynamics | 013 | matmul, transpose | 6 | **PASS** |
| `validate_barracuda_gpu_hmm` | HMM phylogenetics | 016-018 | matmul, transpose | 5 | **PASS** |
| `validate_barracuda_gpu_fitness` | Evolutionary computation | 011-015 | matmul, transpose | 7 | **PASS** |
| `validate_barracuda_gpu_nn` | Neural nets | 015, 020-021 | matmul, transpose, tanh, add | 5 | **PASS** |
| `validate_barracuda_gpu_pairwise` | Pairwise distance | 017, 019, 024-025 | matmul, transpose | 5 | **PASS** (S-16 fixed) |
| `validate_barracuda_gpu_anderson` | Anderson localization | 023 | matmul, transpose | 7 | **PASS** (S-15 RESOLVED upstream) |
| `validate_barracuda_surrogate` | Surrogate MLP (Exp 001) | 001 | matmul, tanh | 7 | **PASS** |
| `validate_barracuda_transfer` | Transfer Learning (Exp 004) | 004 | matmul, tanh | 7 | **PASS** |
| `validate_barracuda_gpu_transformer` | Transformer (Exp 002) | 002 | matmul, transpose, tanh | 7 | **PASS** |
| `validate_barracuda_sequence` | Sequence (Exp 003) | 003 | matmul, tanh, sigmoid | 7 | **PASS** |
| `validate_barracuda_lenet` | LeNet-5 (Study 003) | S003 | matmul, tanh | 5 | **PASS** |
| `validate_barracuda_lstm` | LSTM (Study 004) | S004 | matmul, tanh, sigmoid | 6 | **PASS** |
| `validate_barracuda_bio_ops` | Upstream bio wrappers | 011-025 | BatchFitnessGpu, PairwiseHammingGpu, PairwiseJaccardGpu, LocusVarianceGpu, SpatialPayoffGpu, BatchIprGpu | 12 | **PASS** |
| `validate_barracuda_hmm_f64` | Upstream HMM f64 batch | 016-018 | HmmBatchForwardF64 (wetSpring) | 11 | **PASS** |

### Cross-dispatch Validators (xD — 15/15 Phase 0++ papers)

| Validator | Papers Covered | Checks | Status |
|-----------|---------------|--------|--------|
| `validate_cross_dispatch` | 011-015 | 8 | **PASS** |
| `validate_cross_dispatch_genomics` | 016-018 | 8 | **PASS** |
| `validate_cross_dispatch_extended` | 019-025 | 12 | **PASS** |
| `validate_cross_dispatch_phase4e` | PINN, DeepONet | 9 | **PASS** |
| `validate_cross_dispatch_hmm` | 016, 018 | 4 | **PASS** |
| `validate_cross_dispatch_ode` | 020 | 4 | **PASS** |

### Shortcoming Resolution

| # | Shortcoming | Severity | Root Cause | Resolution |
|---|-------------|----------|------------|------------|
| S-14 | Naive matmul hang (small square matrices) | Medium | Driver/binary complexity interaction | **RESOLVED** upstream (`a4996b34` S39: Naive tier removed) |
| S-15 | Matmul hang when f32 elements ≤ 0.1 magnitude | Critical | WGPU/Vulkan driver bug (RTX 4070) | **RESOLVED** upstream (`a4996b34` S39) |
| S-16 | 2D transpose dispatch uses divisor 256 vs tile 16 | High | `optimal_workgroup_size(ElementWise)` | **RESOLVED** upstream (`a4996b34` S39: `const TILE: u32 = 16`) |

Full details: `wateringHole/handoffs/`

---

## S-12: Absorbed Upstream (`77f70b2e`)

neuralSpring's Householder+QR eigensolver was absorbed by ToadStool as
`barracuda::ops::linalg::eigh_householder_qr`. `src/eigh.rs` now delegates
to upstream. The local fossil is preserved at `metalForge/fossils/evolved_s01_s11/eigh_local.rs`.

Validated by `validate_eigh_accuracy` (9/9 PASS). ToadStool also added
NAK-optimized GPU eigensolve shaders (`WGSL_BATCHED_EIGH_NAK_OPTIMIZED`).

---

## Code Quality (Post-Deep-Audit, February 23 2026)

| Aspect | Status |
|--------|--------|
| `cargo fmt` | **Clean** — zero formatting violations |
| `cargo clippy` pedantic + nursery | **0 warnings** — all `#[allow]` migrated to `#[expect(, reason)]` (0 in production code; 6 in `#[cfg(test)]` where `expect_used`/`unwrap_used` don't fire) |
| `cargo doc --no-deps` | **0 warnings** — all rustdoc links valid |
| `cargo test --lib` | **1152 tests PASS** |
| `cargo test --test integration` | **9 integration tests PASS** |
| `#[must_use]` | Applied to 24+ pure public functions across 5 modules |
| `#![forbid(unsafe_code)]` | Enforced at crate root — zero `unsafe` blocks permitted |
| Centralized tolerances | Split into `tolerances/` module (`mod.rs` + `gpu.rs` + `training.rs` + `registry.rs`) — 180+ `NamedTolerance` entries across 10 categories, zero inline magic numbers in production code |
| GPU validation helpers | Shared `gpu_readback`, `max_abs_diff_gpu_vs_cpu`, `gpu_tensor!` macro — deduplicated ~400 LOC from 24 binaries |
| GPU device init | Unified via `Gpu::new()` (removed ~800 LOC duplication) |
| Modular `gpu_ops/` | Refactored from monolithic 1328-line file into 6 focused submodules (`linalg`, `activation`, `reduction`, `bio`, `population`, `eigensolver`) — all under 1000 LOC |
| GPU dispatch coverage | `Dispatcher` CPU-fallback paths: **33 tests** covering all 26 dispatched operations |
| `GpuCapabilities` tested | Mock-based unit tests for `workgroup_size`, `dispatch_count`, `supports_workgroup` — no GPU required |
| Idiomatic Rust | HMM flat row-major layout, spectral flat layout, `NkLandscape.k` accessor, `mul_add` for FMA, infallible casts via `From` |
| Consolidated math primitives | Shannon, Hill, sigmoid, RK4 centralized in `primitives.rs` — no duplicated math |
| GPU-ready flat layouts | HMM, spectral, anderson_localization, directed_evolution, sate_alignment use flat row-major `Vec<f64>` — direct GPU buffer upload |
| Graceful error handling | `require!` macro replaces `.expect()` in all validation binaries — no panic on GPU failure |
| Zero-copy genotype handling | `eco_dynamics.rs` uses `&[u8]` / `HashSet<&[u8]>` — avoids `Vec<u8>` clones |
| Provenance | All hardcoded validation targets sourced with script, commit, date, exact command |
| Determinism tests | **16 tests** covering all stochastic modules (up from 7) |
| SPDX headers | All 40 Python/shell files have `AGPL-3.0-or-later` license identifier |
| Line coverage | **91.66%** line via `cargo llvm-cov` (remaining gap: GPU-only code paths unreachable on CPU) |
| All files < 1000 LOC | Largest bin: `validate_modern_cross_spring.rs` (949), largest lib: `glucose_prediction.rs` (812) |
| `unsafe` | Forbidden (`#![forbid(unsafe_code)]`) |
| Mocks/stubs | Zero in production code — zero `todo!`/`unimplemented!` |
| External dependencies | All pure Rust — zero C/C++ wrapper crates |
| Centralized tolerances | `tolerances/` module — **141+ `NamedTolerance`** entries across 10 categories including `domain_guards` |
| Magic numbers eliminated | All production `1e-10`/`1e-12` constants centralized via `tolerances::` |
| Cast safety | `cpu_fallback.rs` activator indices bounds-checked via `safe_idx()` |
| coralForge rename | `sovereign_folding` + `structure_module` → unified `coral_forge/` with `structure/` submodule |

---

## Dependency Analysis

All external dependencies are pure Rust with no C/C++ bindings:

| Crate | Version | Role | Evolution Path |
|-------|---------|------|----------------|
| `barracuda` | path (v0.3.5 at `0649cd0`) | GPU compute abstraction (in-house) | Standalone barraCuda primal (extracted from ToadStool S89) |
| `neural-spring-forge` | path | Shader catalog (in-house) | Evolves with metalForge |
| `biomeos-primal-sdk` | path (opt) | Primal IPC framework (in-house) | Evolves with biomeOS |
| `bytemuck` | 1.21 | Zero-copy GPU buffer casting | Stable, pure Rust, no alternative needed |
| `serde` + `serde_json` | 1 | JSON baseline I/O | Stable ecosystem standard |
| `tokio` | 1.49 | Async runtime for wgpu | Required by wgpu device creation |
| `wgpu` | 28 | WebGPU abstraction | Must match barracuda's version |
| `anyhow` | 1 (opt) | Error handling in primal | Primal-only, lightweight |
| `uuid` | 1 (opt) | Request IDs in primal | Primal-only |
| `chrono` | 0.4 (opt) | Timestamps in primal | Primal-only |
| `log` + `env_logger` | 0.4/0.11 (opt) | Primal logging | Primal-only |
| `approx` | 0.5 (dev) | Float comparison in tests | Test-only |

**No dependencies require evolution to Rust** — all are already pure Rust crates.
The optional primal deps (`anyhow`, `uuid`, `chrono`, `log`, `env_logger`) are gated behind the `primal` feature and only affect the `neuralspring_primal` binary.

---

## Full Validation Stack (7 Tiers × 25 Papers)

The validation progression proves math portability at each level:

```
Tier 1 (Py)  → Open data + Python: reproducible science baseline
Tier 2 (Rs)  → Rust native: same math, type-safe, deterministic
Tier 3 (bC)  → BarraCUDA CPU: proves Rust math matches via barracuda primitives
Tier 4 (gT)  → BarraCUDA GPU Tensor: proves math is portable CPU → GPU
Tier 5 (mF)  → metalForge WGSL: domain-specific GPU kernels, validated vs CPU
Tier 6 (gP)  → GPU Pipeline: end-to-end multi-kernel GPU chains
Tier 7 (xD)  → Cross-dispatch: CPU ↔ GPU parity via dispatch routing
```

| Tier | Coverage | Status |
|------|----------|--------|
| Py (Python) | 25/25 (100%) | **ALL PASS** |
| Rs (Rust) | 25/25 (100%) | **ALL PASS** |
| bC (BarraCUDA CPU) | 24/25 (96%) | **ALL GREEN** |
| gT (GPU Tensor) | 23/25 (92%) | **ALL GREEN** |
| mF (metalForge WGSL) | 14/25 (56%) | **ALL PASS** |
| gP (GPU Pipeline) | 7/25 (28%) | **ALL PASS** |
| xD (Cross-dispatch) | 15/15 (100%) | **ALL GREEN** |

---

## Cross-Spring Shader Evolution Lineage

The ToadStool/BarraCuda ecosystem benefits from cross-spring evolution: each
Spring contributes domain-specific shaders that are generalized into the shared
crate, then consumed by all Springs. This table tracks provenance.

### hotSpring → BarraCuda → neuralSpring (Precision & Physics)

| Primitive | hotSpring Origin | BarraCuda Location | neuralSpring Use |
|-----------|------------------|-------------------|-----------------|
| Taylor-series trig (sin/cos) | TS-003 7-term Taylor + Cody-Waite | `special::trig` | Spectral theory (17/17 checks) |
| Extended exp/log polynomials | TS-001 pow_f64 fix | `special::erf`, `math::exp/log` | Anderson localization, PINN |
| Lanczos eigensolver | hotSpring v0.5.16 lattice QCD | `spectral::lanczos` | Spectral 5/5, Hofstadter 5/5 |
| HFB deformed | hotSpring nuclear physics | `ops::physics::hfb_deformed` | Cross-validated (Session 39 absorption) |
| 19 new f64 WGSL shaders (S42) | chi_squared, factorial, rk45, cubic_spline | `shaders/special/*`, `shaders/math/*` | Available for GPU promotion |

### wetSpring → BarraCuda → neuralSpring (Bio/Genomics)

| Primitive | wetSpring Origin | BarraCuda Location | neuralSpring Use |
|-----------|------------------|-------------------|-----------------|
| HMM batch forward f64 | wetSpring phylogenetics | `ops::bio::hmm` | HMM validation 11/11, GPU HMM 13/13 |
| Quality filter | wetSpring FASTQ pipeline | `ops::bio::quality_filter` | bC genomics validation |
| DADA2 E-step | wetSpring amplicon denoising | `ops::bio::dada2` | Cross-dispatch genomics |

### neuralSpring → BarraCuda → all Springs (ML & Evolution)

| Primitive | neuralSpring Origin | BarraCuda Location | Beneficiary |
|-----------|---------------------|-------------------|-------------|
| batch_fitness_eval | Paper 011-015 (ML) | `ops::bio::BatchFitnessGpu` | wetSpring, hotSpring |
| pairwise_hamming | Paper 017 (SATé) | `ops::bio::PairwiseHammingGpu` | wetSpring genomics |
| pairwise_jaccard | Paper 024 (Pangenome) | `ops::bio::PairwiseJaccardGpu` | wetSpring metagenomics |
| spatial_payoff | Paper 019 (Game Theory) | `ops::bio::SpatialPayoffGpu` | Ecological modeling |
| locus_variance | Paper 025 (MetaPop) | `ops::bio::LocusVarianceGpu` | Population genetics |
| batch_ipr | Paper 022-023 (Anderson) | `spectral::BatchIprGpu` | hotSpring condensed matter |
| hill_gate | Paper 021 (Signal) | `ops::bio::HillGateGpu` | Regulatory network modeling |
| multi_obj_fitness | Paper 014 (Directed Evo) | `ops::bio::MultiObjFitnessGpu` | Optimization pipelines |
| pairwise_l2 | Paper 012 (MODES) | `ops::bio::PairwiseL2Gpu` | Novelty search, clustering |
| swarm_nn_forward | Paper 015 (Swarm) | `ops::bio::SwarmNnGpu` | Neuroevolution controllers |
| Householder+QR eigensolver | `eigh.rs` | `linalg::sparse::eigh` | hotSpring, wetSpring |
| 4-tier matmul KernelRouter | S-14/S-15 **RESOLVED** | `ops::matmul` | All Springs |
| Capability-based dispatch | `Gpu::dispatch_1d` | Pattern adopted | All Springs |

### Upstream Parity Benchmark (10 Kernels, RTX 4070)

| Kernel | Origin | Local µs | Upstream µs | Ratio |
|--------|--------|----------|-------------|-------|
| BatchFitness 10K×32 | neuralSpring 011-015 | 3153 | 2346 | 0.74× |
| Hamming 200×500 | neuralSpring 017 (SATé) | 4396 | 3388 | 0.77× |
| Jaccard 100×500 | neuralSpring 024 (Pangenome) | 2269 | 2272 | 1.00× |
| LocusVariance 50×500 | neuralSpring 025 (MetaPop) | 2270 | 2284 | 1.01× |
| SpatialPayoff 256² | neuralSpring 019 (GameTheory) | 2284 | 2266 | 0.99× |
| BatchIPR 1K×256 | neuralSpring 022-023 (Anderson) | 3150 | 2259 | 0.72× |
| **HillGate 100×100** | **neuralSpring 021 (Signal)** | **2236** | **2279** | **1.02×** |
| **MultiObjFitness 5K×4** | **neuralSpring 014 (DirEvo)** | **2432** | **2358** | **0.97×** |
| **PairwiseL2 200×50** | **neuralSpring 012 (MODES)** | **2271** | **2269** | **1.00×** |
| **SwarmNN 500×20** | **neuralSpring 015 (Swarm)** | **2279** | **2513** | **1.10×** |

All 10 upstream wrappers show negligible overhead (0.72–1.10×).
Bold entries are newly wired in Session 42 ToadStool sync.

### Session 50 — baseCamp Biophysical AI Interpretability (82/82 PASS)

5 new library modules implementing cross-domain analysis of AI systems using
validated physics/biology primitives. Each module composes existing neuralSpring
primitives (`eigh`, `anderson_localization`, `hmm`, `game_theory`) into novel
analysis pipelines. 861 unit tests, 0 clippy warnings, 0 doc warnings.

| Module | Sub-thesis | Checks | Key Primitives |
|--------|-----------|--------|----------------|
| `weight_spectral` | nS-01: Weight Hamiltonians | 15/15 | ESD, IPR, level spacing ratio, Marchenko-Pastur |
| `information_flow` | nS-02: Information Propagation | 15/15 | Depth scale, gate disorder, attention Hamiltonian |
| `loss_landscape` | nS-03: Loss Landscapes | 19/19 | Numerical Hessian, Boltzmann MCMC, spectral gap |
| `neural_pgm` | nS-04: Neural PGMs | 15/15 | Belief propagation, KL divergence, effective rank |
| `agent_coordination` | nS-05: Multi-Agent QS | 18/18 | Graph Laplacian, QS signaling, dimensional sweep |

**GPU promotion (Session 55):** All 4 candidates now have `Dispatcher` methods
routing to GPU or CPU fallback via `validate_basecamp_dispatch` (19/19 PASS).

**Upstream rewiring (Session 56 — ToadStool `9404fdb4`):** 4 functions now
delegate to upstream BarraCUDA, eliminating local implementations:

| Local Function | Upstream Module | Effect |
|----------------|----------------|--------|
| `graph_laplacian` | `barracuda::linalg::graph` | Thin wrapper → upstream |
| `disordered_laplacian` | `barracuda::linalg::graph` | Thin wrapper → upstream |
| `belief_propagation_chain` | `barracuda::linalg::graph` | Thin wrapper → upstream |
| `numerical_hessian` | `barracuda::numerical` | Thin wrapper → upstream |

Public API preserved; callers unchanged. Validated via `cargo test --lib` (861 PASS).

See `whitePaper/baseCamp/extensions.md` for the full research program.

### Session 49 — Code Quality Status

| Quality Gate | Status |
|--------------|--------|
| Hardcoded paths | **0** (all via `validation::baseline_path`) |
| TODO/FIXME/MOCK/STUB | **0** in src/ |
| `unsafe` blocks | **0** (`forbid` enforced) |
| `.unwrap()` in non-test | **0** |
| Clippy warnings | **0** (pedantic + nursery) |
| Doc warnings | **0** |
| Max file size | 965 lines (under 1000 wateringHole limit) |
| Dispatch pattern | 7 core methods delegate to upstream `domain_ops`; remainder use `gpu_or_cpu` |
| GPU skip policy | All 79 binaries use `exit_no_gpu()` (CI-fidelity) |

### Session 56 — ToadStool S53 Sync + Upstream Rewiring

| Action | Detail |
|--------|--------|
| **Pulled ToadStool HEAD** | `f78cf3b0` (absorbed Sessions 51–53 handoffs) |
| **New upstream modules** | `barracuda::linalg::graph`, `barracuda::numerical`, `barracuda::ops::bio::swarm_nn`, `barracuda::ops::bio::xoshiro128ss` |
| **Rewired 4 functions** | `graph_laplacian`, `disordered_laplacian`, `belief_propagation_chain`, `numerical_hessian` → delegate to upstream |
| **3 new validators** | `validate_basecamp_dispatch` (19 checks), `validate_barracuda_parity` (34 checks), `validate_metalforge_pcie` (36 checks) |
| **Total checks** | 2010+ (206 Python + 1810+ Rust+GPU) |
| **Lib tests** | 861 PASS |
| **Forge tests** | 30 PASS |
| **Quality gates** | fmt ✓ · clippy ✓ (pedantic+nursery) · doc ✓ |

### Session 57 — ToadStool S58–S59 Sync

| Action | Detail |
|--------|--------|
| **Pulled ToadStool HEAD** | `9404fdb4` (S58: df64/Fp64Strategy/ODE bio/NMF; S59: anderson correlated/ridge/ValidationHarness) |
| **Confirmed absorptions** | `ValidationHarness`, `exit_no_gpu`, `require!` macro — all from neuralSpring, now in `barracuda::validation` |
| **Consolidated** | 4 duplicate `patch_pow_to_polyfill` → `validation::patch_pow_to_polyfill` (shared) |
| **New upstream available** | `barracuda::spectral::anderson` (3D correlated, sweep averaged, find_w_c), `barracuda::linalg::ridge`, `barracuda::linalg::nmf`, `barracuda::numerical::ode_bio`, `barracuda::dispatch::domain_ops`, `barracuda::device::driver_profile` |
| **Quality gates** | fmt ✓ · clippy ✓ (pedantic+nursery) · 861 lib ✓ · 202/202 validate_all |

### Session 58 — Upstream Dispatch Rewiring + GpuDriverProfile

| Action | Detail |
|--------|--------|
| **Rewired 7 Dispatcher methods** | `mat_mul`, `frobenius_norm`, `transpose`, `softmax`, `l2_distance`, `mean`, `variance` → delegate to `barracuda::dispatch::domain_ops` |
| **Wired GpuDriverProfile** | `Dispatcher` now exposes `driver_profile()`, `fp64_strategy()`, `needs_pow_workaround()` via upstream `barracuda::device::driver_profile` (hotSpring-evolved) |
| **Driver detection confirmed** | RTX 4070: Ada arch, NvidiaPtxas compiler, Throttled FP64 → Hybrid strategy, pow workaround needed |
| **New validator** | `validate_cross_spring_evolution` (10/10 PASS): rewired method parity + driver profile + cross-spring benchmark |
| **Total rewired functions** | 11 (4 from S56 + 7 from S58) — all delegating to upstream BarraCUDA |
| **Quality gates** | fmt ✓ · clippy ✓ (pedantic+nursery) · 861 lib ✓ · 202/202 validate_all |

### Session 61 — Deep Code Quality Sweep (February 25, 2026)

| Action | Detail |
|--------|--------|
| **Deep code quality sweep** | Property tests, tolerance centralization, vestigial allow removal |
| **13 property tests added** | `src/property_tests.rs` — invariants across stochastic and numerical modules |
| **6 tolerance constants centralized** | Added to `tolerances/` registry |
| **4 vestigial `#[allow]` attributes removed** | Underlying code fixed, redundant suppression removed |
| **Line coverage** | **90%+** via `cargo llvm-cov` |
| **Lib tests** | **861 PASS** |

### Session 66 — Phase C GPU Promotion (February 25, 2026)

6 new `Dispatcher` methods, 3 new `gpu_ops` functions. HMM forward/Viterbi chains,
pairwise/global FST, inter-population AF variance — all now GPU-dispatchable.
`validate_gpu_phase_c` 18/18 PASS. GPU coverage: ~90% → ~97% of production math.

|| Session 66: Phase C GPU promotion | 6 Dispatcher methods, 3 gpu_ops, validate_gpu_phase_c 18/18 | **~97% GPU** |
|| Session 66: Python baselines | 25/25 PASS — zero drift, 83.6× Rust faster (geomean) | **ALL GREEN** |

### Session 67 — CPU Math Parity Validation (February 25, 2026)

Cross-language parity: `control/generate_cpu_references.py` → JSON →
`validate_cpu_math_parity` 39/39 PASS (9 primitives + 9 paper kernels + 6
Dispatcher cpu_only). All within 1e-10 tolerance. Proves BarraCUDA CPU = Python/NumPy.

|| Session 67: CPU↔Python parity | `validate_cpu_math_parity` 39/39 PASS (1e-10) | **ALL GREEN** |

### Session 67b — Dispatch Tier Benchmarks (February 25, 2026)

Three-tier benchmark: Library direct → Dispatcher::cpu_only() → Dispatcher::new() GPU.
9/10 ops ≤1.04× CPU dispatch overhead. Per-call GPU driver-bound for small workloads —
motivates StatefulPipeline/UnidirectionalPipeline batching for GPU-resident acceleration.

|| Session 67b: Dispatch tiers | `bench_dispatch_tiers` — 9/10 ops ≤1.04× CPU overhead | **Transparent** |

### Session 68 — Deep Debt Audit (February 25, 2026)

Full barracuda usage audit: 90+ import sites, 20+ submodules, zero duplicates.
Tolerance centralization: 104+ named constants, zero ad-hoc magic numbers.
Rewired `boltzmann_sampling` → `barracuda::sample::boltzmann_sampling` (17th function rewire).
861 lib tests, 90%+ coverage.

|| Session 68: Deep debt audit | 104+ tolerances, 90%+ coverage, 0 debt markers | **ALL GREEN** |
|| Session 68: boltzmann rewire | 17th function rewired to upstream | **LEAN** |

### Session 69 — Validator Shader Rewiring + Cross-Spring Benchmarks (February 25, 2026)

6 validator binaries rewired from local `include_str!` to upstream barracuda shader
constants. Cross-spring benchmarks refreshed. Upstream-vs-local: 10/10 ≈ or ~ (zero ⚠).
Complete cross-spring provenance mapped: hotSpring precision, wetSpring bio, neuralSpring ML.

|| Session 69: Shader source rewire | 6 validators → upstream barracuda constants | **LEAN** |
|| Session 69: Cross-spring bench | 10/10 upstream ≈ local, 39/39 evolution PASS | **ALL GREEN** |
|| Session 69: validate_all | 202/202 PASS | **ALL GREEN** |

### Session 80 — Comprehensive Debt Audit and Coverage Expansion (February 26, 2026)

Full codebase audit followed by systematic debt resolution. All inline magic numbers
promoted to named tolerances. Validation binary error handling evolved from `unwrap()`
to graceful Result-based flow. Low-coverage modules brought above 90% target.
Shared validation helpers extracted for reuse across binaries.

| Action | Detail |
|--------|--------|
| **WDM EOS provenance** | Added `WDM_EOS_PROVENANCE` record with script, commit, date, command, environment |
| **Tolerance evolution** | 4 inline `1e-30` guards → `tolerances::LOG_ZERO_GUARD` (reduction, population, wdm_surrogate) |
| **Tolerance documentation** | Derivation annotations for `LOG_ZERO_GUARD`, `SWARM_FITNESS_COMPARISON`, `KAPPUS_WEGNER_REL` |
| **Coverage: wdm_surrogate** | 43.3% → 97.6% — 14 new tests (JSON parsing, edge cases, error paths) |
| **Coverage: basecamp** | 48.7% → 90.6% — 12 new tests (landscape, spectral, propagation, belief, interaction) |
| **Binary evolution** | `validate_barracuda_wdm_eos`: 16 `unwrap()` → `Result<Vec<f32>, String>` via `gpu_mlp_forward` |
| **Shared helpers** | `validate_tensor_unary` + `validate_tensor_reduction` extracted to `validation.rs` |
| **Binary refactoring** | `validate_barracuda_tensor.rs`: 966 → 911 lines via shared helpers |
| **Baselines script** | Added WDM EOS + ML inference; enhanced with git commit, tree state, dep versions |
| **CI evolution** | `baselines.yml`: artifact upload. `rust.yml`: cross-validation job (Python + Rust parity) |
| **Quality gates** | fmt ✓ · clippy ✓ (pedantic+nursery) · 861 lib ✓ · doc ✓ · 90%+ coverage |

|| Session 80: Debt audit | 861 lib tests, 90%+ coverage, zero inline magic numbers | **ALL GREEN** |
|| Session 80: Coverage | wdm_surrogate 97.6%, basecamp 90.6% (both 90%+ target) | **COMPLETE** |
|| Session 80: Binary evolution | 16 unwrap → Result, 2 shared validation helpers | **EVOLVED** |

### Session 81 — Deep Debt Evolution (February 26, 2026)

25 new named tolerance constants added (spectral, population genetics, game theory,
quantization, GPU commutator, hardware dispatch). 21 validation binaries swept to
replace ~50 inline magic numbers. `spectral_entropy` rewired to
`barracuda::stats::shannon_from_frequencies` (39th function rewire). metalForge
`probe.rs` gated behind `#[cfg(target_os = "linux")]` for cross-platform.
7 PyTorch scripts gained full deterministic seeding.

|| Session 81: Deep debt evolution | 129+ tolerances, 39th barracuda rewire, cross-platform | **ALL GREEN** |

### Session 82 — Titan V Pure Rust Pipeline Validation (February 26, 2026)

Full pure Rust GPU pipeline validated on NVIDIA TITAN V (NVK GV100, Volta SM70).
Fixed `fma(f64)` WGSL spec violation in `batched_eigh_nak_optimized_f64.wgsl` —
Sovereign Compiler re-fuses `a * b + c` into `OpFMulAdd` at IR level. Explicit
f64 typing for bare float literals in `select()` and division contexts. 33 validation
binaries, 384/384 GPU checks PASS. Zero RTX 4070 regressions. 861/861 lib tests.

|| Session 82: Titan V validation | 384/384 GPU checks, fma(f64) shader fix, zero regressions | **ALL GREEN** |

### Session 83 — ToadStool S68 Universal Precision Sync (February 26, 2026)

ToadStool S68 evolved all 700 shaders to f64 canonical with runtime downcast via
`LazyLock<String>`. This privatized 3 shader constants neuralSpring re-exported,
renamed 1 (`rk4_parallel.wgsl` → `rk4_parallel_f64.wgsl`), and changed 2 types
(`pub const &str` → `LazyLock<String>`). Fixed all imports: 3 switched to local
copies, 1 to new f64 pub const, 1 local f32 copy retained (f64 requires polyfill
injection), 2 validator binaries rewired to forge constants. API gap #3
(`variance_ddof`) closed upstream. All 14 ToadStool HEAD references updated.

|| Session 83: ToadStool S68 sync | 861/861 lib, 43/43 forge, 150/150 validators, 0 clippy | **ALL GREEN** |

### Session 95 — WDM + AlphaFold3 GPU Tensor Validators + Drift Fix (February 28, 2026)

4 new BarraCUDA GPU Tensor validators proving GPU math portability for WDM
surrogates (nW-01 transport, nW-03 S(q,ω), nW-05 ESN) and AlphaFold3 confidence
heads (nF-03 Phase C pLDDT/PAE/pDE). Python baseline drift fully resolved:
isomorphic catalog shader name mappings (20% → 100% BarraCUDA coverage) and
4 control scripts fixed for path resolution via `Path(__file__).parent`.

| Action | Detail |
|--------|--------|
| **3 WDM GPU validators** | Transport MLP (matmul/add/relu), ESN recurrence (matmul/add/tanh/argmax), SQW LSTM (LstmGpuWeights struct) |
| **AlphaFold3 confidence GPU** | pLDDT (sigmoid), PAE/pDE (matmul + CPU-side softmax/expected distance) |
| **Python drift fix** | Isomorphic catalog: full BarraCUDA shader name resolution. 4 path fixes. |
| **validate_all** | 232 binaries |

|| Session 95: WDM+AF3 GPU validators | 4 new GPU Tensor validators, 39/39 Python drift PASS, 861 lib, 0 clippy | **ALL GREEN** |

### Session 104 — ToadStool f97fc2ae Sync (March 1, 2026)

Synced with ToadStool HEAD `f97fc2ae` (3 commits since `8dc01a37`). Rewired spectral
analysis functions to upstream delegates. Verified all local shaders, API surfaces,
and tests against the evolved barracuda crate.

| Action | Detail |
|--------|--------|
| **Pulled ToadStool HEAD** | `f97fc2ae` — S78 (libc→rustix, AFIT migration), S79 (Spring absorption, MultiHeadEsn, spectral extensions), FFT buffer fix + `enable f64` naga strip |
| **Blocking bug: jackknife bitcast** | **FIXED** upstream (S79) — `jackknife_mean_f64.wgsl` f64 params moved to storage buffers for DF64 safety |
| **Blocking bug: `enable f64` naga** | **FIXED** upstream (f97fc2ae) — `ShaderTemplate::for_driver_auto` strips `enable f64;` lines, unblocking ~30 f64 shaders on naga fallback path |
| **Blocking bug: FFT buffer** | **FIXED** upstream (f97fc2ae) — `fft_1d.rs` ping-pong buffer selection corrected for odd-stage FFTs |
| **`asin_df64` iterative** | **CONFIRMED** upstream — coral forge GPU pipeline SDPA/IPA/backbone/torsion 16/16 PASS |
| **Rewired `spectral_bandwidth`** | `weight_spectral.rs` → delegates to `barracuda::spectral::spectral_bandwidth` (absorbed from neuralSpring V69 handoff) |
| **Rewired `spectral_condition_number`** | `weight_spectral.rs` → delegates to `barracuda::spectral::spectral_condition_number` (absorbed from neuralSpring V69 handoff) |
| **Retained `classify_phase`** | Local `SpectralPhase` (Extended/Critical/Localized) retained — upstream `classify_spectral_phase` uses different scheme (MP outlier fraction → Bulk/EdgeOfChaos/Chaotic) |
| **New upstream available** | `barracuda::spectral::SpectralAnalysis::from_eigenvalues(gamma)`, `barracuda::spectral::SpectralPhase`, `barracuda::esn_v2::multi_head::MultiHeadEsn` |
| **ComputeDispatch migrations** | 5 upstream ops (boltzmann, multinomial, diversity_fusion, elementwise_f64, earth_mover) migrated to builder — flows via path dep |
| **Shader absorption map** | 24/42 local WGSL now have upstream equivalents; 18 remain local (4 truly unique: HEAD_SPLIT, HEAD_CONCAT, XOSHIRO128SS, SWARM_NN_SCORES) |
| **Total rewired functions** | 15 (11 from S56-S81 + 2 from S104 + 2 from S104b: chi_squared_gpu, kl_divergence_gpu) — all delegating to upstream BarraCUDA |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings) · 861 lib tests ✓ · 0 regressions |

|| Session 104: ToadStool sync | f97fc2ae sync, 2 spectral rewires, 3 blocking bugs fixed, 861 lib, 0 clippy | **ALL GREEN** |

### Session 104b — Complete Rewiring + Cross-Spring Benchmark (March 2, 2026)

Deep rewiring pass: two fused GPU ops, all validation binaries migrated from
`include_str!` to forge constants, `Fp64Strategy::Concurrent` handling, and a
new cross-spring validation+benchmark binary documenting provenance of every
rewired path.

| Action | Detail |
|--------|--------|
| **Rewired `chi_squared_gpu`** | `gpu_ops/reduction.rs` → delegates to `barracuda::ops::fused_chi_squared_f64::FusedChiSquaredGpu::execute` (f64 fused single-dispatch shader). Was f32 multi-pass Tensor ops with CPU readback for division. Origin: neuralSpring `chi_squared_f64.wgsl` → ToadStool S76 absorption → fused upstream op |
| **Rewired `kl_divergence_gpu`** | `gpu_ops/reduction.rs` → delegates to `barracuda::ops::fused_kl_divergence_f64::FusedKlDivergenceGpu::execute` (f64 fused single-dispatch shader). Was CPU-computed with GPU sum only. Normalizes inputs to maintain backward compat |
| **Forge constant migration** | 12 `include_str!` → forge constants: `validate_mha_gpu.rs` (HEAD_SPLIT, HEAD_CONCAT), `validate_gpu_pure_workload.rs` (MEAN_REDUCE), `bench_upstream_vs_local.rs` (10 shader constants). Single source of truth for all shader references |
| **`Fp64Strategy::Concurrent`** | Added handling in `validate_cross_spring_evolution.rs` strategy match. Documented in `gpu_dispatch/mod.rs` — ToadStool S70++ enables DF64+native f64 side-by-side for precision cross-checking |
| **New validator** | `validate_toadstool_s79_rewire` — exercises all rewired paths with cross-spring provenance annotations: spectral (upstream delegates), chi²/KL (fused f64), entropy/variance/pearson (wetSpring→hotSpring→ToadStool), Fp64Strategy, weight spectral composition |
| **Total rewired functions** | **15** (13 from S56–S104 + 2 from S104b: chi_squared_gpu, kl_divergence_gpu) — all delegating to upstream BarraCUDA |
| **Cross-spring provenance** | hotSpring→f64 pipeline, VarianceReduceF64. wetSpring→FusedMapReduceF64, CorrelationF64. neuralSpring→chi², KL, spectral→ToadStool→FusedChiSquared, FusedKL (round-trip) |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings) · 861 lib tests ✓ · 9 integration ✓ · doc ✓ · 0 regressions |

|| Session 104b: Complete rewire | 2 fused GPU ops, 12 forge constants, Fp64Strategy::Concurrent, cross-spring benchmark | **ALL GREEN** |

### Session 105 — Deep Evolution: hotSpring Ingestion + Technical Debt (March 2, 2026)

Full-spectrum evolution pass: 5 hotSpring cross-spring ingestion items, systematic
technical debt resolution, and 5 large-file smart refactorings.

| Action | Detail |
|--------|--------|
| **MultiHeadWdmClassifier** | New `wdm_esn.rs` struct wrapping `barracuda::esn_v2::MultiHeadEsn` with 3 WDM-specific heads (Anderson regime label, Steering spectral bandwidth, Meta confidence). Typed JSON deserialization replaces manual `serde_json::Value` parsing. `head_disagreement()` for phase boundary uncertainty |
| **TrainingMonitor** | New `training_monitor.rs` — hotSpring `BrainInterrupt` pattern adapted for training: GREEN/YELLOW/RED attention FSM with spectral bandwidth, IPR collapse, and loss divergence triggers. `DriftMonitor` integration |
| **NPU export pipeline** | `MultiHeadWdmClassifier::export_npu_weights()` — int8 quantized via `barracuda::esn_v2::quantize_affine_i8_f64` for AKD1000 deployment |
| **Nautilus training bridge** | `SpectralNautilusBridge::observe_training_epoch()` — maps training loss + spectral metrics to `BetaObservation` for drift detection |
| **Dispatcher::kl_divergence** | Missing GPU+CPU fallback path added to `dispatch_stats.rs` via fused f64 WGSL + `counterdiabatic::kl_divergence` CPU fallback |
| **expect() → Result** | `gpu_shader_validation::dispatch_and_read` evolved to `Result<Vec<f64>, String>`, 15 callers updated |
| **Cast audit** | All `as f64` casts audited across 9 files — all from `usize` (no `From` impl), codebase already clean |
| **Primal protocol** | `lifecycle.*` → `nucleus.*` (register/heartbeat/deregister), SIGTERM handler, env-configurable `NEURALSPRING_IPC_TIMEOUT_SECS` and `NEURALSPRING_MAX_CONCURRENT` |
| **Refactor validation.rs** | 916 → `validation/{mod,stats,gpu,env}.rs` (4 focused submodules, 48 tests) |
| **Refactor provenance.rs** | 817 → `provenance/{mod,experiments,references}.rs` with `provenance!` macro (5 tests) |
| **Refactor weight_spectral.rs** | 715 → `weight_spectral/{mod,metrics,phase}.rs` (38 tests) |
| **Refactor meta_population.rs** | 595 → `meta_population/{mod,fst,geography}.rs` (12 tests) |
| **Refactor gpu_ops/bio.rs** | 662 → `gpu_ops/bio/{mod,hmm,activation,evolution}.rs` (7 tests) |
| **New validators** | `validate_multi_head_esn` (MultiHeadEsn + NPU + JSON), `validate_training_monitor` (FSM + interrupts + drift) |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings) · 861 lib tests ✓ · 226 binaries ✓ · doc ✓ · 0 regressions |
| **baseCamp Paper 12** | Sub-thesis 06 (B-16..B-21): Anderson localization in immunological signaling. `immunological_anderson.rs`: AD skin state classifier, Pielou evenness → disorder W, IC50 → barrier height, tissue geometry factor, `PharmacoMonitor`, Gonzales IC50 constants, lokivetmab PK data. +11 unit tests (861 total lib) |
| **nS-06 experiment buildout** | Python control: `control/immunological_anderson/immunological_anderson.py` (20/20 PASS, seed=42, JSON baseline). Rust validator: `validate_immunological_anderson` (53/53 PASS, cross-language parity at 1e-10). Provenance: `IMMUNOLOGICAL_ANDERSON_PROVENANCE`. Wired into `check_drift.sh` (40 baselines). GPU parity: `validate_basecamp_gpu` 18/18 (+4 nS-06: KL cytokine shift, Shannon entropy healthy/inflamed dermis). Dispatch: `validate_compute_dispatch` 19/19 (+3 nS-06: KL + entropy via Dispatcher). Mixed hardware: `validate_mixed_hardware` 21/21 (+7 nS-06: NUCLEUS tower eigensolve, node KL, nest entropy, PCIe NPU export cost) |

|| Session 105: Deep Evolution + baseCamp Paper 12 | 5 hotSpring ingestions, 5 smart refactors, 3 new validators, baseCamp Sub-thesis 06 (full experiment buildout: Py 20/20, Rs 53/53, GPU 4, dispatch 3, mixed 7), 861 lib, 226 bins, 0 clippy | **ALL GREEN** |

### Session 106 — baseCamp GPU Promotions + WDM Mixed-Hardware Gaps (March 2, 2026)

Systematic validation tier closure: baseCamp domain-specific GPU promotions,
WDM nW-03/nW-04 mixed-hardware coverage, and `validate_all` gap closure.

| Action | Detail |
|--------|--------|
| **`validate_barracuda_basecamp`** | New validator (26 checks): weight_spectral H² matmul parity, loss_landscape Hessian GPU eigensolve + spectral entropy + H×x product, neural_pgm HMM forward chain + single BP step matmul, agent_coordination pairwise L2 matrix + Laplacian eigensolve + chi², cross-module GPU determinism |
| **WS-GPU: weight_spectral promotion** | GPU matmul for Hamiltonian squared (H²), GPU eigensolve of H², non-negativity proof (σ²≥0), IPR parity, spectral variance parity — proves `weight_to_hamiltonian` math is GPU-portable |
| **LL-GPU: loss_landscape promotion** | Hessian eigensolve GPU parity, analytical eigenvalue verification (diagonal quadratic), spectral entropy, Hessian-vector product via GPU matmul — proves `numerical_hessian` spectral analysis is GPU-portable |
| **PGM-GPU: neural_pgm promotion** | HMM forward chain log-likelihood GPU parity, single belief propagation step via GPU matmul, KL(CPU_BP ∥ GPU_BP) near-zero — proves `belief_propagation_chain` is GPU-portable |
| **AC-GPU: agent_coordination promotion** | Pairwise L2 matrix GPU parity (16 agents × 3D), Laplacian eigensolve, smallest eigenvalue ≈ 0, chi² agent distribution — proves `interaction_graph` is GPU-portable |
| **nW-03 metalForge gap closure** | `validate_metalforge_wdm_coral` now includes SQW LSTM gate computation via `mixed_dispatch` (MixedSubstrate::GpuOnly routing) |
| **nW-04 metalForge gap closure** | `validate_metalforge_wdm_coral` now includes transfer classical→WDM MLP via `mixed_dispatch` + `validate_wdm_alphafold_dispatch` nW-04 dispatch parity |
| **`validate_all` gap closure** | Added `validate_immunological_anderson`, `validate_multi_head_esn`, `validate_training_monitor`, `validate_barracuda_basecamp` |
| **PAPER_REVIEW_QUEUE spec update** | nS-06 row updated from "0/0 proposal" to actual counts (53/53 Rs, 20/20 Py, 18 GPU, 19 dispatch, 21 mH). baseCamp table expanded to 6 sub-theses |
| **Clippy fixes** | `immunological_anderson.rs`: const assertion blocks. `training_monitor.rs`: u32 range avoids sign-loss cast |
| **Quality gates** | fmt ✓ · clippy ✓ (0 new warnings) · 861/861 lib ✓ · 226 binaries ✓ · Python 40/40 ✓ · doc ✓ |

|| Session 106: baseCamp GPU Promotions | 4 GPU promotions (WS/LL/PGM/AC), 2 WDM mixed-hardware gaps closed (nW-03/nW-04), 4 validators added to `validate_all`, spec update, 861 lib, 226 bins | **ALL GREEN** |

### Session 107 — Gonzales Deep Modeling + 3D Lattice + Fajgenbaum MATRIX (March 2, 2026)

Extended baseCamp Paper 12 (nS-06) with three deep integration areas:

| Area | Details |
|------|---------|
| **nS-601: Gonzales dose-response** | Generalized Hill equation (n-cooperative), IC50 sweep for all 6 cytokines, cytokine barrier heights (W = ln(IC50) × scale), dose-response saturation validation |
| **nS-602: Pruritus time-series** | Treatment decay model (baseline → nadir → recovery), Gonzales (2016) G3 time-series validation, asymptotic baseline approach proof |
| **nS-603: Lokivetmab PK** | Exponential PK decay (`C(t) = C_0 × exp(-kt)`), log-linear duration regression (A=10.09, B=33.28), regression fit < 5 day error on G4 data |
| **nS-604: 3D tissue lattice** | Three-compartment disorder (immune/skin/neural Pielou → W), multi-layer Hamiltonian construction, level spacing ratio, barrier promotion spectral sweep (2D→3D), cross-compartment variance |
| **nS-605: Fajgenbaum MATRIX** | 6 drug candidates (Rapamycin, Tofacitinib, Tanezumab, Trametinib, Crisaborole, Nemolizumab), pathway×geometry×W scoring, Anderson-filtered ranking, AD flare + chronic profiles, integrated dose-response×MATRIX |
| **Python control** | 28/28 extended checks (`immunological_anderson_extended.py`), drift-checked |
| **Rust cross-language parity** | 187/187 checks in `validate_immunological_anderson_extended`, 16 new lib unit tests (27 total) |
| **Quality gates** | fmt ✓ · clippy ✓ (0 new warnings) · 861/861 lib ✓ · 226 binaries ✓ · Python 48/48 ✓ · doc ✓ |

|| Session 107: Gonzales + 3D + MATRIX | 28 Py + 187 Rs extended checks, 16 new unit tests, Hill/PK/pruritus/3D lattice/Fajgenbaum MATRIX, 861 lib, 226 bins | **ALL GREEN** |

### Session 108 — Deep Debt Execution + Doc Sweep + V71 Handoff (March 2, 2026)

Comprehensive audit and evolution of remaining technical debt, documentation alignment, and ToadStool handoff.

| Area | Details |
|------|---------|
| **Primal hardcoding → env-configurable** | `ORCHESTRATOR_SOCKET` → `orchestrator_socket()` (reads `BIOMEOS_ORCHESTRATOR_SOCKET`), `HEARTBEAT_INTERVAL_SECS` → `heartbeat_interval_secs()` (reads `NEURALSPRING_HEARTBEAT_SECS`). Runtime-configurable with sensible defaults. |
| **rpc_error dead_code** | Narrowed blanket `#[allow(dead_code)]` to only truly unused constants (`INVALID_REQUEST`, `INTERNAL_ERROR`); 5 used constants no longer suppressed. |
| **Provenance module refactored** | 851-line flat `provenance.rs` → 3-file module: `mod.rs` (201 lines: struct, env, `RuntimeEnvironment`, tests), `experiments.rs` (557 lines: 42 provenance records), `references.rs` (107 lines: analytical + cross-language refs). All under 1000 LOC. |
| **Doc link fixes** | Fixed 10 `cargo doc` warnings (unresolved `[BASELINE_COMMIT]`/`[BASELINE_DATE]` links in `references.rs`). Fixed `McCandless` clippy doc_markdown in `experiments.rs`. |
| **Wildcard import fix** | `experiments.rs`: `use super::*` → explicit imports per clippy::wildcard_imports. |
| **gpu.rs exit documented** | `process::exit(0)` for adapter listing documented as intentional CLI escape hatch. |
| **Deep audit: false positives resolved** | `as f64` casts: all 100+ are `usize → f64` (no `From` impl) — already correct. `Vec<f64>` params: all require ownership (struct fields or RPC serialization) — already correct. `.unwrap()` in library: all inside `#[cfg(test)]` — acceptable. |
| **Scripts synced** | `run_all_baselines.sh` updated to include nS-06 immunological_anderson (39 experiments, matches check_drift.sh). |
| **Doc sweep** | README, control/README, EVOLUTION_READINESS, CHANGELOG, CONTROL_EXPERIMENT_STATUS aligned to 330 Python, 861 lib tests, 226 binaries, 41 modules. |
| **V71 handoff** | ToadStool absorption handoff crafted. |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings, pedantic+nursery) · doc ✓ (0 warnings) · 861/861 lib ✓ · 226 binaries ✓ |

|| Session 108: Deep Debt + Doc Sweep + V71 | Primal env-configurable, provenance refactored (851→3 files), 10 doc warnings fixed, scripts synced (39 experiments), full doc sweep, V71 handoff | **ALL GREEN** |

### Sessions 110–111 — Paper Queue Validation & CPU Benchmark Buildout (March 2, 2026)

Full 10-tier validation pyramid confirmed green. CPU benchmark expanded from 11 to 14 domains.

| Area | Details |
|------|---------|
| **Control validation (S110)** | 207/207 validate_all PASS. Fixed 4 bugs: BP chain length (3→4), matmul orientation (right→left), eigenvalue variance tolerance (abs→rel), ESN error string mismatch |
| **Dispatch parity expanded (S110)** | +11 new checks: KL divergence, softmax_row_wise, HMM forward step, hill_gate, thermal diversity, global FST variance decomposition, pairwise FST (FST+FIS+FIT) |
| **ToadStool compute parity (S110)** | +6 checks: HMM chain, variance, eigh, matmul, entropy, allele_freq |
| **biomeOS graph coordination (S110)** | +5 checks: AF→π→FST→entropy multi-stage pipeline with CPU↔GPU parity per stage |
| **CPU benchmark buildout (S111)** | 3 new Python bench scripts (013 eco, 023 anderson, 025 meta-pop). 14/14 domains benchmarked. 31/31 PASS. 38.6× geomean (honest: includes 2 BLAS-bound domains) |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings) · 861/861 lib ✓ · 207/207 validate_all ✓ |
| **V73 handoff** | ToadStool absorption handoff: BarraCUDA usage audit (205 files, 25+ submodules), 4 absorption targets |

|| Sessions 110–111: Paper Queue + CPU Bench | 207/207 validate_all, 14-domain bench (38.6×), 3 bench scripts, 22 new parity checks, 4 bug fixes, V73 handoff | **ALL GREEN** |

### Session 112 — ToadStool S86 Rewire + Nautilus Absorption (March 2, 2026)

ToadStool pin bumped 7 commits (S79→S86). Nautilus dependency absorbed into BarraCUDA upstream.

| Area | Details |
|------|---------|
| **ToadStool pin bump** | `f97fc2ae` → `2fee1969` (7 commits: S80 nautilus/BatchedEncoder/Nelder-Mead, S81-82 deep debt +16 ComputeDispatch, S84-86 +33 ComputeDispatch, hydrology module split) |
| **Nautilus absorption** | `bingocube-nautilus` path dep removed. All imports migrated to `barracuda::nautilus`. 3 files changed (Cargo.toml, nautilus_bridge.rs, training_monitor.rs) |
| **DriftMonitor API** | `record(epoch, pop_size, mean, best)` → `record(&GenerationRecord, pop_size)`. `history` → `ne_s_history`. `consecutive_drift` → `is_drifting()` |
| **New validator** | `validate_toadstool_s86_rewire` (27/27 PASS): nautilus types, DriftMonitor lifecycle, bridge absorption, BetaObservation |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings) · 861/861 lib ✓ · 208/208 validate_all ✓ |
| **V74 handoff** | S86 rewire + nautilus absorption + new capabilities catalog |

|| Session 112: ToadStool S86 Rewire | Pin f97fc2ae→2fee1969, nautilus absorbed, DriftMonitor API migrated, 208/208 validate_all, V74 handoff | **ALL GREEN** |

### Session 113 — Cross-Spring S86 Evolution Benchmark (March 2, 2026)

Full cross-spring evolution validation and benchmarking of ToadStool S86 surface.

| Area | Details |
|------|---------|
| **validate_modern_cross_spring** | 57 → **68/68 PASS**: +11 checks for S80 nautilus (brain/observe/drift/bridge), S81 hydrology (5 ET₀ methods), S86 ComputeDispatch |
| **bench_cross_spring_modern** | 10 → **14/14 PASS**: +hydrology (5 ET₀ sub-µs), +nautilus (brain 8.6µs, drift 0.5µs, bridge 950ms) |
| **Provenance tracking** | 6 source springs documented: hotSpring, wetSpring, airSpring, groundSpring, bingoCube, neuralSpring |
| **Cross-spring shaders** | 844+ WGSL in ToadStool, 41 in metalForge, 15+ neuralSpring→ToadStool absorbed |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings) · 861/861 lib ✓ · 208/208 validate_all ✓ |

|| Session 113: Cross-Spring S86 Evolution | validate_modern_cross_spring 68/68, bench_cross_spring_modern 14/14, 6-spring provenance, 208/208 validate_all | **ALL GREEN** |

### Session 119 — Deep Lint Evolution + Shared Helpers (March 3, 2026)

Full `#[allow(` → `#[expect(` migration across entire codebase. 4 shared validation
helpers extracted (max_abs_diff_f64, bench_once, bench_median, median_duration_us).
13 bin files migrated to shared helpers. 8 new tests.

| Action | Detail |
|--------|--------|
| **Bin #![allow( → #![expect(** | 208 module-level + 31 inline → `#![expect(` with reasons. 477+ unfulfilled expectations resolved |
| **Lib #![allow( → #![expect(** | 28 remaining module-level converted. Only 6 `#[allow(` remain (all `#[cfg(test)]`) |
| **Zero lib warnings** | 28 warnings fixed (GPU ops, Anderson, triangle, geography, registry) |
| **Shared helpers** | `max_abs_diff_f64`, `bench_once`, `bench_median`, `median_duration_us` extracted to validation module |
| **13 bin migrations** | 3 max_diff + 4 bench_once + 6 median → shared helpers |
| **Tests** | 869 lib tests (up from 861 — 8 new for shared helpers) |
| **Quality gates** | fmt ✓ · clippy ✓ (0 lib/0 bin/0 unfulfilled) · doc ✓ · 869/869 lib ✓ |

||| Session 119: Deep Lint Evolution | 239 allow→expect, 477+ unfulfilled resolved, 4 shared helpers, 13 migrations, 869 lib tests | **ALL GREEN** |

### Session 121 — SimpleMlp Rewire + HMM f64 ComputeDispatch (March 4, 2026)

WDM surrogates rewired to `barracuda::nn::SimpleMlp`, eliminating local `MlpLayer`
(~300 LOC). HMM Viterbi chain rewired from per-step f32 Tensor loop to single f64
`barracuda::ops::bio::hmm_viterbi` ComputeDispatch. Cross-spring modern benchmark
documents provenance across 5 springs.

| Action | Detail |
|--------|--------|
| **WDM EOS + Transport** | `wdm_surrogate.rs` + `wdm_transport.rs` → `barracuda::nn::SimpleMlp` with `DenseLayer` format, domain normalization preserved |
| **HMM Viterbi chain** | `gpu_ops/bio/hmm.rs` → `barracuda::ops::bio::hmm_viterbi` (f64 log-domain `ComputeDispatch`) |
| **GPU validation binaries** | `validate_barracuda_wdm_eos.rs` + `validate_barracuda_wdm_transport.rs` updated for `SimpleMlp` layer iteration |
| **New: S121 rewire validator** | `validate_barracuda_s121_rewire` — 80/80 PASS (SimpleMlp EOS/Transport + HMM Viterbi/forward) |
| **New: modern bench** | `bench_cross_spring_modern` — 28/28 PASS (5-spring provenance) |
| **Upstream rewires** | 44 → **46** (SimpleMlp + hmm_viterbi) |
| **Quality gates** | fmt ✓ · clippy ✓ (0 warnings) · test ✓ (869/869 lib) · doc ✓ |

### S121 Current State (March 4, 2026)

| Metric | Value |
|--------|-------|
| validate_all | **220/220 PASS** |
| Binaries | 234 |
| Library tests | 869 |
| Latest handoff | **V81** (S121) |
| Sessions covered | 44–121 |
| Upstream rewires | 46 |
| Dispatch parity | 53/53 |
| ComputeDispatch bridge | 14/14 |
| NUCLEUS PCIe | 38/38 |
| `#[allow(` in lib (non-test) | **0** |
| `#[allow(` in bins | **0** |
| S121 rewire validation | 80/80 |
| Cross-spring modern bench | 28/28 |

---

### Benchmark Coverage (S123)

| Comparison | Coverage | Location |
|------------|----------|----------|
| **Python ↔ BarraCUDA CPU** | 15 domains | `validate_barracuda_cpu_bench` + 15 `control/*/bench_*.py` scripts |
| **Python ↔ CPU ↔ GPU** (3-way) | MLP/Transformer scaling | `metalForge/fossils/bench/bench_scaling.{py,rs}` |
| **Paper 026 BarraCUDA promotion** | CPU (11) + GPU (14) = 25/25 | `validate_barracuda_glucose_prediction` — LSTM Tensor matmul, GPU↔CPU parity ≤1.07e-6 |
| **Kokkos ↔ BarraCUDA GPU** | External only | `wateringHole/BARRACUDA_KOKKOS_GPU_BENCHMARK_RESULTS_MAR04_2026.md` |
| **Kokkos scripts in neuralSpring** | None | Kokkos benchmarks live in wateringHole as handoff docs |

Python ↔ BarraCUDA CPU benchmarks exist for all 14 Phase 0++ domains (pairwise L2,
Jaccard, Hill gate, multi-objective fitness, commutator, swarm NN, Anderson IPR,
eco batch fitness, NK fitness, RK4 GRN, HMM forward, replicator dynamics, Hamming
distance, global FST).

Kokkos comparison data (BarraCUDA WGSL vs Kokkos CUDA) is documented in wateringHole
handoff docs. No Kokkos benchmark scripts live in neuralSpring — Kokkos serves as
an external Tier 1 performance baseline per `BARRACUDA_KOKKOS_VALIDATION_BASELINE_NOTICE`.

*Evolution readiness tracker — following the hotSpring pattern for ToadStool absorption.*
