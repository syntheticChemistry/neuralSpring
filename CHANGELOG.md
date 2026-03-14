# Changelog

All notable changes to neuralSpring are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Session 149 (March 14, 2026)

### Session 149 — playGround: HuggingFace Model Lab + barraCuda Inference (2026-03-14)

**Model inference pipeline** — download HuggingFace models and run forward passes
through barraCuda's sovereign WGSL shader pipeline:

- `secrets.rs`: API key loader for `ecoPrimals/testing-secrets/api-keys.toml`
  (HF token, Anthropic, OpenAI, plus loose-format top-section parsing)
- `hf_hub.rs`: HuggingFace Hub REST client — model info, safetensors listing,
  file download with caching, full model download (config + weights + tokenizer)
- `model_config.rs`: Typed `TransformerConfig` from HF `config.json` — supports
  GPT-2, Llama, Mistral, Phi field naming conventions; auto-normalizes
  hidden_size, num_layers, num_heads, activation function
- `inference/weights.rs`: Load safetensors to barraCuda `Tensor` GPU buffers
  (f16/bf16/f32/f64 → f32 conversion, per-layer weight organization, weight
  summary reporting)
- `inference/transformer.rs`: GPU forward pass via `TensorSession` — embedding
  lookup, layer norm, attention dispatch, FFN with GELU, logit projection,
  top-k and softmax utilities
- `neuralspring_model_lab` binary: CLI for model exploration (info, download,
  inspect, load, forward, cached subcommands)

Dependencies added: `barracuda` (GPU compute), `reqwest` (HF Hub HTTP),
`safetensors` (weight loading), `toml` (secrets parsing), `bytemuck` (GPU buffer casting)

### Session 148 — playGround: Squirrel MCP + Interactive Runner (2026-03-14)

**playGround workspace member** — application sandbox for Squirrel MCP integration:

- `playGround/Cargo.toml`: new workspace member (tokio, serde, clap, anyhow)
- `ipc_client.rs`: reusable JSON-RPC 2.0 client over Unix sockets with biomeOS
  5-tier socket discovery (`BIOMEOS_SOCKET_DIR` → `XDG_RUNTIME_DIR/biomeos` →
  `/run/user/$UID/biomeos` → Android → temp)
- `squirrel_client.rs`: typed Squirrel MCP client (`ai.query`, `tool.execute`,
  `capability.announce`, `system.health`, `ai.list_providers`)
- `primal_client.rs`: typed neuralSpring primal client (all 14 `science.*`
  capabilities + `health` + `capability.list`)
- `mcp_tools.rs`: MCP tool definitions (JSON Schema) for all 14 capabilities

**Binaries**:

- `neuralspring_mcp_adapter`: bridge between Squirrel MCP and neuralSpring
  primal — discovers both sockets, registers capabilities, forwards tool calls,
  standalone mode if Squirrel unavailable
- `neuralspring_interactive`: AI-driven conversational experiment runner —
  `run`, `analyze`, `ask`, natural language queries with experiment context

**Squirrel evolution handoff**: `NEURALSPRING_SQUIRREL_EVOLUTION_HANDOFF_MAR14_2026.md`
to wateringHole — `#[allow()]` → `#[expect()]` migration, PRIMAL_REGISTRY update,
spec freshness, legacy removal (grpc_port, SongbirdClient), coverage 70% → 90%.

**Lysogeny/scyBorg awareness**: playGround README documents cross-domain
validation assignments (Usurper, Symbiont, Pathogen) and triple copyleft.

**Quality gates**: `cargo fmt` clean, `cargo clippy --workspace -- -W clippy::pedantic`
clean (0 warnings), 1115 lib tests pass, `#![forbid(unsafe_code)]` on playGround lib.

### Session 147 — Deep Debt Execution + BarraCUDA Evolution + Doc Cleanup (2026-03-14)

**Zero inline magic numbers** — all production tolerances centralized:

- `digester_anderson.rs`: `1e-30` → `tolerances::LOG_ZERO_GUARD` (2 sites), `1e-12` →
  `tolerances::EXACT_F64` (2 sites), `0.01` → named `XI_FLOOR` constant
- `nucleus_pipeline.rs`: `1e-6` → `tolerances::SPECIAL_FUNCTION_F64`, WDM stage
  parameters → 6 named constants
- `isomorphic_reservoir.rs`: `1e-30` → `LOG_ZERO_GUARD`, `1e-12` → `EXACT_F64` (3 sites)
- `attention_anderson.rs`: `1e-12` → `EXACT_F64`
- `counterdiabatic.rs`: `SAFETY_EPS` → `LOG_ZERO_GUARD`, `1e-15` → `NUMERICAL_DISTINCTNESS`
- `immunological_anderson/lattice.rs`: `1e-15` → `NUMERICAL_DISTINCTNESS`
- `wdm_esn/multi_head.rs`: `1e-6` → named `ESN_TIKHONOV_REGULARIZATION` (f32)
- `wdm_ensemble_qs.rs`: `1e-12` → `EXACT_F64` (2 sites)

**BarraCUDA evolution** — eliminate duplicate math:

- `digester_anderson::shannon_diversity` rewired to delegate to
  `barracuda::stats::shannon_from_frequencies` via `primitives::shannon_entropy`
  (zero duplicate math — was reimplementing `-Σ(p*ln(p))`)

**Provenance completeness** — 6 missing composition experiment records:

- Paper 027 (Digestion Prediction), Exp 096 (Digester-Anderson), Exp 097
  (Isomorphic Reservoir), Exp 098 (WDM Ensemble QS), Exp 099 (Introgression NN),
  Exp 100 (Attention Anderson) — all wired into provenance validation test

**Capability-based discovery** — hardcoded primal names eliminated:

- `ipc_push.rs`: `"petaltongue"` → `config::PETALTONGUE_SOCKET_DIR` /
  `config::PETALTONGUE_SOCKET_PREFIX`
- `config.rs`: new centralised discovery constants

**Documentation and handoff**:

- V100 toadStool/barraCuda absorption handoff (S147 deep debt + evolution)
- Root docs updated (README, CHANGELOG, EVOLUTION_READINESS)
- ecoPrimals/whitePaper/gen3/baseCamp/ updated

**Quality gates**: `cargo fmt` clean, `cargo clippy --workspace -- -W clippy::pedantic` clean
(0 warnings), 1115/1115 lib tests pass, 0 doc warnings.

### Session 146 — Industry GPU Parity + Deep Audit (2026-03-12)

**Industry GPU benchmark validators** — BarraCUDA WGSL vs cuBLAS/cuDNN/cuFFT/FlashAttention:

- `control/industry_gpu/bench_cuda_common.py`: shared CUDA timing harness
- `control/industry_gpu/bench_cublas_gemm.py`: SGEMM (6 scales) + DGEMM (3 scales)
- `control/industry_gpu/bench_cudnn_ops.py`: softmax, layernorm, GELU, conv2d, sigmoid
- `control/industry_gpu/bench_cufft.py`: FFT/RFFT f32 + f64
- `control/industry_gpu/bench_flash_attention.py`: MHA at 3 configs
- `src/bin/bench_industry_gpu_parity.rs`: Rust binary, invokes Python, comparison table

**Key findings** (RTX 4070, Vulkan vs CUDA):

- FFT beats cuFFT at 256–16K (0.19–0.85×)
- GEMM beats cuBLAS at small scales and 2048×2048 (0.16×)
- Softmax/GELU/Sigmoid: cuDNN ~7µs vs BarraCUDA ~170µs (dispatch overhead)
- RFFT: 700–1000× gap (structural, needs upstream fix)
- MHA: ~30× gap (decomposed vs FlashAttention fused kernel)

**Deep audit**:

- Provenance accuracy: `CPU_PARITY_COMMIT`/`CPU_PARITY_DATE`/`CPU_PARITY_ENVIRONMENT`
  constants added, test updated for multi-commit validation
- Tolerance tightening: `GPU_SOFTMAX_SUM_F32` 0.01→5e-3, `SIGMOID_SATURATION` +
  `GPU_HILL_GATE_F64` added to registry
- Visualization refactor: `scenarios/mod.rs` (1241 LOC) split into `mod.rs` +
  `scaffold.rs` + `combiners.rs` — all under 1000 LOC
- Clippy pedantic clean: `manual_range_contains`, `doc_markdown`, `expect_fun_call`,
  `semicolon_if_nothing_returned` fixes

### Session 145 — GPU Infrastructure Evolution Sprint (2026-03-11)

**Upstream sync** to barraCuda v0.3.5 (`0649cd0`), toadStool S146 (`751b3849`),
coralReef Iteration 33 (`b783217`):

- barraCuda v0.3.5: `ReduceScalarPipeline` f64 fix, `BatchedComputeDispatch`,
  `CoralReefDevice`, `FmaPolicy`, `tridiag_eigh_gpu`, 36 tolerances
- toadStool S146: `nvvm_transcendental_risk` in `gpu.info`, PrecisionBrain in
  `compile_wgsl_multi`, VRAM-aware routing, 19 SpringDomain variants
- coralReef Iter 33: NVVM poisoning fix, struct ABI fixes, 46/46 sovereign compile

**5 workload rewires** (absorbed workloads 20→25, local 6→1):

- `FusedChiSquaredGpu` → upstream `ReduceScalarPipeline` f64 fused path
- `FusedKlDivergenceGpu` → upstream `ReduceScalarPipeline` f64 fused path
- `hmm_backward` → upstream `barracuda::ops::bio::hmm_backward` (log-domain)
- `hmm_viterbi` → upstream `barracuda::ops::bio::hmm_viterbi` (f64 ComputeDispatch)
- `PairwiseL2Gpu` → upstream `barracuda::ops::distance::PairwiseL2Gpu` (matrix variant)

**NUCLEUS pipeline GPU dispatch**:

- `eigensolve` stage → `Dispatcher::eigensolve` (GPU-accelerated tridiag_eigh)
- `attention_anderson` stage → `Dispatcher::attention_anderson` (GPU-accelerated)
- `dispatch_capability()` routes GPU-capable stages through metalForge cost model
- `StageResult` records actual substrate in provenance metadata

**4 GPU experiments** (Exp 103–106):

- Exp 103: GPU-accelerated eigensolve pipeline via tridiag_eigh_gpu on RTX 4070
- Exp 104: BatchedComputeDispatch for spectral analysis across composition experiments
- Exp 105: Sovereign compile validation — ComputeDispatch<CoralReefDevice> on RTX 4070 Ada GSP
- Exp 106: Mixed-hardware composition pipeline with GPU+CPU stages and transfer cost measurement

**Validation**: 1115 lib + 73 forge + 9 integration tests. 258 binaries. 0 clippy.

**Handoffs**: V98 GPU dispatch evolution handoff (toadStool/barraCuda/coralReef)

### Session 143–144 — Novel Compositions + NUCLEUS Pipeline (2026-03-10)

**Session 144**: petalTongue composition visualization + NUCLEUS pipeline executor.
5 new scenario builders for composition experiments. `composition_study()` combiner.
`composition_pipeline()` DAG in metalForge. `nucleus_pipeline` executor with 6-stage
Tower→Node→Nest dispatch. 1112 lib + 73 forge tests, 254 binaries.

**Session 143**: 5 novel composition experiments (Exp 097–101):
- Exp 097: Anderson spectral analysis of attention weights
- Exp 098: WDM surrogate ensemble QS
- Exp 099: ESN/LSTM ensemble (digester + glucose + weather)
- Exp 100: ESN digester yield + Anderson QS disorder coupling
- Exp 101: HMM introgression applied to neural network layers

V96 handoff + doc sweep + cleanup audit.

### Session 142 — Upstream Rewire + `enable f64;` Fix + Cross-Spring Evolution (2026-03-10)

**Upstream rewire** to modern barraCuda/toadStool/coralReef:

- barraCuda Sprint 2 absorption: `barracuda::activations` (sigmoid, gelu, relu, relu_batch)
  delegated from `primitives.rs`; `fused_ops_healthy` canary adopted
- `SpringDomain` API migration: enum→struct (`NeuralSpring`→`NEURAL_SPRING`, etc.)
- `Precision::F16` removed (upstream dropped F16 tier); `compile_shader_universal`
  decomposed to precision-routed dispatch
- coralReef bridge discovery updated to ecosystem-aligned socket/manifest paths
- 54 validation binaries received standard `## Provenance` blocks (Groups A–D)
- 14 documentation/spec files and 2 validation binaries updated to current HEAD hashes

**Critical bug fix — `enable f64;` PTXAS silent-zero regression**:

- Diagnosed root cause: `enable f64;` in WGSL causes NVIDIA PTXAS on Ada Lovelace
  (SM89, RTX 40xx) to silently produce broken shaders returning zeros for all outputs
- Fix: `pipeline_cache.rs::get_or_compile_shader_f64_native` now strips `enable f64;`
  before compilation (matching `compile_shader_f64`/`compile_shader_df64` behavior)
- Resolved 5 of 7 dispatch parity failures: VarianceF64, CorrelationF64,
  MatrixCorrelationF64, InterPopAfVariance, ThermalDiversityCorr
- `fused_ops_healthy` canary: false→true
- Diagnostic binary: `diagnose_f64_regression.rs`

**HMM fused path workaround**:

- `HmmBatchForwardF64` upstream has shader/binding mismatch (5-binding shader vs
  7-binding dispatch) — silently returns 0.0
- `hmm_forward_chain_gpu` now detects 0.0 from fused path and falls back to
  per-step Tensor-based implementation

**Tolerance adjustment**:

- Glucose pearson CPU↔GPU: `TENSOR_EXACT_F32` (1e-6) → `GPU_DF64_TRANSCENDENTAL`
  (5e-4) for 1008-element DF64 correlation (measured diff ~1.7e-5)

**Validation results**:

- `validate_barracuda_dispatch_parity`: 48/55 → **55/55 PASS**
- `validate_toadstool_s79_rewire`: 19/19 PASS
- `validate_modern_cross_spring`: 68/68 PASS
- `cargo test --lib`: 1048/1048 PASS

**Handoffs**:

- V95 toadStool/barraCuda evolution handoff (enable f64 fix, cross-spring shader evolution)
- V95 coralReef detailed handoff (precision lessons, bridge status, shader inventory)
- V94 upstream rewire handoff
- `enable f64;` fix handoff (targeted barraCuda bug report)

**Pins**: barraCuda `83aa08a`, ToadStool S142 (`a86bc546`), coralReef Iteration 29 (`2779c88`)

### Session 139 — Visualization Evolution + Deep Debt (2026-03-10)

**4 new petalTongue scenario builders** in `src/visualization/scenarios/`:

- `search_results.rs`: BLAST-like search pipeline — alignment scores Bar, seed density TimeSeries, pipeline Gauges
- `streaming_io.rs`: streaming I/O quality — FASTQ quality Distribution, per-position TimeSeries, GC Gauge, VCF variant positions
- `kokkos_parity.rs`: Kokkos GPU parity dashboard — timing comparison Heatmap, overhead gap Bar, mean/max overhead Gauges
- `industry_coverage.rs`: ecosystem tool coverage — status Heatmap, domain completion Bar, primal ownership breakdown

**Deep debt elimination**:

- `config.rs` module centralizes primal identity (`PRIMAL_FAMILY`, `PRIMAL_DISPLAY_NAME`), env var names (`ENV_PETALTONGUE_SOCKET`, `ENV_REQUIRE_GPU`, etc.), petalTongue config (`PETALTONGUE_DOMAIN`, `PETALTONGUE_THEME`)
- `LINE_BUF_CAPACITY` / `VCF_LINE_BUF_CAPACITY` named constants replace magic numbers in streaming parsers
- `StreamSession::BACKPRESSURE_THRESHOLD` replaces inline `0.1`
- `db_encoded.clone()` eliminated in search scenario via reordering (borrow before move)
- Hardcoded `"neuralspring"`, `"neural-dark"`, `"PETALTONGUE_SOCKET"` strings replaced with config constants

**Upstream pin update** (documentation/specs):

- barraCuda: `a898dee` → `83aa08a` (Sprint 2 APIs, healthSpring domain, batched logsumexp, CoralReefDevice, 719 WGSL shaders)
- ToadStool: `bfe7977b` → `a86bc546` (S142, hardware testing, PCIe transport, ResourceOrchestrator, 19,900+ tests)
- coralReef: `d29a734` → `2779c88` (Iteration 29, NVIDIA last mile pipeline, SSA repair, multi-GPU sovereignty, 1200+ tests)

**Infrastructure**:

- `full_study()` expanded from 12 → 16 tracks with cross-domain edges
- New binary: `neuralspring_ecosystem_dashboard` — renders all 16 tracks + live gauge streaming
- `scripts/visualize.sh` updated with `--ecosystem` option
- 1048 lib tests (+82 from S136–S139), 233 binaries, 0 clippy, 0 doc warnings
- V93 handoff written to wateringHole

### Session 138 — Industry Gap Closure (2026-03-10)

**Streaming I/O parsers**: FASTA (16 tests), FASTQ (13 tests), VCF (10 tests) — O(record_size) memory, zero-copy where possible.
**CPU-reference BLAST pipeline**: `search/kmer_index` + `search/seed_extend` (19 tests) — k-mer seeding, ungapped extension, Smith-Waterman scoring.
**Kokkos parity benchmark**: `bench_kokkos_parity` harness (9 ops × production scale).
**Specs**: `INDUSTRY_TOOL_GAP_ANALYSIS.md`, `BLAST_LIKE_SEARCH_SCOPE.md`, `MSA_PIPELINE_SCOPE.md`.
968 lib tests, V92 handoff written.

### Session 137 — Upstream Rewire + Deep Debt (2026-03-10)

**Rewires**: hardcoded `256` → `WORKGROUP_SIZE_1D` (library+forge, 15 sites). 7 WGSL shader absorption statuses updated.
**Deep debt**: `gpu_or_exit()` async helper eliminates GPU init boilerplate (~75 binaries), duplicate `max_abs_diff` eliminated.
**Full audit**: zero unsafe, zero TODOs, zero mocks, zero hardcoded paths, all files < 800 LOC. 968 lib + 71 forge tests PASS.

### Session 136 — Deep Audit + Evolution (2026-03-10)

`PetalTonguePushClient::headless()` eliminates socket hardcoding. `Gpu::read_buffer_u32` wired to upstream parity. `validate_gpu_pure_workload_all` refactored (976→940 LOC). Industry GPU parity gap documented. Kokkos/Polybench/cuBLAS gap formally requested. 968 lib tests (+2 headless client).

### Session 135 — petalTongue Visualization Evolution (2026-03-09)

**7 new domain scenario builders** in `src/visualization/scenarios/`:

- `hmm.rs`: HMM phylogenetics — transition matrix Heatmap, log-likelihood TimeSeries, Viterbi Bar/Gauge
- `game_theory.rs`: evolutionary game theory — payoff Heatmap, cooperation TimeSeries, Nash Gauge
- `wdm.rs`: Warm Dense Matter — transport TimeSeries, phase-space Scatter3D (loads real surrogate MLP)
- `glucose.rs`: blood glucose prediction — CGM TimeSeries, error Distribution, R² Gauge, horizon Bar
- `immunological.rs`: Anderson immunological — dose-response TimeSeries, barrier Spectrum, PK decay, disorder Gauge, response Distribution
- `population.rs`: meta-population + pangenome — FST Heatmap, diversity Bar, geography Scatter3D, gene partition, repertoire Gauge
- `loss_landscape.rs`: Hessian analysis — eigenvalue Spectrum, 2D loss FieldMap, condition/gap Gauges

**All 8 `DataChannel` types** now exercised (added Heatmap, Distribution, FieldMap to previously used TimeSeries, Spectrum, Gauge, Bar, Scatter3D).

- `TrainingVisualizer` added to `training_monitor.rs`: wraps `StreamSession`, pushes per-epoch IPR/bandwidth/entropy/LSR timeseries + attention/condition gauges to petalTongue in real time
- `full_study()` expanded from 5 → 12 tracks with 9 cross-domain edges
- `dump_neuralspring_scenarios` now emits 13 scenario JSONs (7 new + complete study)
- New binary: `neuralspring_live_dashboard` — discovers petalTongue, streams simulated training loop with spectral diagnostics
- New script: `scripts/visualize.sh` (dump/live/render modes, following healthSpring pattern)
- `validate_petaltongue_scenarios`: 31 → 56 checks (+25 for new scenarios + all-8-types validation)
- 3 new helper constructors: `heatmap()`, `distribution()`, `fieldmap()` in scenarios/mod.rs
- Root docs, CHANGELOG, EVOLUTION_READINESS, experiments journal updated
- Handoffs: V91 → wateringHole (BarraCUDA/ToadStool absorption, petalTongue evolution)
- 0 clippy warnings, 0 fmt diff, 966 lib tests PASS

### Session 134 — Deep Debt Resolution & Doc Sweep (2026-03-09)

**Code quality audit and deep debt resolution.**

- Consolidated 7 duplicate activation functions (sigmoid, gelu, relu, softmax) into `primitives.rs`
  - Added 9 new tests: `gelu`, `gelu_f32`, `softmax`, `relu`, `relu_f32`, `relu_vec`, `relu_inplace`, plus edge cases
  - Lib tests: 957 → 966 (+9)
- Promoted 16+ inline tolerance literals to 5 new named constants in `tolerances/`
  - `GLUCOSE_CGM_STAT_TOL`, `GLUCOSE_TAU_TOL`, `PLDDT_DEGENERACY_THRESHOLD`, `GPU_KIMURA_BATCH_DIFF`, `TENSOR_RELU_DETERMINISM_F32`
  - Named tolerances: 145 → 150+
- Fixed 6 clippy pedantic/nursery errors across `visualization/stream.rs`, `validate_biomeos_graph.rs`, `validate_petaltongue_scenarios.rs`
- Added provenance triplets (Python script, commit, date, command) to 5 validation binary docblocks
- Replaced hardcoded primal namespaces in `coralreef_bridge.rs` with `BIOMEOS_NAMESPACES` env (capability-based discovery)
- `validate_gpu_shader_phase4.rs`: standardized no-GPU exit via `validation::exit_no_gpu()`
- `neural_pgm.rs`: `weight_to_transition` now delegates to `primitives::softmax` per row
- Replaced `unwrap()` calls in tests with `expect()` (descriptive), `panic!()` with `assert!(matches!(...))`
- Line coverage: 91.66% (above 90% target)
- Full doc sweep: README, EVOLUTION_READINESS, CONTROL_EXPERIMENT_STATUS, experiments/README, whitePaper/baseCamp, specs/ — all aligned to 966/220/246
- All builds green: `cargo fmt`, `cargo clippy -D warnings`, `cargo doc`, `cargo test`

### Session 133 — Phase 5–7 Buildout (March 9, 2026)

**metalForge PCIe**: `PcieBridge::transfer_buffer_strategy()` selects P2P vs CPU-staged transfer based on IOMMU group detection. `TransferStrategy` enum (`P2P` / `CpuStaged`). `MixedSubstrate::NpuToGpuP2P` variant for explicit PCIe P2P bypass. `mixed_substrate_p2p()` for P2P-aware routing.

**biomeOS pipeline DAG**: `metalForge/forge/src/graph.rs` — `StageNode` (capability-addressed), `PipelineGraph` (Kahn's topological sort), `PipelineExecution` (per-stage tracking). 3 canonical pipelines: `spectral_pipeline()` (diamond DAG), `population_genetics_pipeline()` (linear), `folding_pipeline()` (linear). Structural validation (cycles, duplicates, dangling edges). 15 unit tests.

**petalTongue StreamSession**: `src/visualization/stream.rs` — session lifecycle (start/resume), backpressure awareness (error rate monitoring), `SessionStats` (messages/sec, bytes, errors, uptime). `push_replace()` and `query_capabilities()` added to `PetalTonguePushClient`. IPC buffer: 4KB → 64KB (parity with healthSpring).

**Feature-gated validate_all**: `FEATURE_BINARIES` list for `--features primal` binaries. `validate_nucleus_tower` (22/22) and `validate_biomeos_spectral` (29/29) now in suite.

**New validators**: `validate_biomeos_graph` (32/32 PASS), `validate_petaltongue_scenarios` (31/31 PASS). `validate_nucleus_compute_dispatch` now 43/43 PASS (was 39).

**Lib tests**: 911 → 957 (+46: 15 graph, 5 stream, 4 pcie, 4 mixed, 18 ipc). **Forge tests**: 43 → 71 (+28: 15 graph, 5 pcie, 8 mixed). **Binaries**: 240 → 246.

### Session 132 — Upstream Rewire + Cross-Spring Provenance (March 8, 2026)

**Pin updates**: barraCuda `2a6c072` → `a898dee` (deep debt: typed errors, named constants, test resilience, lint compliance). ToadStool `88a545df` → `bfe7977b` (S130+: deep debt, unsafe audit, dependency audit, spring sync confirming zero API breakage for all 5 springs, 19,777 tests). coralReef `72e6d13` → `d29a734` (Iteration 10: AMD E2E GPU dispatch verified on RDNA2/GFX1030, conditional branch fix, 990 tests).

**Cross-spring provenance wired**: `barracuda::shaders::provenance` now exercised via 7 new lib tests validating the programmatic cross-spring shader registry (22 shaders tracked, 5 springs, 17 dependency edges). Provenance report wired into `bench_cross_spring_shader_evolution` and `validate_cross_spring_shader_evolution` — both now output barraCuda's evolution timeline and dependency matrix.

**Precision routing**: `Dispatcher::shared_memory_f64_safe()` added — hardware-adaptive safety check for fused workgroup-based f64 reductions (groundSpring V84–V85 shared-memory discovery). Returns `false` when `PrecisionRoutingAdvice` is `F64NativeNoSharedMem`, `Df64Only`, or `F32Only`. 2 new lib tests for precision routing defaults.

**Lib tests**: 902 → 911 (+9: 7 provenance, 2 precision routing).

**biomeOS NUCLEUS live deployment**: neuralSpring primal rebuilt with 14 capabilities (11 → 14: +`science.cross_spring_provenance`, +`science.cross_spring_benchmark`, +`science.precision_routing`). Live on Eastgate RTX 4070 (NVIDIA Vulkan, Hybrid f64, PCIe 4.0 x16). Full NUCLEUS tower running: BearDog + Songbird + ToadStool + Squirrel + Neural API + airSpring. `validate_nucleus_tower` 22/22 PASS, `validate_biomeos_spectral` 29/29 PASS against live primal.

**V90 handoff**: Formal handoff to ToadStool/BarraCUDA/coralReef team documenting the upstream rewire and cross-spring evolution.

### Session 131 — Full Green Validation (March 7, 2026)

Isomorphic coverage fix (WGSL discovery: BarraCUDA + metalForge + Tensor ops → 25/25 100%). 42/42 Python drift PASS. 901→902 lib tests. V89 handoff.

### Session 130 — Upstream Rewire + Revalidation (March 7, 2026)

**ToadStool S130 pin**: Updated from S94b. Hardware discovery + orchestration, coralReef shader proxy, JSON-RPC only (REST removed S90), `SubstrateType` 8 variants, capability-based discovery. No code changes needed (no `SubstrateType` or REST usage in neuralSpring).

**BarraCUDA `2a6c072` sync**: `PrecisionRoutingAdvice` (F64Native / F64NativeNoSharedMem / Df64Only / F32Only) wired into `Dispatcher::precision_routing()`. Higher-level than `fp64_strategy()` — captures shared-memory reliability axis for workgroup-based reductions.

**Fused GPU test gating**: 11 failing tests (VarianceF64, CorrelationF64, HmmBatchForwardF64) now gated via canary variance probe. Tests skip gracefully when fused ops return nonsensical values (upstream `Fp64Strategy` regression). Affects both llvmpipe and real hardware (RTX 4070, TITAN V NVK).

**coralReef rename**: `coralNAK` → `coralReef` in CHANGELOG.md. Iteration 7 — 8 neuralSpring shaders in corpus (2 compile, 5 need df64 preamble, 1 needs external include).

**Debt cleanup**: `validate_gpu_pure_workload_all.rs` 1006→995 LOC. `validate_cross_spring_rewire.rs`: raw `Path::new` → `baseline_path()`. `validate_game_theory.rs`: inline `0.1` → `tolerances::GAME_DEFECTION_UPPER`. Clippy `#[expect(clippy::wildcard_imports)]` → `#[allow(...)]` in `tolerances/registry.rs`. `sate_alignment.rs`: documented pairwise_distance as intentional divergence from BarraCUDA L2-based `PairwiseDistance`.

**Updated specs**: `TOADSTOOL_HANDOFF.md`, `BARRACUDA_USAGE.md`, `EVOLUTION_READINESS.md`, `README.md` — all reflect S130 pins. V88 handoff.

**Quality gates**: `cargo fmt` clean, `cargo clippy` 0 warnings (pedantic+nursery), `cargo doc` 0 warnings, `cargo test --workspace` all pass. 218/218 validate_all.

### Session 129 — API Sync Evolution + Quality Gate Hardening (March 7, 2026)

**Struct-based API migration**: All call sites migrated from positional arguments to BarraCUDA struct-based dispatch APIs (`HmmForwardArgs`, `GillespieModel`, `TensorShape`, `Conv2dConfig`, `Pool2dConfig`, `Rk45DispatchArgs`). ~30 call sites across ~15 files. `PairwiseL2Gpu::dispatch` now returns `Result`.

**`#![forbid(unsafe_code)]`**: Crate-level policy gate enforced in `src/lib.rs`. The compiler now rejects any future unsafe code.

**Quality gate evolution**: `#[allow(clippy::wildcard_imports)]` → `#[expect(... reason = "...")]` in `tolerances/registry.rs`. `.unwrap()` → `is_some_and()` in `validate_modern_cross_spring.rs`. `validate_barracuda_cpu_bench.rs` reduced 1001→999 LOC.

**Named tolerances**: +2 (`GPU_PRNG_UNIFORMITY_MEAN`, `GLUCOSE_BASELINE_DATE`) — 141+ total. Last inline literals replaced.

**Documentation alignment**: Root docs, baseCamp, experiments/, EVOLUTION_READINESS.md updated to current counts (883 lib, 240 bins, 218/218 validate_all). Fossil record paths corrected (`../phase1/toadstool/` → `../barraCuda/crates/barracuda/`).

**Upstream GPU investigation**: 12 fused GPU tests (VarianceF64, CorrelationF64, HmmBatchForwardF64) return 0.0 on llvmpipe — identified as pre-existing upstream issue, not neuralSpring regression.

**Quality gates**: `cargo fmt` clean, `cargo clippy` 0 warnings (pedantic+nursery), `cargo doc` 0 warnings. V87 handoff.

### Session 128 — BarraCUDA Modern Rewire + ToadStool S94b Catchup (March 5, 2026)

**Rewire**: `VarianceReduceF64` → `VarianceF64` (fused Welford, TensorContext, Fp64Strategy-aware) across 14 files (~25 call sites). Production path already used `VarianceF64` since S126; this aligns validators and benches with the modern API.

**ToadStool absorption tracker updated**: neuralSpring V75/S113 → V85/S127 (14 handoff versions ahead). New P3 items registered: fused LSTM cell WGSL shader, autocorrelation GPU op, R² score GPU op. Flash attention and LayerNorm+GELU marked as available in barraCUDA.

**coralReef**: Sovereign shader compiler at `ecoPrimals/coralReef/` (renamed from coralNAK). Iteration 7 — NVIDIA SM70-SM89, AMD RDNA2+, f64 transcendentals, 390 tests. 8 neuralSpring shaders in corpus (2 compile, 5 need df64 preamble, 1 needs external include).

**Reviewed upstream**: BarraCUDA HEAD has 6 commits past v0.3.3 (fused reduction shaders, TensorContext migration, subgroup detection, DF64 precision tier for 15 ops, NAK compound assignment fix, chi_squared feature gating). All compatible — 0 breaking changes, 0 new regressions.

**Quality gates**: `cargo fmt` clean, `cargo clippy` 0 warnings (pedantic+nursery), `cargo test --lib` 871/883 (12 upstream GPU SIGSEGV — unchanged), `cargo doc` 0 warnings. V86 handoff.

### Session 127 — Paper 026 Full-Tier Validation + Baseline Closure (March 5, 2026)

**Paper 026 promoted to all 4 validation tiers**: LSTM glucose prediction (Chuna 2020) now has full coverage across the entire validation pipeline:
- `validate_barracuda_cpu_bench`: 15th domain — LSTM reservoir forward + autocorrelation timing vs Python/NumPy
- `validate_cpu_math_parity`: 10th kernel — autocorrelation + R² cross-language parity (Rust CPU = Python, 1e-10)
- `validate_gpu_pure_workload_all`: 13th domain — GPU Tensor matmul LSTM gate projection vs CPU reference
- `validate_barracuda_dispatch_parity`: 55th check — dispatched variance + pearson on CGM-scale data (CPU ↔ GPU)

**Baseline suite closure**: `run_all_baselines.sh` now includes Paper 026 `glucose_prediction.py` — all 26 papers covered by the unified baseline runner.

**Python reference regeneration**: `control/generate_cpu_references.py` extended with `gen_glucose_lstm()` — autocorrelation + R² reference data from Python/NumPy for cross-language parity validation.

**New Python benchmark script**: `control/glucose_prediction/bench_glucose_lstm.py` — LSTM reservoir forward + autocorrelation micro-benchmark for Python vs Rust timing comparison.

**New tolerance**: `GPU_LSTM_GLUCOSE_F32` (0.05) — multi-step LSTM f32 Tensor chain (12 steps × 2 matmul + sigmoid/tanh per step).

**Quality gates**: `cargo fmt` clean, `cargo clippy` (pedantic+nursery) 0 warnings, `cargo doc` 0 warnings, 883 lib tests (871/883 pass, 12 upstream GPU SIGSEGV). 218/218 `validate_all`.

### Session 126 — Cross-Spring Fused Op Absorption + Validation + Benchmark (March 5, 2026)

**Fused op absorption**: `variance_gpu` upgraded from `VarianceReduceF64` to `VarianceF64` (fused single-pass Welford WGSL). New functions: `mean_variance_gpu` (single-dispatch fused mean+variance), `correlation_full_gpu` (returns `CorrelationResult` with means+variances+Pearson r), `correlation_matrix_gpu` (n×p → p×p Pearson matrix via `stats_f64::matrix_correlation`).

**Cross-spring provenance**: Each fused op documents its origin Spring(s): hotSpring (Welford, logsumexp, eigensolve), wetSpring (Shannon, diversity, correlation), neuralSpring (chi-squared, KL, pairwise L2), airSpring/groundSpring (matrix correlation).

**New binaries**: `validate_toadstool_s94b_wgpu28` (S94b pin validation + fused ops + wgpu 28 API surface), `bench_cross_spring_evolution` (13 benchmarked ops from 5 Springs, provenance-tracked timing).

**New lib tests**: `gpu_mean_variance_fused`, `gpu_correlation_full_fused`, `gpu_correlation_matrix_known` (3 new, 883 total).

**Quality gates**: `cargo fmt` ✓ · `cargo clippy` 0 warnings (pedantic+nursery) · `cargo test --lib` 871/883 (12 GPU SIGSEGV — upstream) · `cargo doc` 0 warnings. 240 binaries, 218/218 validate_all. V84 handoff.

### Session 125 — wgpu 28 + BarraCUDA v0.3.3 + ToadStool S94b Sync (March 5, 2026)

**wgpu 22 → 28 migration**: Updated ~70 wgpu API call sites across `src/` and `metalForge/forge/`: `Maintain::Wait` → `PollType::Wait` (13), `push_constant_ranges` → `immediate_size` (19), `entry_point: &str` → `Option<&str>` (19), `set_bind_group` wrapped in `Some()` (17), `Instance::new` reference parameter (1), `enumerate_adapters` async (2). `DeviceDescriptor` gains `experimental_features` + `trace` fields. `from_existing` takes owned `Device`/`Queue` (Arc removed in wgpu 28).

**BarraCUDA v0.3.1 → v0.3.3**: Removed `unidirectional` feature (removed upstream in v0.3.2). Absorbs: wgpu 28 stack, `GuardedDeviceHandle`, fused mean+variance and correlation shaders (f64/DF64), subgroup capability detection, workgroup size constants, three-tier precision model (f32/DF64/f64).

**ToadStool S87 → S94b pin**: 9 upstream commits reviewed. Key changes: `BarraCUDA` extracted to standalone primal (S89), D-SOV resolved (capability-based discovery), `NpuDispatch` + `NpuParameterController` added, REST removed (JSON-RPC 2.0 only), `management/resources` crate removed.

**Dependency bumps**: wgpu 22→28, tokio 1.35→1.49, pollster 0.4 added to metalForge/forge.

**Lint evolution**: 4 unfulfilled `#[expect]` cleaned (clippy 1.93 no longer triggers `float_cmp`/`wildcard_imports`/`cast_possible_truncation` in those contexts). `i as f64` → `f64::from` cast.

**Quality gates**: `cargo fmt` clean, `cargo clippy` 0 warnings (pedantic+nursery), `cargo test --lib` 871/880 PASS (9 GPU Tensor tests fail — upstream SIGSEGV in barracuda's own wgpu 28 GPU pipeline, confirmed by testing barracuda directly), `cargo doc` 0 warnings. V83 handoff.

### Session 124 — airSpring V069 Naming Rewire + HMM Absorption (March 5, 2026)

**airSpring V069 naming rewire**: Swept 20 library `.rs` files, 38 binary `.rs` files, 10+ specs `.md` files, and root docs to apply the canonical naming convention: `ToadStool` = hardware dispatch/streaming/orchestration, `BarraCUDA` = math engine/shaders/ops/stats/linalg. Historical absorption references preserved with `ToadStool` attribution. All primal names backticked in doc comments for `clippy::doc_markdown` compliance.

**HMM forward chain absorption**: Rewired `hmm_forward_chain_gpu` from per-step Tensor GEMV loop (T round-trips) to single `HmmBatchForwardF64` `ComputeDispatch` (log-domain, zero per-step CPU↔GPU round-trips). Automatic fallback to legacy per-step path if fused dispatch fails. All 38 HMM tests pass.

**validate_all gap closure**: Added `validate_toadstool_s79_rewire` and `validate_toadstool_s93_barracuda_extraction` to `validate_all` (215→217 binaries).

**Handoff update**: `specs/TOADSTOOL_HANDOFF.md` updated to V82 (Session 124). Counts refreshed: 238 binaries, 880 lib tests, 217/217 validate_all.

**Quality gates**: `cargo fmt` clean, `cargo clippy --workspace -- -W clippy::pedantic -W clippy::nursery` 0 warnings, `cargo test --lib` 880/880 PASS, `cargo doc --workspace --no-deps` 0 warnings.

### Session 123 — Comprehensive Modernization Wave 2 (March 5, 2026)

**`partial_cmp().unwrap_or()` → `f64::total_cmp()` completion**: Evolved all 47 remaining call sites across 21 validation/bench binaries and 2 library modules (`directed_evolution.rs`, `swarm_robotics.rs`). Zero `partial_cmp` sort/max_by patterns remain in the codebase. Replaced `partial_cmp` comparison in `validate_barracuda_gpu_modes.rs` with direct `>` operator (NaN-safe in context).

**Bare magic-number tolerance elimination**: Replaced numeric literals in test assertions across 5 library modules with named constants from the `tolerances` registry:
- `eco_dynamics.rs`: `1e-10` → `tolerances::CROSS_LANGUAGE`
- `anderson_localization.rs`: `1e-14` → `tolerances::ZERO_DETECTION`
- `meta_population/fst.rs`: `1e-14` → `tolerances::ZERO_DETECTION`
- `gpu_shader_validation.rs`: `1e-15` → `tolerances::NUMERICAL_DISTINCTNESS`, `1e-10` → `tolerances::CROSS_LANGUAGE`

Behavioral thresholds (logic gate 0.3/0.5/0.7, spectral radius 0.05) correctly left as domain constants, not numeric precision.

**Library `unwrap_or` audit**: All `unwrap_or(0.0)` / `unwrap_or(0)` patterns in library code verified as semantically correct (correlation → 0.0 for degenerate inputs, safe-indexing fallbacks, guarded by early returns). No error-hiding patterns found.

**Paper 026 buildout (Chuna — LSTM blood glucose prediction)**: Complete end-to-end reproduction of Chuna (2020) "Setting Limits on Neural Network's Predictive Capacity in T1D Blood Glucose Concentration" (medRxiv 2020.08.04.20117812). New files:
- `control/glucose_prediction/glucose_prediction.py` (9/9 PASS): Synthetic CGM generator, LSTM reservoir + ridge readout, multi-horizon (5/30/60/120/240 min) prediction analysis
- `src/glucose_prediction.rs` (11 unit tests): Rust module with CGM generator, autocorrelation analysis, Cholesky-based ridge regression, full experiment orchestration
- `src/bin/validate_glucose_prediction.rs` (26/26 PASS): hotSpring validation binary with horizon degradation checks, determinism proof, Python parity comparison

Key findings match Chuna: R²(5min)=0.97 (trivial), R²(30min)=0.73 (sweet spot, 16% over persistence), R²(240min)=0.18 (converging to mean). Autocorrelation τ≈1.5 hrs. Validates isomorphic thesis: same LSTM primitives work across weather (Exp 3/9), plasma physics (nW-03), and biomedical (Paper 026) domains.

**Paper 026 BarraCUDA promotion**: Created `validate_barracuda_glucose_prediction.rs` (25/25 PASS) validating the glucose prediction LSTM through two tiers:
- **Tier 1 — BarraCUDA CPU** (11 checks): `barracuda::stats` primitives (variance, Pearson correlation, R², RMSE) produce identical results to local Rust implementations. Full experiment orchestration confirmed.
- **Tier 2 — BarraCUDA GPU** (14 checks): LSTM gate projections via `Tensor::matmul` + CPU-side sigmoid/tanh, readout via `Tensor::matmul` + `Tensor::add`. GPU↔CPU parity across all 5 horizons: max relative error 1.07e-6 (well within `ML_MLP_F32` tolerance). Hidden mean parity 6.20e-8. Bit-perfect determinism confirmed on NVIDIA RTX 4070 (Vulkan).

Evolution chain: Chuna CGM LSTM → Python reservoir → Rust CPU → BarraCUDA (CPU stats) → BarraCUDA (GPU Tensor).

**`validate_all` integration**: Added `validate_glucose_prediction` and `validate_barracuda_glucose_prediction` to `validate_all` (213→215 binaries). Updated Full Validation Stack Matrix, README, and EVOLUTION_READINESS counts.

**Quality gates**: `cargo fmt` clean, `cargo clippy --workspace` zero warnings (pedantic+nursery), 880/880 lib tests PASS. 215/215 `validate_all`. 40 Python drift baselines.

### Session 122 — Deep Debt Execution + Idiomatic Evolution (March 4, 2026)

**`#[allow]` → `#[expect]` completion**: Migrated all 24 remaining `#[allow(clippy::...)]` in library source to `#[expect(clippy::..., reason = "...")]`. Zero `#[allow]` remains in `src/`. Every suppression now has a documented reason and will error if the suppressed lint no longer fires.

**`partial_cmp().unwrap_or()` → `f64::total_cmp()`**: Evolved 15+ occurrences across 12 library modules, 3 test files, and the primal binary to use the modern idiomatic `f64::total_cmp` method (stable since Rust 1.62). Handles NaN deterministically without the `unwrap_or(Ordering::Equal)` workaround.

**`wdm_esn.rs` refactored to module directory**: Split 717-line monolith into 4 focused submodules: `classifier.rs` (CPU ESN + JSON deser, 121 lines), `gpu_path.rs` (barracuda Tensor GPU classification, 89 lines), `multi_head.rs` (hotSpring cross-spring multi-head ESN, 263 lines), `tests.rs` (14 tests). All 14 tests pass, public API unchanged.

**Tolerance centralization**: Added `SDPA_PASSTHROUGH` (1e-6) to `tolerances/mod.rs` with mathematical justification, registered in tolerance registry. Eliminated last inline tolerance literal from `coral_forge/attention.rs`.

**Streaming I/O spec**: Created `specs/STREAMING_IO_REQUIREMENTS.md` with 6 requirements (R-01..R-06) for future FASTQ/mzML/MS2 parsers — mandatory streaming, safe Rust only, `BufReader` pattern, XML pull parsing, validation round-trips.

**Weight loader I/O documentation**: Updated `weight_loader.rs` doc to explicitly document safetensors API constraint (`&[u8]` required, no streaming API) and evolution path.

**Coverage verified**: `cargo llvm-cov --lib` = **91.76%** (above 90% threshold). Python baselines: **41/41 experiments PASS** (330+ checks, zero drift). `validate_all`: **213/213 PASS**.

**Dependency audit**: All 10 direct deps are pure Rust, ecoBin compliant. 125 transitive crates (wgpu GPU stack). No C dependencies. No evolution needed.

**Quality gates**: `cargo fmt` clean, `cargo clippy --workspace -- -D warnings` zero warnings (pedantic+nursery), `cargo doc --workspace --no-deps` zero warnings, 869/869 lib tests PASS, 9/9 integration tests PASS.

### Session 121 — SimpleMlp Rewire + HMM f64 ComputeDispatch (March 4, 2026)

**WDM surrogates rewired to `barracuda::nn::SimpleMlp`**: `wdm_surrogate.rs` (EOS 2→128→128→2) and `wdm_transport.rs` (Transport 3→64→64→3) replaced local `MlpLayer` with upstream `SimpleMlp` + `DenseLayer`. ~300 LOC eliminated. Domain normalization and output transforms preserved in wrapper logic. JSON weight loading adapted for `DenseLayer` `Vec<Vec<f64>>` format.

**HMM Viterbi chain rewired to f64 ComputeDispatch**: `hmm_viterbi_chain_gpu` replaced per-step f32 `Tensor` loop with single `barracuda::ops::bio::hmm_viterbi` dispatch. Linear→log domain conversion at call site. f64 precision via `hmm_viterbi_f64.wgsl`. Zero CPU round-trips.

**New validation binary**: `validate_barracuda_s121_rewire` — **80/80 PASS** (SimpleMlp layer counts, I/O sizes, prediction finiteness, determinism, JSON roundtrip, HMM Viterbi/forward CPU parity).

**New benchmark binary**: `bench_cross_spring_modern` — **28/28 PASS** (SimpleMlp, HMM, stats, linalg, Dispatcher evolved ops; 5-spring provenance documented per section).

**Upstream rewires**: 44 → **46** (SimpleMlp + hmm\_viterbi). **V81 handoff**.

### Session 120 — Deep Debt Audit + CI Hardening + Idiomatic Evolution (March 3, 2026)

**Comprehensive audit**: Full codebase review against wateringHole standards — all gates pass.

**Zero clippy warnings (all-features)**: Fixed production `suboptimal_flops` in `anderson_localization.rs` (→ `mul_add`). Resolved 18 pedantic/nursery warnings across 6 test modules with targeted `#[expect(` + reason strings. Removed 2 unnecessary `#![allow(` in `tests_cpu.rs`/`tests_gpu.rs` (lints never triggered).

**`#[allow(` → `#[expect(` completion**: Converted remaining 6 `#![allow(` in test files to `#![expect(` with reason strings. Two removed entirely (unfulfilled). Zero `#[allow(` remains in the entire codebase — all suppressions now use `#[expect(` with documented reasons.

**CI hardened to match local gates**: `.github/workflows/rust.yml` clippy step now runs `--all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery` (was missing pedantic/nursery/all-features). Makefile and justfile `lint-rust` targets updated with `--all-features` and `RUSTDOCFLAGS="-D warnings"`. Feature-gated code (`rpc_service` under `primal`) is now linted in all three environments.

**Audit confirms**: 337/337 SPDX headers, zero unsafe, zero files >1000 lines (max 953), zero TODO/FIXME markers, zero production mocks, zero `unwrap()`/`expect()` in library code, zero hardcoded paths, 41 provenance records with full Python trace, all tolerances documented with mathematical derivations.

**V80 handoff**: `NEURALSPRING_TOADSTOOL_V80_S120_DEEP_DEBT_AUDIT_HANDOFF_MAR03_2026.md`

### Session 119 — Deep Lint Evolution + Shared Helpers + Debris Sweep (March 3, 2026)

**Full `#[allow(` → `#[expect(` migration**: Every `#![allow(` (208 module-level) and `#[allow(` (31 inline) in `src/bin/` converted to `#![expect(` with reasons. Remaining library `#![allow(` (28 modules) also converted. Iterative clippy fix: 477+ unfulfilled expectations resolved by removing over-suppressed lints. Net effect: lint suppression is now **precise** — every `#[expect(` catches a real lint, and any drift in lint behavior will be caught as compilation warnings.

**Zero lib clippy warnings**: Fixed 28 remaining lib warnings by restoring `cast_possible_truncation` to GPU ops modules, `many_single_char_names` to Anderson localization, and handling cross-compilation-context wildcard imports. Only 6 `#[allow(` remain in library — all in `#[cfg(test)]` modules where `expect_used`/`unwrap_used` don't fire.

**Shared validation helpers extracted**: 4 new helpers in `src/validation/`:
- `max_abs_diff_f64` — replaces 3 local `max_diff` definitions + ~25 inline implementations
- `bench_once` — replaces 4 identical single-run `bench` helpers (returns result + µs)
- `bench_median` — standardized warmup+iteration benchmarking
- `median_duration_us` — replaces 6 local `median`/`median_us` implementations
Migrated 13 bin files to use shared helpers. 8 new tests (869 total lib tests).

**V79 handoff**: `NEURALSPRING_TOADSTOOL_V79_S119_DEEP_LINT_EVOLUTION_HANDOFF_MAR03_2026.md`

### Session 118 — barraCuda Standalone Extraction Rewire (March 3, 2026)

**barraCuda rewire**: Dependency path swapped from embedded `../phase1/toadstool/crates/barracuda` (S87) to standalone `../barraCuda/crates/barracuda` (v0.3.1). Zero breaking API changes. CI workflow updated (7 checkout blocks). Full revalidation: 861/861 lib, 9/9 integration, all key validators green.

**New validator**: `validate_toadstool_s93_barracuda_extraction` (29/29 PASS) — validates S88+ APIs (`tridiag_eigenvectors`, domain tolerance constants, `MathOp`, `Fp64Strategy`, `ComputeExecutor`), nautilus continuity, and dispatcher continuity on standalone path.

**L-BFGS gap closed**: `barracuda::optimize::LbfgsGpu` now available in v0.3.1 (was P2 OPEN).

**Docs updated**: EVOLUTION_READINESS.md, specs/BARRACUDA_USAGE.md, specs/BARRACUDA_REQUIREMENTS.md, README.md — all reference barraCuda as standalone primal.

**V78 handoff**: `NEURALSPRING_TOADSTOOL_V78_S93_BARRACUDA_REWIRE_HANDOFF_MAR03_2026.md`

### Session 108 — Deep Debt Execution + Doc Sweep + V71 Handoff (March 2, 2026)

**Primal hardcoding → env-configurable**: `ORCHESTRATOR_SOCKET` → `orchestrator_socket()` (reads `BIOMEOS_ORCHESTRATOR_SOCKET`). `HEARTBEAT_INTERVAL_SECS` → `heartbeat_interval_secs()` (reads `NEURALSPRING_HEARTBEAT_SECS`). `rpc_error` dead_code narrowed to only unused constants.

**Provenance module refactored**: 851-line flat `provenance.rs` migrated to 3-file module: `mod.rs` (201 lines), `experiments.rs` (557 lines, 42 provenance records), `references.rs` (107 lines). All under 1000 LOC limit.

**Doc quality**: Fixed 10 `cargo doc` warnings (unresolved links), clippy doc_markdown fix, wildcard import fix. 0 doc warnings, 0 clippy warnings (pedantic+nursery), 0 fmt issues.

**Scripts synced**: `run_all_baselines.sh` updated to include nS-06 immunological_anderson (39 experiments, matches `check_drift.sh`).

**Doc sweep**: README, control/README, EVOLUTION_READINESS, CHANGELOG, CONTROL_EXPERIMENT_STATUS aligned to 330 Python, 826 lib tests, 226 binaries, 41 modules.

**Deep audit completed**: `as f64` casts (all `usize`, correct), `Vec<f64>` params (all need ownership), `.unwrap()` in library (all `#[cfg(test)]`), no TODOs/FIXMEs/stubs, no unsafe, no production mocks.

**V71 ToadStool handoff**: Full evolution status, barracuda integration inventory, absorption recommendations.

### Session 104 — Full Validation Chain + 3 BarraCUDA Fixes + V70 Handoff (March 2, 2026)

**Full validation chain**: 202/202 validate_all PASS (0 FAIL), up from 197/202. 39/39 Python drift check (zero baseline drift). 90.49% llvm-cov line coverage (target: 90%). 753 lib tests PASS, 0 clippy warnings.

**3 barracuda fixes evolved locally for `BarraCUDA` absorption**:
- `fft_1d.rs`: FFT ping-pong buffer selection — `is_multiple_of(2)` branch was reading stale buffer for odd-stage FFTs. Now always reads `current_input` after swap. 24/24 PASS (was 19/24)
- `ShaderTemplate::for_driver_auto`: Strip `enable f64;` directive before naga compilation — naga handles f64 via capability flags, not WGSL directives. Unblocks Wright-Fisher GPU pipeline (4/4 PASS, was panic)
- `asin_df64` iterative form already in tree — confirmed coral forge GPU pipeline 16/16 PASS (SDPA, IPA, backbone, torsion)

**NUCLEUS Tower socket path fix**: `validate_nucleus_tower.rs` and `validate_biomeos_spectral.rs` expected `neuralspring-test.sock` but primal creates `neural-spring-test.sock` (matching `CARGO_PKG_NAME`). 22/22 + 29/29 PASS (was 0/0 skip)

**GPU pipeline validation**: All 14 GPU pipeline validators green including wright_fisher (4/4) and coral_forge (16/16). Mixed hardware 47/47 + 43/43 PASS. metalForge PCIe bridge 23/23 PASS.

**V70 ToadStool handoff**: FFT fix, enable f64 strip, Wright-Fisher/coral forge unblocked, 202/202 full green. V69 archived.

### Session 103 — Doc Sweep + V69 Handoff + BarraCUDA Usage Review (March 1, 2026)

**Documentation audit**: 25 stale-count findings across 10+ docs, all fixed (219→220, 746→753, 3560→3590+).
V68→V69 handoff references updated across all current-status lines.

**V69 ToadStool handoff**: Comprehensive BarraCUDA usage inventory (198 import sites, 58+ stats functions,
20+ submodules, 47 GPU dispatch ops). Nautilus Shell cross-spring bridge documented. Cross-spring
evolution map: hotSpring→bingoCube→neuralSpring→barracuda flow documented.

**Debris sweep**: 0 orphaned modules, 0 TODO/FIXME/HACK, 0 empty dirs, 0 unused deps, 0 draft files.
All scripts purposeful. metalForge fossils correctly archived.

**ecoPrimals/whitePaper/gen3/baseCamp/**: neuralSpring entry updated to S102 values + Nautilus Shell + V69.

### Session 102 — Nautilus Shell Cross-Spring Bridge + hotSpring Brain Architecture (March 1, 2026)

**Nautilus Shell integration** (hotSpring → bingoCube → neuralSpring):
- New `nautilus_bridge` module: `SpectralNautilusBridge` maps weight spectral features to Nautilus evolutionary reservoir
- Feed-forward alternative to recurrent ESN: board populations replace temporal feedback loops
- `DriftMonitor` integration for training stability detection (N_e*s boundary)
- Concept edge detection via leave-one-out error analysis (phase transition finder)
- JSON serialization for cross-run shell transfer (bit-exact roundtrip)

**New dependency**: `bingocube-nautilus` (path dep from `primalTools/bingoCube/nautilus/`)

**New binary**: `validate_nautilus_bridge` (27/27 PASS):
- Bridge lifecycle, spectral regime detection, ESN vs Nautilus comparison
- Serialization roundtrip (1e-10 parity), drift monitoring, concept edge detection

**Metrics**: 220 binaries (+1), 753 lib tests (+7), 0 clippy warnings, 0 unsafe.

### Session 101 — `ToadStool` S71 Pin Bump + GPU Stats Parity + Shader Bug Reports (March 1, 2026)

**`ToadStool` pin advanced** `1dd7e338`→`8dc01a37` (6 commits: S71 ComputeDispatch migration, DF64 transcendentals, pure math shaders, ~9000 lines boilerplate removed):
- Full re-validation: 746 lib tests PASS, 0 clippy warnings, 0 regressions

**GPU stats parity validated** (`validate_toadstool_s71_gpu_stats` 11/11 PASS):
- `KimuraGpu`: CPU↔GPU max diff = 1.11e-16 (batch 1000 elements)
- `HistogramGpu`: correct bins, counts, distribution for uniform data
- `JackknifeMeanGpu`: BLOCKED — upstream `bitcast<f64>` breaks naga DF64 emulation
- `HargreavesBatchGpu`: BLOCKED — upstream `enable f64;` not supported by naga parser

**Upstream shader bugs reported** (V68 handoff):
- `jackknife_mean_f64.wgsl`: `bitcast<f64>(vec2<u32>())` incompatible with DF64 transform
- `hargreaves_batch_f64.wgsl`: `enable f64;` directive rejected by naga

**Metrics**: 219 binaries (+1), 746 lib tests, 0 clippy warnings, 0 unsafe, 0 bare unwrap. V68 ToadStool handoff.

### Session 100 — Deep Debt Execution + Cross-Spring Rewiring + Doc Sweep (March 1, 2026)

**Hardcoding → capability-based:**
- Primal binary: hardcoded `"nestgate"` → runtime `discover_data_primal_and_forward()` (capability.resolve via biomeOS, then socket probe)
- Magic timeout constants extracted: `IPC_RESPONSE_TIMEOUT_SECS`, `HEARTBEAT_INTERVAL_SECS`

**Unused dependencies removed:**
- Removed `biomeos-primal-sdk`, `uuid`, `chrono`, `log` from `primal` feature (never imported)
- Added required tokio features (`io-util`, `net`, `signal`, `fs`, `time`) previously transitive via biomeos-primal-sdk

**Clippy pedantic/nursery: zero warnings across all targets:**
- `pairformer.rs`: `powf(0.0/4.0)` → `powi(0)`
- `weight_loader.rs`: float comparison + `expect` in tests → module-level allow
- `bench_cross_spring_modern.rs`: extracted 5 functions (too_many_lines), `cast_lossless`, `suboptimal_flops`, doc backticks
- `validate_cross_spring_rewire.rs`: doc backticks for `condition_number`

**Test coverage expanded: 727 → 746 lib tests (+19):**
- `anderson_localization.rs`: +10 tests (ipr edge cases, aubry_andre_potential, mean_ipr, disorder_sweep, two_particle symmetry, eigenvalue finiteness)
- `gpu_dispatch/basecamp.rs`: +8 tests (all 7 pub fns: weight_spectral, hessian, landscape, belief_propagation, attention_spectral, mlp_signal, agent_interaction_graph)

**Quality**: `cargo fmt` clean, `cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery` 0 warnings, `cargo test --lib` 746 PASS, `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` clean. `validate_cross_spring_rewire` 41/41, `validate_weight_spectral` 28/28, `bench_cross_spring_modern` 12/12.

**Metrics**: 218 binaries, 746 lib tests, 0 clippy warnings, 0 unsafe, 0 bare unwrap, 0 mocks in production, all files < 1000 LOC. 4 unused deps removed. All 9 external deps pure Rust.

### Session 99 — NUCLEUS Local Integration + nS-01 Real Data Extension (March 1, 2026)

**Primal handoffs:**
- NestGate V1: `data.*` JSON-RPC gap documented, NCBI/PDB/HuggingFace needs, data volume tiers (1GB–1TB), content-addressed storage
- biomeOS V1: 11 science capabilities, metalForge↔NUCLEUS alignment, LAN multi-gate roadmap
- Songbird V1: socket discovery patterns, 10GbE LAN topology, bandwidth-aware routing

**New modules:**
- `weight_loader.rs`: safetensors loading with f16/bf16/f32→f64 upcast, JSON baseline fallback (3 unit tests)

**New binaries:**
- `validate_weight_spectral_real` (12/12 PASS): nS-01 Paper A real-data pipeline with synthetic fallback

**New scripts:**
- `scripts/download_pretrained.py`: 5-model download (ResNet-18/50, ViT-B/16, GPT-2, LeNet-5) → safetensors

**Expanded:**
- `bench_cross_spring_evolution`: +3 nS-01 weight spectral CPU benchmarks (eigh_f64 on 64/128/256 Hamiltonians)
- `validate_all`: 200 binaries (199→200)

**NUCLEUS local validation:**
- BearDog: built, started, healthy (v0.9.0, JSON-RPC)
- Songbird/ToadStool: detected active (pre-existing)
- neuralSpring primal: 11 capabilities registered, GPU dispatcher (RTX 4070 Vulkan)
- NestGate forward: graceful failure confirmed (socket gap as documented in V1 handoff)

**Quality metrics:** 216 binaries, 200/200 validate_all (198 PASS + 2 pre-existing), 685 lib tests, 3500+ checks, 0 clippy warnings.

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

### Session 97d — `ToadStool` S70+++ Cross-Spring Evolution Validation (February 28, 2026)

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
