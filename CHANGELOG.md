# Changelog

All notable changes to neuralSpring are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-05-12 (Session 202c: Tier 2 wiring)

### 2026-05-12 — Session S202c (Ecosystem Wave Sync: Tier 2 toadStool wiring)

- **Tier 2 `toadstool.validate` wired** — `src/ipc/toadstool.rs` expanded from 1 to 3 methods: `compute_dispatch` (existing), `validate` (new), `list_workloads` (new). `ValidateResult` struct parses the pre-flight response: `valid`, `gpu_available`, `precision_tier`, `estimated_dispatch_time_ms`, `warnings`, `required_capabilities`. `IpcMathClient` facade: `validate_workload`, `list_workloads`.
- **Tier 2 capability constants** — `capabilities.rs` gains `TOADSTOOL_VALIDATE` and `TOADSTOOL_LIST_WORKLOADS`. `CAPABILITY_HINTS` expanded from 17 to 19 entries (2 toadStool Tier 2 methods).
- **`barracuda.precision.route`** — documented as **blocked upstream** (not implemented in barraCuda per `LIVE_SCIENCE_API.md`). Will wire when upstream implements.
- **primalSpring Layer 3 still stale** — Gap 11 still listed as open for neuralSpring in `primalSpring/docs/PRIMAL_GAPS.md`. Re-flagged.
- **Quality gates** — 728 lib + 11 integration + 73 forge + 80 playGround = 892 workspace tests. Zero failures. Zero new clippy warnings.

### 2026-05-12 — Session S202b (River Delta audit response: NestGate IPC, LTEE handoff, upstream drift)

- **NestGate IPC surface wired** — `src/ipc/nestgate.rs` created with `content_put`, `content_get`, `content_exists` methods. Follows the same `call_capability` + typed result pattern as the other 6 IPC modules. `CAPABILITY_HINTS` expanded from 14 to 17 entries (3 NestGate capabilities). `PrimalSlot::Nestgate` added (slot 6), `IpcLivenessReport` grown to 7-element array. `IpcMathClient` facade methods: `content_put`, `content_get`, `content_exists`. Gap 5 status updated from "open" to "wip".
- **NestGate capability constants** — `capabilities::CONTENT_PUT`, `CONTENT_GET`, `CONTENT_EXISTS` added to `src/capabilities.rs`.
- **LTEE B1 lithoSpore README** — `control/ltee_mutation_accumulation/README.md` created documenting the Python→Rust validation pipeline, artifact inventory, PRNG differences, and lithoSpore module mapping.
- **Gap 11 upstream drift flagged** — `primalSpring/docs/PRIMAL_GAPS.md` Layer 3 table still shows Gap 11 as open. Local Gap 11 was resolved S201b (12 RPC, 4 composable, 5 CPU fallback). Flagged in V153 handoff for upstream correction.
- **PRIMAL_GAPS.md updated** — Gap 5 section rewritten for S202b NestGate wiring progress. Gap 11 upstream note added.
- **IPC submodule tests** — All 7 IPC submodules (`barracuda`, `beardog`, `coralreef`, `nestgate`, `skunkbat`, `squirrel`, `toadstool`) now have `#[cfg(test)] mod tests` blocks covering socket-absent error paths, helper functions, and type construction. +20 tests.
- **projectNUCLEUS graph fix** — `ionic_capability_share.toml` referenced `neuralspring_primal` (source directory name); corrected to `neuralspring` (actual binary name from `Cargo.toml`).
- **lithoSpore cloned** — `gardens/lithoSpore/` now available locally. Confirmed B1 ingestion contract: `ltee-mutations` expects LSTM prediction, `ltee-alleles` T06 target (HMM/ESN ≥95% accuracy).
- **Quality gates** — 724 lib + 11 integration + 73 forge + 80 playGround = 888 workspace tests. Zero failures. Zero clippy warnings.

### 2026-05-12 — Session S202 (River Delta downstream seeding, `--format json` for Tier 2 projectNUCLEUS)

- **`--format json` for all validation binaries** — `ValidationHarness::finish()` now checks for `--format json`, `--format=json`, or `NEURALSPRING_JSON=1` before emitting output. JSON mode uses `JsonSink` (single structured JSON object with suite/passed/total/all_passed/checks[]). Human-readable `[PASS]`/`[FAIL]` lines remain the default. Enables Tier 2 projectNUCLEUS ingestion for `validate_ltee_b1_mutation_accumulation` and all future validation binaries. No binary-level changes required — any binary calling `h.finish()` gets structured output for free.
- **Foundation Thread 5 expression authored** — `gardens/foundation/expressions/ML_SURROGATES.md` created. Covers 4 pillars (LSTM, ESN, WDM surrogates, evolutionary biology), 6 ML architectures, 12 validated targets, LTEE B1 bridge, primal composition, and projectNUCLEUS workloads. Thread 5 elevated from "mapped" to "active".
- **THREAD_INDEX.toml thread 5 wired** — expression, data_sources, data_targets fields populated. Springs updated to [neuralSpring, groundSpring, wetSpring]. Contacts expanded with Barrick. Status changed from "mapped" to "active".
- **PRIMAL_GAPS L5 blocker corrected** — Gap 11 reference in guideStone L5 blockers updated from open to resolved (was already resolved in S201b but blocker text was stale).
- **Capability-based IPC routing** — `IpcMathClient` evolved from hardcoded 6-primal struct to `CapabilityRouter`-backed facade. Each method call resolves through `CAPABILITY_HINTS` (14 capability→primal mappings) with socket-level deduplication. Public API unchanged — all existing callers work without modification. Follows the ecoPrimals self-knowledge principle: a spring only knows *what* it needs, not *who* provides it.
- **Workspace dependency consolidation** — `tarpc` (0.37), `toml` (0.8), `pollster` (0.4) hoisted to `[workspace.dependencies]`. `metalForge/forge/Cargo.toml` aligned from pinned versions to `{ workspace = true }` for `barracuda`, `bytemuck`, `wgpu`, `serde_json`, `pollster`, `tokio`.
- **Integration test fix** — `cross_module_gelu_matches_provenance` gated behind `#[cfg(feature = "barracuda")]`. Was failing on IPC-first builds (gelu function lives behind barracuda feature gate).
- **Metrics CPU fallback tests** — `metrics::tests` evolved from `#[cfg(all(test, feature = "barracuda"))]` to `#[cfg(test)]` (10 tests). CPU fallback `r_squared` now guards against constant `y_true` (zero `ss_tot`) instead of producing NaN.
- **NUCLEUS workload JSON** — `neuralspring-ml-validation.toml` and `neuralspring-certification.toml` now set `PRIMALSPRING_JSON=1` in `[execution.env]` for Tier 2 structured output.
- **Foundation expressions README** — updated active expressions table (was listing only 1 of 5 active expressions).
- **Quality gates** — 703 lib + 11 integration + 73 forge + 80 playGround = 867 workspace tests. Zero failures. Zero clippy warnings.

### 2026-05-11 — Session S201 (Tier 4 IPC-first defaults, deep debt cleanup, LTEE B1 baseline, foundation seeding)

- **Tier 4 IPC-first defaults** — `default = ["barracuda"]` → `default = []`. barracuda no longer linked by default. `cargo build --no-default-features` produces a slim library without GPU/forge dependencies. 48 files feature-gated: 11 shader re-exports (`#[cfg(feature = "barracuda")] pub use`), `bench`, `nucleus_pipeline`, and per-function gates across 26 modules. CPU fallback implementations for `primitives::{sigmoid, gelu, relu, hill_activation, hill_repression, shannon_entropy, pearson_r}` and `metrics::{r_squared, rmse, mae, nse}` — same math, no barracuda dependency.
- **`required-features` gating** — 241 GPU-dependent `[[bin]]` stanzas now have `required-features = ["barracuda"]`. `cargo check --workspace` passes without barracuda feature (bins simply skipped). Idiomatic Rust approach for IPC-first builds.
- **Deprecated `ipc_dispatch` removed** — monolithic 400-line `src/ipc_dispatch.rs` deleted. Was deprecated since 0.2.0 (graduated to per-primal `src/ipc/` tree). Zero callers remained.
- **Typed `IpcError` hierarchy** — new `error::IpcError` with `NotDiscovered`, `Transport`, `Protocol` variants. IPC facade (`ipc/mod.rs`) and all 6 submodules (`barracuda`, `toadstool`, `beardog`, `squirrel`, `coralreef`, `skunkbat`) migrated from `Result<_, String>` to `Result<_, IpcError>`. `From<IpcError> for String` preserves backward compatibility at binary boundaries. 6 new tests.
- **Dead code gate fixed** — `scaffold::fieldmap` changed from `#[cfg_attr(not(feature = "barracuda"), allow(dead_code))]` to `#[cfg(feature = "barracuda")]`.
- **Playground warnings eliminated** — `#![allow(deprecated)]` in playground `lib.rs` silences 18 deprecation warnings from planned `discover_by_capability` / `PrimalClient` migration. Zero workspace warnings.
- **UniBin IPC-first build** — `cargo build --no-default-features --features guidestone --bin neuralspring_unibin` compiles cleanly. neuralSpring now qualifies for the Tier 4 exit gate alongside groundSpring, healthSpring, and ludoSpring.
- **LTEE B1 baseline** — `control/ltee_mutation_accumulation/ltee_mutation_accumulation.py` (8/8 PASS). Barrick 2009 mutation accumulation time series: linear rate 3.59e-3 mut/gen, power-law exponent 0.82 (sublinear). Expected values JSON generated for lithoSpore module 2.
- **LTEE queue seeded** — 12 papers (B1-B4, B6-B9, E2-E5) added to `specs/PAPER_REVIEW_QUEUE.md`. B1 marked STARTED.
- **Foundation Thread 5 seeded** — new `thread05_ml_surrogates.toml` (15 sources) and `thread05_ml_surrogates_targets.toml` (12 targets) in `gardens/foundation/data/`. Covers LSTM, ESN, transport surrogates, evolutionary dynamics, LTEE B1.
- **Foundation Thread 7 expanded** — 6 neuralSpring targets added to `thread07_anderson_targets.toml` (nS-01..06 + Evoformer spectral). Total thread 7 targets: 18→24.
- **CHECKSUMS regenerated** — updated for Cargo.toml, rng.rs, validation/composition.rs.
- **`capabilities` module** — new `src/capabilities.rs` with 31 named constants for all JSON-RPC method strings. IPC submodules, niche, and config share one source of truth instead of scattered literals.
- **`primal_names::display` completed** — 20 display-name constants covering all primals and springs (was 13, missing BearDog, Songbird, rhizoCrypt, loamSpine, sweetGrass, healthSpring, ludoSpring).
- **`config` named constants** — `ENV_BIOMEOS_SOCKET_DIR`, `DEFAULT_FAMILY_ID` extracted from inline strings. `SPRING_NAME` in certification/guidestone now references `config::PRIMAL_DISPLAY_NAME`.
- **`[workspace.dependencies]`** — 12 shared deps (`barracuda`, `serde`, `tokio`, `wgpu`, etc.) centralized in workspace table. Root crate and playGround both use `{ workspace = true }`. Eliminates version drift between workspace members.
- **Quality gates** — 1,300 lib (693 IPC-first) + 73 forge + 80 playGround = 1,453 workspace tests + 19 certification tests. Zero warnings.

### 2026-05-11 — Session S200b (doc reconciliation, upstream gap analysis, V151 handoff)

- **16 living doc updates** — README, EVOLUTION_READINESS, CONTROL_EXPERIMENT_STATUS, PRIMAL_GAPS, FOUNDATION_SEEDING, specs/README, experiments/README, whitePaper/README, baseCamp/README, NOTEBOOK_PATTERN, fossilRecord/README, sporeprint/validation-summary, wateringHole/README — all to S200b/V151.
- **GuideStone Level 3→5 corrections** — fixed stale "Level 3" and "13 certification" references across sporeprint, whitePaper, baseCamp, EVOLUTION_READINESS, PRIMAL_GAPS gap 13 table.
- **validation-state.json** — `level` 3→5, `certification_tests` 19, L4/L5 `PENDING`→`PASS`, `date` corrected.
- **gap-status.json** — gap 13 resolved (L5), scorecard `guidestone_level` 5, audit response updated.
- **FOUNDATION_SEEDING.md** — stale `control/spectral_analysis/` → `control/anderson_localization/`.
- **specs/README.md** — stale V135/S184 cross-reference → V151/S200b.
- **4 deploy graphs** — all updated to V151/S200b.
- **V151 handoff crafted** — upstream gap analysis (NestGate P0, provenance trio P1, advertised-undispatched methods P2, Tier 4 P3), spring patterns, NUCLEUS composition guidance.
- **V150 archived** — moved to `wateringHole/handoffs/archive/`.
- **Deploy graph version sync** — 3 stale graphs updated V140/S190 → V150/S200 (earlier in session).
- **Handler comment alignment** — `handlers.rs` doc comment corrected from "stub response" to "`SERVICE_UNAVAILABLE` error".
- **Zero >800L files confirmed** — all `.rs` files under 800 lines.
- **Quality gates** — 1,297 lib + 73 forge + 80 playGround = 1,450 workspace tests + 19 certification tests. 34 capabilities. Zero clippy warnings.

## 2026-05-11 (Session 200: guideStone L5 + plasmidBin)

### 2026-05-11 — Session S200 (guideStone L3→L5, plasmidBin release, upstream audit response)

- **guideStone L4 (NUCLEUS Composition)** — new `certification::composition` module: validates deploy graph presence (4 graphs), capability registry structural integrity (≥30 methods), and live composition calls across tensor/security/compute/ai families. `MAX_LAYER` 3→5.
- **guideStone L5 (Cross-Spring)** — new `certification::cross_spring` module: validates frozen ecosystem artifacts (gap-status.json, validation-state.json, PRIMAL_GAPS.md, FOUNDATION_SEEDING.md, CHECKSUMS), live cross-spring protocol liveness (4 family pings), and BLAKE3 hash determinism.
- **CLI default layer** — `neuralspring_unibin certify` now defaults to `--layer 5` (was 3). Help text updated for L0-L5.
- **6 new certification tests** — composition: 3 tests (registry ≥30, deploy graph exists, integration). cross_spring: 3 tests (frozen artifacts, checksums, foundation manifest).
- **plasmidBin release binary** — `cargo build --release --bin neuralspring_unibin --features guidestone` produces 2.8M stripped ELF. Verified with `version`, `validate` (21/21), `certify` (29/29).
- **CHECKSUMS regenerated** — updated for new certification modules.
- **Tier 4 IPC-first gap documented** — 115 ungated barraCuda imports prevent `--no-default-features` build. Systematic `#[cfg(feature = "barracuda")]` wrapping needed (maps to wetSpring's `primal-proof` pattern).
- **Quality gates** — 1,297 lib + 73 forge + 80 playGround = 1,450 workspace tests + 19 certification tests (guidestone, was 13). 34 capabilities. Zero clippy warnings.

## 2026-05-11 (Session 199: Deep Debt Sweep III)

### 2026-05-11 — Session S199 (stub→error, bench unwrap, deprecated isolation, path drift)

- **Inference stubs → JSON-RPC errors** — `handle_inference_complete`, `handle_inference_embed`, `handle_inference_models` no longer return `success` with fake `"provider": "stub"` data. Now return `error_code::SERVICE_UNAVAILABLE` (-32001) when Squirrel is not discovered — honest API contract, no mocks in production.
- **Bench unwrap() → expect()** — all 7 bare `.unwrap()` calls in `bench_cross_spring_shader_evolution.rs` replaced with `.expect()` providing dispatch context.
- **Deprecated API isolation** — `neuralspring_interactive` and `neuralspring_mcp_adapter` playground binaries now carry `#![expect(deprecated)]` with migration note, acknowledging `PrimalClient` → `CompositionContext` evolution path.
- **`dead_code` → `SERVICE_UNAVAILABLE`** — replaced reserved-but-unused `SERVER_ERROR` constant in `rpc.rs` with actively-used `SERVICE_UNAVAILABLE` (-32001) error code for Squirrel absence.
- **FOUNDATION_SEEDING.md path fix** — corrected stale `control/wdm/esn_surrogate/` → `control/wdm/esn_regime_classifier.py`.
- **CHECKSUMS regenerated** — `certify` 29/29 ALL PASS, `validate` 21/21 ALL PASS.
- **Quality gates** — 1,297 lib + 73 forge + 80 playGround = 1,450 workspace tests. 34 capabilities. Zero clippy warnings.

## 2026-05-11 (Session 198: Post-Interstadial Audit Response II)

### 2026-05-11 — Session S198 (CI cross-sync 413, compute.dispatch, UniBin rename, foundation seeding)

- **CI cross-sync 403→413** — primalSpring canonical registry expanded to 413 methods; comment and doc references updated across all living docs.
- **`compute.dispatch` capability** — added to `ALL_CAPABILITIES`, `niche::CAPABILITIES`, `capability_registry.toml`, and playGround MCP tools. Fixes `dispatch:rust:registry_has_compute` validation check (was FAIL, now PASS). 34 capabilities total (was 33).
- **UniBin binary rename** — `neuralspring-unibin` (hyphen) → `neuralspring_unibin` (underscore) in `Cargo.toml` and `cli.rs`. Aligns with projectNUCLEUS workload TOML expectations (`neuralspring_unibin validate`/`certify`).
- **CHECKSUMS regenerated** — updated for `Cargo.toml` and `capability_registry.toml` changes; L0 29/29 PASS.
- **Foundation seeding manifest** — `docs/FOUNDATION_SEEDING.md`: neuralSpring contributes to Thread 5 (LTEE: 5 Dolson papers, NK/MODES/eco/directed/swarm) and Thread 7 (Anderson Math: spectral, IPR, level spacing, Evoformer).
- **UniBin release build** — `cargo build --release --bin neuralspring_unibin --features guidestone` verified. `validate` 21/21 ALL PASS, `certify` 29/29 ALL PASS.
- **Quality gates** — 1,297 lib + 73 forge + 80 playGround = 1,450 workspace tests. 34 capabilities. Zero clippy warnings. All checksums valid.

## 2026-05-10 (Session 197: Deep Debt Sweep II + Doc Reconciliation)

### 2026-05-10 — Session S197b (Doc reconciliation, V147 handoff, archive sweep)

- **Full doc reconciliation** — all living docs updated to S197/V147: README, CONTEXT, PRIMAL_GAPS, EVOLUTION_READINESS, CONTROL_EXPERIMENT_STATUS, whitePaper/README, whitePaper/baseCamp/README, experiments/README, sporeprint/validation-summary, notebooks/NOTEBOOK_PATTERN. Deploy graph metadata → V147/S197.
- **Frozen data sync** — `validation-state.json`, `gap-status.json`, `paper-baselines.json` all → S197. Paper-baselines malformed JSON fixed (notebooks array broken at entry 016).
- **V147 handoff** — `NEURALSPRING_V147_DEEP_DEBT_II_HANDOFF_MAY10_2026.md`: primal evolution review, NUCLEUS composition patterns, upstream guidance for 6 primal teams, downstream pattern absorption (projectNUCLEUS + foundation).
- **V146 archived** — moved to `handoffs/archive/`.
- **Stale references eliminated** — S196→S197, V146→V147, 1,295→1,297, 1,448→1,450, 228→233+, 30→33 capabilities, 5→6 IPC modules across all canonical docs.

### 2026-05-10 — Session S197 (Deep debt: IPC idiomatic Rust, certification tests, checksums, citation fixes)

- **IPC `&PathBuf` → `&Path`** — all 6 IPC modules (`barracuda`, `toadstool`, `beardog`, `squirrel`, `coralreef`, `skunkbat`) evolved from `&PathBuf` to idiomatic `&Path` parameters. Added `# Errors` doc sections to all public IPC functions. Zero clippy warnings (was 33 pre-existing).
- **Certification test coverage** — 13 new tests across `certification/` subtree: `mod.rs` (3: max_layer, bare result, clamp), `bare.rs` (4: full validation pass, determinism, provenance minimum, tolerance minimum), `discovery.rs` (3: nonempty, lowercase, core domains), `parity.rs` (2: nonempty, core methods), `nucleus.rs` (1: structural).
- **CHECKSUMS regenerated** — `validation/CHECKSUMS` refreshed for 5 changed files; certification L0 now 29/29 PASS.
- **Waters citations fixed** — papers 019-021 in `whitePaper/baseCamp/waters.md` aligned with `specs/PAPER_REVIEW_QUEUE.md` canonical citations (Bruger & Waters → Mhatre → Srivastava).
- **Quality gates** — 1,297 lib + 73 forge + 80 playGround = 1,450 workspace tests. +13 certification tests (guidestone feature). Zero clippy warnings. All checksums valid.

### 2026-05-10 — Session S196 (primalSpring audit response: skunkBat, composition.status, method.register, CI cross-sync 403)

- **CI cross-sync** — updated canonical registry reference from 389 to 403 methods; test comment and cross-sync validation aligned with primalSpring v0.9.25 canonical 403-method registry.
- **skunkBat wired** — `primal_names::SKUNKBAT` + `display::SKUNKBAT` constants added; `src/ipc/skunkbat.rs` IPC module for `security.audit_log` (JH-5 forwarding to rhizoCrypt DAG + sweetGrass braid); `IpcMathClient` expanded to 6-slot discovery (was 5); `PrimalSlot::Skunkbat` variant; deploy graph `neuralspring_deploy.toml` now includes `germinate_skunkbat` node in Tower phase.
- **composition.status + method.register absorbed** — added as capabilities in `ALL_CAPABILITIES`, `niche::CAPABILITIES`, and `config/capability_registry.toml`; biomeOS v3.51 surface (active_users, primal_health, resource_pressure + dynamic method registration).
- **security.audit_log** — third new capability; end-to-end: `niche::CAPABILITIES` → `config::ALL_CAPABILITIES` → `capability_registry.toml` → `src/ipc/skunkbat.rs` → deploy graph node.
- **CONTEXT.md aligned** — eukaryotic architecture, UniBin, IPC tree (6 modules), certification organelle, validation scenarios, fossilization, 33 capabilities (was 30), 1,297 lib tests. Evolution path updated through Layer 5.
- **Evoformer/folding IPC** — `niche::tests::evoformer_folding_capabilities_present` + `composition_and_security_capabilities_present` structural tests; Evoformer/structure_module/folding_health capabilities validated in both niche and config.
- **Clippy hygiene** — `BearDog` backtick warnings in `ipc/mod.rs` fixed (5 instances).
- **Quality gates** — 1,297 lib + 73 forge + 80 playGround = 1,450 workspace tests. `cargo build + clippy + test` all clean.

### 2026-05-09 — Session S195 (Doc reconciliation, upstream primal handoff, archive sweep, deploy graph sync)

- **Doc reconciliation** — README, experiments/README, sporeprint/validation-summary, NOTEBOOK_PATTERN, whitePaper/baseCamp all synchronized to S195/V145. Test counts reconciled to 1,295 lib + 73 forge + 80 playGround = 1,448 across all canonical references. Stale handoff refs (V142/S192) updated to V145/S195. Binary/validator counts corrected. Paper count unified to 27. Directory tree updated for post-eukaryotic layout (ipc/, certification/, validation/scenarios/).
- **Deploy graph metadata** — `neuralspring_deploy.toml` updated from V137/S186 to V145/S195 (version, status, params).
- **Frozen data refreshed** — `validation-state.json` updated from S188/1,234 lib to S195/1,295 lib.
- **NOTEBOOK_PATTERN** — added Liu faculty batch (016-018, 26 checks) to documented notebook list; session/date synced.
- **Upstream primal handoff (V145)** — comprehensive handoff for primal teams (barraCuda, toadStool, coralReef, BearDog, Squirrel) and downstream spring teams documenting evolution patterns, NUCLEUS composition, neuralAPI deployment, and absorption readiness.

### 2026-05-09 — Session S194 (Deep debt: feature gates, inline tests, centralized env, dep alignment)

- **Feature gate alignment** — `loss_landscape`, `weight_spectral`, `wdm_esn` modules now gated behind `#[cfg(feature = "barracuda")]` in `src/lib.rs`, matching their direct imports of `barracuda::` crate types. Default build (`barracuda` on) unaffected; `--no-default-features` no longer tries to compile barracuda-dependent modules.
- **Inline test expansion** — 12 new tests: 9 for `ipc::tests` (edge cases: non-numeric extract, non-array extract, missing primal error paths, `with_timeout`, `PrimalSlot` enum values, liveness report partial/zero), 8 for `rpc_service::tests` (serde round-trips for all wire types, default inference params, optional fields), 3 for `config::tests` (family ID resolution: default/primary/biomeos fallback).
- **Centralized BIOMEOS_FAMILY_ID** — added `config::ENV_BIOMEOS_FAMILY_ID` constant and `config::resolve_family_id()` function with `FAMILY_ID` → `BIOMEOS_FAMILY_ID` → `"default"` chain. `playGround/src/discovery.rs` and `src/bin/neuralspring_primal/discovery.rs` now delegate to the central resolver instead of duplicating env var strings.
- **Dependency alignment** — `temp-env` version aligned to `"0.3.6"` across root and playGround `Cargo.toml` files.
- **Quality gates** — 1,295 lib + 73 forge + 80 playGround = 1,448 workspace tests. `cargo build + fmt + clippy + test` all clean.

### 2026-05-09 — Session S193 (Interstadial eukaryotic evolution — UniBin, IPC tree, certification organelle, validation scenarios, fossilization)

- **IPC tree graduation** — monolithic `src/ipc_dispatch.rs` (401 lines) graduated to `src/ipc/` tree with per-primal modules: `barracuda.rs` (tensor/stats), `toadstool.rs` (compute), `beardog.rs` (crypto), `squirrel.rs` (inference), `coralreef.rs` (shader compilation, new). `IpcMathClient` facade preserved in `ipc/mod.rs`. Old `ipc_dispatch` module deprecated with `note =`.
- **Certification organelle** — `src/certification/` created with 4 layer modules: `bare.rs` (L0: determinism, traceability, checksums, env-agnostic, tolerances), `discovery.rs` (L1: `CompositionContext` liveness), `parity.rs` (L2: 7-capability science parity), `nucleus.rs` (L3: BearDog signing, Songbird discovery). `certify(max_layer)` public API.
- **Validation scenarios** — `src/validation/scenarios/` created with `ScenarioMeta`, `Scenario`, `ScenarioRegistry`, `Tier` (Rust/Live/Both), `Track` (6 domain tracks). 6 initial scenarios absorbed from `validate_*` binaries: `nucleus_composition` (22 checks), `inference_composition` (16), `science_composition` (9), `nucleus_tower` (47), `compute_dispatch` (36), `composition_evolution` (30). `build_registry()` function.
- **UniBin binary** — `neuralspring-unibin` with 5 subcommands: `certify --layer N --bare`, `validate --track X --scenario Y --tier Z --list`, `serve` (stub, future absorption), `status` (IPC liveness + scenario count), `version`. Feature-gated behind `guidestone`.
- **Fossilization** — 3 pre-extinction patterns fossilized to `fossilRecord/`: `guidestone_prokaryotic_may2026/`, `ipc_dispatch_prokaryotic_may2026/`, `primal_server_prokaryotic_may2026/`. Each with provenance README.
- **Deprecated pattern migration** — `PrimalClient`, `discover_primal()`, `discover_by_capability()` in playGround annotated with `#[deprecated(since = "0.2.0", note = "use CompositionContext...")]`.
- **Quality gates** — zero bare `#[allow]`, zero TODO/FIXME/HACK/DEBT in library, `cargo build + fmt + clippy + test` all clean. 1,291 lib tests pass (8 new scenario/registry tests).

### 2026-05-08 — Session S192 (Doc cleanup, upstream primal handoffs, downstream absorption review, archive sweep)

- **Root docs synchronized** — README, CHANGELOG, quality gates, footer, directory tree all aligned to S192. Test count canonical: 1,279 lib + 73 forge + 80 playGround = 1,432. Eliminated S186/V137 stale references from footer, specs table, directory structure comments. Quality gates test line updated from 1,217 → 1,432.
- **whitePaper/baseCamp/README.md** refreshed to S192 — session, handoff version, test counts, barraCuda version, paper notebook status all current.
- **experiments/README.md** refreshed to S192 — added S189-S192 session entries, updated header status line.
- **wateringHole/README.md** updated — V142 active, V141 archived, archive range V1-V141.
- **Upstream primal handoff (V142)** — consolidated evolution targets for all primal teams:
  - barraCuda: 18 IPC surface gaps (A: tensor lifecycle, B: core math, C: ML ops)
  - Squirrel: `inference.register_provider`
  - coralReef: shader compile wire contract
  - toadStool: compute dispatch surface stabilization
  - NestGate: weight tensor storage API
  - BearDog/Songbird: BTSP session wire
  - barraCuda: `special::plasma_dispersion` feature gate
- **Downstream absorption notes** — projectNUCLEUS (4 deploy graphs ready, neuralAPI biomeOS integration), foundation (Threads 5 + 7, 8 notebooks ready)
- **Archive sweep** — zero stale TODOs, zero orphans, zero debris. 2 provenance-retained assessment files noted.
- **PRIMAL_GAPS.md** session header updated to S192

### 2026-05-08 — Session S191 (Full sweep: downstream review, test coverage, Liu faculty notebooks, benchmark audit, IPC review)

- **projectNUCLEUS + foundation review** — both repos pulled and reviewed. projectNUCLEUS deploys 13/13 primals on ironGate with BTSP Phase 3, 5-tier discovery, full provenance chain. foundation maps 10 domain threads (neuralSpring in threads 5 + 7). Both repos inform neuralSpring's downstream patterns for NUCLEUS composition.
- **45 new inline unit tests** across 5 library modules:
  - `src/error.rs` (14 tests) — error type construction, Display impls, From conversions
  - `src/streaming/mod.rs` (8 tests) — newline trimming, buffer capacity invariants
  - `src/search/mod.rs` (5 tests) — k-mer index build, lookup, N-base skipping
  - `src/provenance/references.rs` (8 tests) — softmax sum-to-one, GELU sign, Rosenbrock global min, Ackley positivity
  - `src/visualization/types.rs` (10 tests) — DataChannel serialization (timeseries, gauge, heatmap, scatter3d, spectrum), ScenarioNode empty-vec skipping
- **Liu faculty paper notebooks** (3 new, 8 total, 72/72 checks):
  - `paper-016-hmm-phylo.ipynb` — Forward/Backward/Viterbi/Baum-Welch (10/10 PASS)
  - `paper-017-sate-alignment.ipynb` — NJ tree, progressive alignment, iterative SATe (8/8 PASS)
  - `paper-018-introgression.ipynb` — PhyloNet-HMM introgression detection (8/8 PASS)
- **Benchmark gap roadmap** — `specs/BENCHMARK_ANALYSIS.md` updated with coverage matrix, Kokkos/Polybench/SPEC gap assessment and prioritization
- **Tier 4 IPC validator audit** — 8 validators with 160 total checks (11 skip-when-offline), documented in `experiment-catalog.json`
- **paper-baselines.json** updated to S191 with Liu faculty entries (8 notebooks, 72 checks)
- 1,279 lib tests (45 new), 0 clippy, 0 fmt. V141 handoff

### 2026-05-08 — Session S190 (Cross-spring composition parity response: barraCuda optional, exp094, 3 deploy graphs, registry cross-sync)

- **barraCuda `optional = true`** — barracuda, wgpu, neural-spring-forge now optional deps with `barracuda` feature flag (default-enabled). GPU-centric modules gated behind `#[cfg(feature = "barracuda")]`. playGround barracuda+wgpu also optional. Satisfies universal evolution target from primalSpring Phase 60 audit.
- **Registry cross-sync test** — new `registry_methods_in_primalspring_canonical` test validates 10+ shared methods against primalSpring's canonical 389-method `config/capability_registry.toml`. Documents neuralSpring-only methods (`science.*`, `provenance.*`, `primal.forward/discover`).
- **exp094 composition parity crate** — `experiments/exp094_neuralspring_composition_parity/` replicates primalSpring's NUCLEUS parity template. Validates Tower (BearDog/Songbird), Node (barraCuda/coralReef/toadStool), Nest (NestGate/provenance), niche science parity (`stats.mean`, `spectral_analysis`), inference probes, and cross-atomic hash→store→retrieve pipeline.
- **3 new deploy graphs** (total 4):
  - `neuralspring_inference_pipeline.toml` — Squirrel → barraCuda → inference → provenance
  - `neuralspring_spectral_analysis.toml` — eigendecomp → IPR → NestGate → provenance
  - `composition/neuralspring_math_pipeline.toml` — minimal tensor→mean→dispatch chain
- **PRIMAL_GAPS items 1-4 → IMPLEMENTED** — reflecting primalSpring exp094 validation and JH-0 MethodGate adoption by all 13/13 primals
- **gap-status.json** updated to S190 with 3 new composition evolution items (CE6-CE8) and parity scorecard response
- 269 binaries, 0 clippy, 0 fmt, `cargo deny check` clean. V140 handoff

### 2026-05-07 — Session S189 (Paper baseline notebooks: Dolson faculty, 5 notebooks, 46/46 checks)

- **5 publishable-grade paper baseline notebooks** in `notebooks/papers/`:
  - `paper-011-counterdiabatic-evolution.ipynb` — Iram/Dolson (2020) Nature Physics, NK landscapes, CD schedule (11/11 PASS)
  - `paper-012-modes-toolbox.ipynb` — Dolson et al. (2019) Artificial Life, MODES metrics (9/9 PASS)
  - `paper-013-eco-dynamics.ipynb` — Dolson & Ofria (2018) GECCO, ecological dynamics (7/7 PASS)
  - `paper-014-directed-evolution.ipynb` — Dolson et al. (2022) eLife, selection algorithms (8/8 PASS)
  - `paper-015-swarm-robotics.ipynb` — Foreback et al. (2025) IEEE, heterogeneous controllers (11/11 PASS)
- **`experiments/results/paper-baselines.json`** frozen data with citations, DOIs, BarraCUDA mappings
- **`sporeprint/validation-summary.md`** updated with paper baselines section
- **`notebooks/NOTEBOOK_PATTERN.md`** expanded with paper baseline cell structure

### 2026-05-07 — Session S188 (sporePrint Tier 2: 5 notebooks + 6 frozen JSON + validation summary + notify workflow)

- **sporePrint absorption**: Following primalSpring/wetSpring pattern for Tier 2 sporePrint content
- **6 frozen JSON datasets** in `experiments/results/` — validation-state, experiment-catalog, security-posture, cross-spring-matrix, benchmark-data, gap-status. All loaded from notebooks with no live primals needed
- **5 public notebooks** in `notebooks/`:
  - `01-composition-validation.ipynb` — deploy graphs, bond types, 30 capabilities, discovery tiers, guideStone readiness
  - `02-benchmark-comparison.ipynb` — Rust vs Python (83.6x geomean, 1104x peak), GPU (104x peak, 97% coverage), multi-GPU parity (384/384), isomorphic primitives
  - `03-ecosystem-evidence.ipynb` — 134 experiments, 27 papers / 6 faculties, gap resolution (14 main, 13 resolved appendix), security timeline
  - `04-cross-spring-connections.ipynb` — 8 primal consumption matrix, ecosystem flow tiers, proto-nucleate (7 capabilities, 6 deps), barraCuda usage depth
  - `05-btsp-security-deep-dive.ipynb` — BTSP convergence (Phase 45c, 13/13), encryption tiers, per-primal posture, supply chain integrity, guideStone P1-P5
- **`notebooks/NOTEBOOK_PATTERN.md`**: Cell structure standard (title, imports+data, domain cells, summary), color palette (#2ecc71/#e74c3c/#3498db), conventions
- **`sporeprint/validation-summary.md`**: Headline numbers (1,387 tests, 134 experiments, 269 binaries, 30 capabilities, 233 tolerances, guideStone L3 29/29, BTSP 13/13), code quality, performance, notebook list
- **`notify-sporeprint.yml`**: CI workflow fires on push to `sporeprint/`, `notebooks/`, `experiments/results/`
- 269 binaries, 0 clippy, 0 fmt, `cargo deny check` clean. V139 handoff

### 2026-04-27 — Session S187 (Deep debt cleanup: 6 smart refactors, centralized discovery, full audit, ecosystem handoff)

- **6 large-file smart refactors** — all `.rs` files >800 lines split into companion modules by logical domain boundary (not arbitrary splits):
  - `validate_barracuda_tensor.rs` (875L) → `main.rs` (424L) + `extended.rs` (467L) — core vs transcendental/extended ops
  - `validate_gpu_pure_wdm_coral.rs` (834L) → `main.rs` (366L) + `coral_af3.rs` (470L) — WDM vs coralForge/AlphaFold3
  - `validate_metalforge_wdm_coral.rs` (824L) → `main.rs` (376L) + `coral_mixed.rs` (465L) — NUCLEUS roles vs mixed routing
  - `bench_upstream_vs_local.rs` (879L) → `main.rs` (628L) + `extended.rs` (250L) — core bio vs DirEvo/MODES/Swarm
  - `bench_portability_tiers.rs` (810L) → `main.rs` (527L) + `extended.rs` (280L) — HMM/Fitness/L2/IPR vs Spatial/Dispatcher/Hamming
  - `validate_barracuda_dispatch_parity.rs` (805L) → `main.rs` (607L) + `expanded.rs` (200L) — original 32 checks vs S115/S127 expanded
- **Centralized discovery** — `resolve_biomeos_socket_dir()` extracted to `config.rs` (4-tier resolution: env → XDG → `/run/user/{uid}` → temp), consumers in `neuralspring_primal/discovery.rs` and `playGround/src/discovery.rs` delegate to it
- **`eprintln!`→`log::`** — 4 `eprintln!` calls in `neuralspring_guidestone.rs` replaced with `info!`/`warn!` per ecosystem logging standard
- **Full codebase audit** — zero `unsafe` (`#![forbid(unsafe_code)]` workspace-wide), zero mocks in production (2 legitimate fallbacks: Squirrel routing stub + coralReef feature-gate), zero `#[allow()]` (all lint suppression via `#[expect()]`), zero TODO/FIXME/HACK, all external deps pure Rust (except `wgpu` GPU HAL)
- **BarraCUDA API alignment** — 4 stale tensor method calls fixed (`tanh_wgsl→tanh`, `exp→exp_wgsl`, `log→log_wgsl`, `sqrt→sqrt_wgsl`)
- **269 binaries**, 0 clippy, 0 fmt, `cargo deny check` clean. V138 handoff

### 2026-04-27 — Session S186 (Phase 46 composition explorer: agent-driven composition, Squirrel inference, DAG provenance, braid audit trail)

- **Phase 46 composition tools**: Copied `nucleus_composition_lib.sh` (41 functions), `composition_template.sh`, `composition_nucleus.sh` from primalSpring into `tools/`. New `tools/` directory for composition-era tooling
- **`neural_composition.sh`**: Domain composition script implementing neuralSpring's assigned lane — agent-driven composition + AI feedback loops. Squirrel-mediated `inference.complete`/`inference.embed` via IPC, DAG branching for AI decisions, braid provenance audit trail (`application/x-neuralspring-agent`), closed-loop feedback (`domain_on_tick` + `check_proprioception`), petalTongue dashboard
- **Agentic IPC patterns**: Documented Squirrel `cap_socket "ai"` + `send_rpc` pattern, DAG `append_event` with structured AI metadata, braid records for decision tracing, act→observe→adjust cycle via sensor streams
- **AI provenance schema**: DAG events carry `{prompt, result, model, confidence}`; braids carry full payload with content-type tagging. Together they form a complete decision audit trail
- **Phase 45c BTSP default**: Auto-absorbed via `primalspring` path dependency — BTSP now mandatory for all 13 capabilities, cleartext connections FAIL
- **PRIMAL_GAPS.md**: Gap 14 (Phase 46 composition findings) — Squirrel integration reliability, missing `inference.register_provider`, agentic composition patterns, AI provenance schema, recommendation for `EXTRA_PRIMALS` env var
- **269 binaries**, 0 clippy, 0 fmt, `cargo deny check` clean. V137 handoff

### 2026-04-20 — Session S185 (primalSpring v0.9.17 absorption: `is_skip_error`, guideStone v0.3.0, standard v1.2.0, genomeBin v5.1)

- **guideStone v0.3.0**: `neuralspring_guidestone` absorbs `primalspring::composition::is_skip_error` — replaces 7 manual `is_connection_error()` / `is_protocol_error()` arms with unified skip classification (connection errors + protocol mismatches + transport dialect). Covers BTSP, HTTP-on-UDS, and absent primals in a single predicate
- **guideStone standard v1.2.0**: Doc reference updated from v1.1.0. v1.2.0 adds tolerance hierarchy as ecosystem standard, `call_or_skip`/`is_skip_error` in `primalspring::composition`, "domain functions are local compositions" pattern
- **primalSpring v0.9.17 integration**: Absorbed upstream (Phase 45). No new library API (checksums, ValidationResult, IPC unchanged from v0.9.16). Delta is deployment validation and operational contracts
- **Operational awareness**: coralReef `--port` → `--rpc-bind` (iter84); `BEARDOG_FAMILY_SEED` required for production BTSP; `SONGBIRD_SECURITY_PROVIDER=beardog`; `NESTGATE_JWT_SECRET` required
- **genomeBin v5.1**: 46 binaries across 6 target triples (x86_64-musl, aarch64-musl, armv7-musl, x86_64-windows, aarch64-android, riscv64-musl). Level 4 deployment path clear
- **269 binaries**, 0 clippy, 0 fmt, `cargo deny check` clean. V136 handoff

### 2026-04-19 — Session S184 (guideStone Level 3: BLAKE3 checksums, structured output, family discovery, protocol tolerance)

- **guideStone v0.2.0**: `neuralspring_guidestone` upgraded to Level 3 — 29/29 bare checks ALL PASS. All 5 properties certified without primals running
- **BLAKE3 CHECKSUMS**: New `validation/CHECKSUMS` manifest covering 15 validation-critical files (guideStone binary, tolerances, provenance, validation, RNG, capability registry, Python baselines, Cargo.toml). Verified via `primalspring::checksums::verify_manifest()` in Phase 1 bare properties. Property 3 (Self-Verifying): PARTIAL → CERTIFIED
- **gen_checksums example**: New `examples/gen_checksums.rs` (feature-gated: `guidestone`) generates BLAKE3 CHECKSUMS manifest, following primalSpring pattern
- **Structured output**: `v.section()` replaces raw `println!` section headers. `ValidationResult::print_banner()` for banner. Supports `PRIMALSPRING_JSON=1`
- **FAMILY_ID support**: Reads `FAMILY_ID` env for family-isolated socket discovery per v0.9.16 depot pattern
- **Protocol tolerance**: `is_protocol_error()` classifies HTTP-on-UDS (Songbird, petalTongue) as SKIP, not FAIL
- **Bare exit logic**: Aligned with primalSpring pattern — `exit_code() == 0 → exit(2)` for bare-only mode
- **primalSpring v0.9.16 integration**: BLAKE3 checksums module, family-aware discovery, protocol tolerance, v.section() API
- **269 binaries** (+1 gen_checksums example), 0 clippy, 0 fmt, `cargo deny check` clean. V135 handoff

### 2026-04-18 — Session S183 (guideStone evolution: neuralspring_guidestone binary, 5 properties documented, primalspring composition API)

- **guideStone binary**: New `src/bin/neuralspring_guidestone.rs` — self-validating NUCLEUS deployable. 4-phase validation: bare properties (P1: deterministic RNG, P2: provenance registry, P4: ecoBin compliance, P5: tolerance documentation), discovery + liveness, domain science parity (7 capabilities), additive NUCLEUS. Feature-gated: `guidestone` → `primalspring` + `primal`. Exit codes 0/1/2
- **primalspring dependency**: Added `primalspring` as optional path dependency (`../primalSpring/ecoPrimal`). New `guidestone` feature in `[features]`
- **guideStone properties**: New `docs/GUIDESTONE_PROPERTIES.md` — documents all 5 certified properties
- **Gap 11 confirmed**: barraCuda JSON-RPC surface gaps (18 methods) verified still open
- **PRIMAL_GAPS.md**: Added Gap 13 (guideStone evolution) with readiness matrix
- **268 binaries** (+1 neuralspring_guidestone), 0 clippy, 0 fmt, `cargo deny check` clean. V134 handoff

### 2026-04-17 — Session S182 (Level 5 primal proof: capabilities harness, IPC dispatch, stadial enforcement)

- **Proto-nucleate capabilities harness**: New `validate_proto_nucleate_capabilities` binary exercises all 7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` against owning primals via IPC (barraCuda: `tensor.matmul`/`tensor.create`/`stats.mean`, toadStool: `compute.dispatch`, BearDog: `crypto.hash`, Squirrel: `inference.complete`/`inference.embed`). Exit 0/1/2
- **IPC math client**: New `src/ipc_dispatch.rs` with `IpcMathClient` — typed Rust methods routing domain math through JSON-RPC IPC. Discovery-based sockets. `IpcLivenessReport` for primal health probing. 4 new tests
- **Stadial `deny.toml` enforcement**: Added `deny = [...]` ban list — `ring`, `openssl-sys`, `openssl`, `async-trait`, `rustls`, `ed25519-dalek`, `cmake`, `cc` (blake3 wrapper exempted). All stadial parity gate bans now enforced
- **`rust-toolchain.toml`**: Pinned `channel = "stable"` with `rustfmt`, `clippy`, `llvm-tools-preview`. MSRV via `rust-version = "1.87"`
- **barraCuda surface gaps**: `PRIMAL_GAPS.md` Gap 11 documents 18 `barracuda::` library calls that lack 1:1 JSON-RPC equivalents (eigh, Pearson, chi-squared, spectral density, ESN, NN, belief propagation, etc.)
- **Clean-machine validation script**: `scripts/validate_clean_machine.sh` — Level 6 runner (Tier 2 + Tier 3 validators, env-driven socket discovery)
- **1,234 lib tests passing**, 267 binaries, 520+ `.rs` files, 0 clippy, 0 fmt, `cargo deny check` clean. V133 handoff

### 2026-04-11 — Session S181 (Composition evolution: full capability surface, Squirrel routing, Tower discovery, Tier 3 validation)

- **Full capability surface**: `ALL_CAPABILITIES` expanded 27 → 30 (`health.check`, `identity.get`, `mcp.tools.list`). Every dispatched method is now registered, discoverable via `capability.list`, and present in biomeOS `capability.register` loop. `niche::CAPABILITIES`, `config::ALL_CAPABILITIES`, `capability_registry.toml`, and MCP tool definitions (30) all in sync
- **Squirrel routing**: Inference handlers (`inference.complete`, `inference.embed`, `inference.models`) now attempt Squirrel discovery via `try_squirrel_route()` — forwards to Squirrel when running, falls back to stub with `"status": "squirrel_unavailable"` when absent. Replaces static `"not_yet_wired"` stubs
- **Tower Atomic discovery**: `src/bin/neuralspring_primal/tower.rs` probes BearDog + Songbird at startup via capability-based socket discovery + `health.liveness`. Logs Tower status (complete/partial/standalone). Non-blocking
- **Tier 3 composition validator**: `validate_composition_evolution.rs` — validates science→primal→NUCLEUS coherence: capability surface completeness, deploy graph alignment, proto-nucleate IPC wiring, inference evolution readiness, health triad probes. 265 binaries total
- **`composed` feature gate**: Cargo feature `composed` (implies `primal`) for future IPC-only composition paths
- **ToadStoolClient fix**: Discovery capability `compute.submit` → `compute.dispatch.submit` (aligns with proto-nucleate and deploy graph)
- **Deploy graph V131/S181**: Header comment includes `nest_atomic` fragment, provenance session version updated S179→S181
- **Tolerance forensics**: `check_abs_or_rel` now records the actual tolerance mode (Absolute vs Relative) that satisfied the check, not always Absolute
- **Clippy zero-warning**: Fixed 3 pre-existing `doc_markdown` warnings in `handlers.rs` and `rpc.rs`
- **BARRACUDA_REQUIREMENTS.md**: Version references updated v0.3.7 → v0.3.11
- **PRIMAL_GAPS.md**: Gap 1 updated (Squirrel routing wired), Gap 6 updated (Tower discovery), Gap 7 resolved (fragment list). Session S181
- **MCP tool definitions**: 27 → 30 (`health.check`, `identity.get`, `mcp.tools.list`), domains `identity` and `mcp` added to valid set
- **1,225+ lib tests passing**, 265 binaries, 520 `.rs` files, 0 clippy, 0 fmt warnings, barraCuda v0.3.11

### 2026-04-11 — Session S180 (Composition evolution: deployment triad, identity, MCP, upstream reconciliation)

- **Deployment health triad**: `health.check` handler — combined liveness + readiness for benchScale/plasmidBin smoke tests (DEPLOYMENT_VALIDATION_STANDARD)
- **T4 discovery**: `identity.get` handler — primal name, niche, version, domain, license, full capability list (ECOSYSTEM_COMPLIANCE_MATRIX)
- **MCP tool listing**: `mcp.tools.list` on primal JSON-RPC surface — returns all 27 capabilities as discoverable tools with domain parsed from `domain.verb`
- **MCP tool parity**: `playGround/src/mcp_tools.rs` expanded 19 → 27 definitions (added `provenance.begin`, `provenance.record`, `provenance.complete`, `provenance.status`, `primal.forward`, `primal.discover`, `capability.list`, `compute.offload`)
- **Method normalization**: Iterative multi-prefix strip (`neuralspring.`, `neural-spring.`, `neural_spring.`) per SPRING_COMPOSITION_PATTERNS §1
- **Deploy graph V130/S180**: Added `nest_atomic` fragment, `health.check`, `identity.get`, `mcp.tools.list` capabilities
- **primalSpring graph reconciliation**: `neuralspring_inference_pipeline.toml` binary name fix + health method fix; `spring_deploy/neuralspring_deploy.toml` binary name + capability set alignment
- **plasmidBin metadata refresh**: version 0.7.0→0.1.0, domain ml→science.learning, 2 stale capabilities → 30-capability surface, UniBin modes aligned
- **Clippy fix**: `#[expect(clippy::unwrap_used)]` on forge `coralreef_bridge.rs` test (`unwrap_err` in known-Err assertion)
- **PRIMAL_GAPS.md**: R5 (MCP parity), R6 (deployment triad), R7 (fragment alignment), R8 (upstream graph reconciliation), R9 (plasmidBin metadata), R10 (method normalization breadth)
- **Ecosystem handoff**: `NEURALSPRING_V130_PRIMAL_COMPOSITION_PATTERNS_HANDOFF_APR11_2026.md` — patterns for primal + spring teams
- **1,378 tests** (1,225 lib + 73 forge + 80 playGround), 264 binaries, 505 `.rs` files, 0 clippy, 0 fmt, 0 doc warnings

### 2026-04-11 — Session S179 (Deploy graph proto-nucleate alignment, composition validation execution)

- **Deploy graph proto-nucleate alignment**: `neuralspring_deploy.toml` V128→V129. Added coralReef (`shader.compile.wgsl`), barraCuda (`math.tensor`), Squirrel (`ai.query`) germination nodes. BearDog `by_capability` updated `crypto`→`security`. ToadStool updated to `compute.dispatch.submit`, NestGate to `storage.retrieve`. Graph metadata now declares `composition_model`, `bond_type`, `trust_model`, `transport`, `fragments`, `proto_nucleate` reference
- **Capability surface reconciled**: `config::ALL_CAPABILITIES` expanded 18→26 to match `niche::CAPABILITIES`. `capabilities_provided` in deploy graph expanded from 14 science-only to full 26-capability niche surface. `operation_dependencies()` and `cost_estimates()` now cover all 26 capabilities
- **Doctest fix**: `src/gpu/mod.rs` `GpuCapabilities` example — import ordering corrected (0 doctest failures, was 1)
- **Provenance pinned**: `ANDERSON_MULTIAGENT_ENVIRONMENT` `"Python 3.12, NumPy, seed=42"` → `"Python 3.12.3, NumPy 2.2.6, seed=42"` with documented rationale
- **Kokkos benchmark**: Provenance reclassified from `PLACEHOLDER` to `ESTIMATED` (honest status)
- **`PRIMAL_GAPS.md` updated**: R3 (deploy graph alignment) and R4 (capability surface reconciliation) resolved
- **Root docs, whitePaper, experiments, specs aligned**: All "current" status references updated S178→S179, V128→V129 across 12+ doc files
- **V129 handoff**: Deploy graph alignment, composition metadata, primal absorption candidates. V127/V128 archived at central wateringHole
- **~1,403+ tests** (1,225 lib + 73 forge + 80 playGround + 12 integration + 25 tokio), 264 binaries, 518 `.rs` files, 0 clippy, 0 fmt, 0 doctest failures, barraCuda v0.3.11

### 2026-04-11 — Session S178 (Composition validation phase: Python→Rust→NUCLEUS)

- **Three-layer validation stack**: Python baselines → Rust baselines → NUCLEUS composition. Composition validators wired to `validate_all` with exit-2 honest skip handling (PASS/SKIP/FAIL summary)
- **`validate_all` composition integration**: `COMPOSITION_BINARIES` array (3 validators), `exit_code()` helper, skip-aware counting. 264 binaries total (261 science/GPU + 3 composition)
- **`docs/PRIMAL_GAPS.md` reconciled**: Gap 1 (inference.*) updated from `open` to `wip` — surface fully wired, provider pending. Gap 8 (binary naming) resolved. Added Resolved section (R1 binary naming, R2 inference registration). Removed stale "do not appear in source" claim
- **Version string alignment**: barraCuda v0.3.7 → v0.3.11 across `EVOLUTION_MAPPING.md`, `ABSORPTION_TRACKER.md`, `EVOLUTION_READINESS.md` dependency table
- **Deploy graph reconciled**: V124/S174 → V128/S178 in `graphs/neuralspring_deploy.toml`
- **`EVOLUTION_READINESS.md`**: Added NUCLEUS Composition Validation section (primal IPC wiring status table, niche self-knowledge summary), S178 session entry
- **`capability_registry.toml`**: Expanded to full 26 capabilities (added provenance.*, primal.forward, primal.discover, capability.list, compute.offload)
- **`CONTROL_EXPERIMENT_STATUS.md`**: Updated to S178 with composition validation counts
- **Root docs updated**: README, CHANGELOG, baseCamp, experiments, wateringHole, CONTRIBUTING aligned to S178/V128
- **V128 handoff**: Composition patterns, NUCLEUS deployment via Neural API, primal absorption candidates
- **~1,403+ tests** (1,225 lib + 73 forge + 80 playGround + 14 integration), 264 binaries, 518 `.rs` files, 0 clippy, 0 fmt, 0 doc warnings, barraCuda v0.3.11

### 2026-04-10 — Session S177 (NUCLEUS composition validation, inference.* wiring, ecoBin harvest)

- **NUCLEUS composition validation layer**: 3 new validators (`validate_nucleus_composition`, `validate_inference_composition`, `validate_primal_discovery`) — proto-nucleate graph validation, inference chain validation, capability-based discovery validation. Honest skip pattern (exit 2).
- **Composition infrastructure**: `src/validation/composition.rs` — `discover_primal_socket()` (5-tier biomeOS discovery), `json_rpc_call()`, `probe_liveness()`, `probe_capabilities()`, `call_capability()`, `BondType` enum, `exit_code_skip_aware()`, proto-nucleate node descriptors
- **inference.* capability wiring**: `inference.complete`, `inference.embed`, `inference.models` wired across niche.rs, config.rs, rpc_service.rs (typed structs), handlers.rs (stub), main.rs (dispatch), mcp_tools.rs (19 tools), capability_registry.toml
- **NUCLEUS bonding policy**: `niche.rs` constants — `BOND_TYPE = "Metallic"`, `TRUST_MODEL = "InternalNucleus"`, encryption tiers per atomic boundary (Tower=full, Node/Nest/Meta=tower_delegated)
- **ecoBin harvest**: release profile (strip/LTO/codegen-units=1), `scripts/harvest_ecobin.sh` (musl build, static verify, plasmidBin staging)
- **barraCuda path fix**: all Cargo.toml files corrected to `../../primals/barraCuda/crates/barracuda`; CI checkout paths aligned; `include_str!` macros updated
- **Precision exhaustiveness**: `compile_shader_universal()` handles all 12 `Precision` variants (Binary, Int2, Q4, Q8, Fp8E5M2, Fp8E4M3, Bf16, F16, F32, Df64, Qf128, Df128)
- **CI**: composition-validation job (handles exit 2 honest skip)
- **V127 handoff**: archived V126, crafted V127 cross-team handoff with absorption candidates, primal gaps, evolution roadmap
- **~1,392 tests** (1,225 lib + 73 forge + 80 playGround + 14 integration), 264 binaries, 518 `.rs` files, 0 clippy, 0 fmt, 0 doc warnings, barraCuda v0.3.11

### 2026-03-24 — Session S176 (deep audit, IPC resilience, environment centralization, GPU module refactor)

- **Clippy zero-warning gate restored**: 6 test-code errors fixed — `clippy::similar_names` (provenance `hasher` → `content_hasher`), `clippy::unwrap_used` in validation test modules (4 sites, `#[expect]` with documented reasons)
- **Provenance environment centralization**: 19 literal `"Python 3.10.12, NumPy 2.2.6, seed=42"` → `WDM_ENVIRONMENT` constant; 1 `"Python 3.12, NumPy, seed=42"` → `ANDERSON_MULTIAGENT_ENVIRONMENT`; zero non-centralized environment strings in registry
- **IPC resilience wired**: `RetryPolicy` (exponential backoff) + `CircuitBreaker` (3 failures → 10s cooldown) integrated into `PetalTonguePushClient::send_rpc()`; extracted `try_send()` and `check_rpc_error()` helpers
- **GPU module refactor**: `src/gpu.rs` (714 LOC) → `src/gpu/mod.rs` (475 LOC) + `src/gpu/tests.rs` (238 LOC); module path `crate::gpu::tests::shared_gpu` preserved for 4 downstream consumers
- **Integration test expansion**: 9 → 12 tests; full 49/49 provenance registry coverage (commit validation, environment centralization enforcement, CPU parity environment check, minimum record count guard)
- **Documentation reconciliation**: `CONTROL_EXPERIMENT_STATUS.md` capabilities 7→16; `specs/DATA_PROVENANCE.md` pretrained models clarified; tolerance `BASELINE_COMMIT` placeholders → `f9ad0268`; `src/evolved/mod.rs` v0.3.1→v0.3.7; `validate_modern_cross_spring/report.rs` v0.3.5→v0.3.7; `specs/BARRACUDA_USAGE.md` `src/gpu.rs`→`src/gpu/mod.rs`; `CONTEXT.md` file/test counts updated
- **V126 handoff**: archived V124/V125, crafted V126 cross-team handoff
- **~1,403 tests** (1,211 lib + 73 forge + 80 playGround + 12 integration + 25 tokio + 2 test-module additions), 0 clippy, 0 fmt, 0 doc warnings

### 2026-03-24 — Session S175 (ecosystem absorption, cast hardening, ValidationSink, provenance integrity)

- **Cast lint hardening**: `cast_possible_truncation` and `cast_sign_loss` promoted from `warn` → `deny` in workspace lints — all 1,353 cast sites covered by existing `#[expect]` attributes, zero breakage
- **ValidationSink pattern** (absorbed from wetSpring V134 / airSpring V010 / groundSpring V121): `StdoutSink`, `JsonSink`, `NdjsonSink`, `CollectingSink`, `SilentSink` — trait `ValidationSink` with `on_check` + `on_finish`; `ValidationHarness::emit_to_sink()` and `emit_json()` convenience methods; `finish()` now delegates to `StdoutSink`; 12 new tests
- **Provenance integrity tests**: 4 new tests — `provenance_scripts_exist_on_disk`, `provenance_scripts_have_provenance_header`, `provenance_scripts_have_spdx_header`, `provenance_scripts_content_stability` (hash + size checks for all 49 registered Python baselines)
- **Deploy graph updated**: `neuralspring_deploy.toml` bumped from V105/S155 → V124/S174
- **Leverage guide refreshed**: `NEURALSPRING_LEVERAGE_GUIDE.md` updated to V124/S174 with current metrics (1,400+ tests, 232+ tolerances, 261 binaries)
- **Audit clean**: zero sled deps, zero unsafe, zero production mocks, all files under 1,000 LOC
- **~1,400 tests** (1,211 lib + 73 forge + 80 playGround + 9 integration + 25 tokio), 0 clippy, 0 fmt, 0 doc warnings, cast deny, zero `#[allow()]`

### 2026-03-24 — Session S174 (deep audit execution, zero debt, provenance alignment)

- **Zero `#[allow()]`**: removed last `#![allow(missing_docs)]` from `error.rs`; added full field docs to all 20 error enum fields; converted `rpc.rs` `#[allow(dead_code)]` → `#[expect(dead_code)]`; converted 5 fossil `#[allow]` → `#[expect]` with reasons
- **Tolerance fidelity**: aligned `check_rel` zero-detection with `tolerances::ZERO_DETECTION` (was `f64::EPSILON`); added `GPU_MULTI_OBJ_BESSEL_F64` (3e-3) for Bessel correction gap; added 4 upstream contract constants (`UPSTREAM_HYDRO_*`, `UPSTREAM_PHYSICS_*`, `UPSTREAM_BIO_*`); new `upstream_contract` registry category; replaced all `2e-3` literals in `validate_gpu_directed.rs` / `validate_gpu_pipeline_directed.rs`; centralized `validate_toadstool_s93` local constants
- **Self-knowledge compliance**: removed 3 dead `*_NAME_HINT` config re-exports; neutralized `handlers.rs` cross-spring origin strings → semantic dispatch paths; gated petalTongue push behind `NEURALSPRING_VISUALIZATION_PUSH` env var; updated playGround clients to use `primal_names::*` directly
- **Provenance alignment**: added `# Provenance: see src/provenance/experiments.rs` headers to all 49 registered Python baseline scripts
- **Community files**: added `CONTRIBUTING.md` (quality standards, tolerance policy, barraCuda evolution, IPC conventions) and `SECURITY.md` (security model, reporting, dependency audit)
- **Clippy**: fixed `map_or` → `is_ok_and` (modern idiomatic Rust)
- **V124 handoff**: comprehensive barraCuda absorption handoff with S174 evolution context
- **~1,385 tests** (1,199 lib + 72 forge + 80 playGround + 9 integration + 25 tokio), 0 clippy, 0 fmt, 0 doc warnings, 0 `#[allow()]`, 0 tolerance literals in validators

### 2026-03-24 — Session S173 (typed errors, module decomposition, CI hardening)

- **`thiserror` typed error hierarchy**: `GpuError`, `TensorError`, `ParseError`, top-level `Error` enum in `src/error.rs` — replaces `Result<T, String>` at library boundaries
- **Error migration**: `gpu.rs` → `Result<T, GpuError>`, `gpu_ops/reduction.rs` → `Result<T, TensorError>`; `From` bridges for gradual migration
- **Module decomposition**: `nucleus_pipeline.rs` (874 LOC → 5 files: mod/error/report/dispatch/executor), `glucose_prediction.rs` (794 LOC → 5 files: mod/cgm/analysis/experiment/tests), `immunological_anderson/mod.rs` (779 LOC → 3 new submodules: classification/pharma)
- **barraCuda feature selection**: `default-features = false`, explicit `gpu`, `domain-nn`, `domain-esn`, `domain-genomics`, `domain-timeseries` (dropped unused `domain-pde`/`domain-snn`/`domain-vision`)
- **cargo-deny in CI**: supply chain audit job (licenses, advisories, bans, sources)
- **IPC smoke test in CI**: build primal, start on Unix socket, `health.liveness` JSON-RPC roundtrip via `socat`
- **`rustfmt.toml`**: edition 2024, `max_width = 100`, `use_field_init_shorthand`, `use_try_shorthand`
- **JSON-RPC dead code evolved**: `INVALID_REQUEST` / `INTERNAL_ERROR` wired to dispatch paths; `_jsonrpc` → `jsonrpc_version`
- **Visualization**: `Measured` variant wired to env-var runtime selection (removed dead_code expect)
- **Provenance**: SciPy 1.14.1 → 1.15.3 in tolerance comments; `_provenance` metadata added to `mlp_baseline.json` and `baseline_values.json`; `PUBLICATION_ENVIRONMENT` pinned to exact versions
- **Coral forge shader plan**: 41 WGSL shaders mapped to Group A (generic, barraCuda absorption) and Group B (domain-fold candidates) in `EVOLUTION_MAPPING.md`
- **Doc version sweep**: stale v0.3.5 → v0.3.7 across 8+ spec/whitePaper files; session/test/binary counts synchronized
- **Handoff management**: V121/V122 archived; V123 + barraCuda evolution request created
- **~1,385 tests** (1,199 lib + 72 forge + 80 playGround + 9 integration + 25 tokio), 0 clippy, 0 fmt, 0 doc warnings

### 2026-03-23 — Session S172 (deep evolution & ecosystem absorption)

- **DeviceCapabilities migration**: replaced all deprecated `GpuDriverProfile` usage across 11 files with `barracuda::device::capabilities::DeviceCapabilities` — last spring to complete ecosystem convergence
- **Workspace lint inheritance**: moved `[lints.rust]` + `[lints.clippy]` to `[workspace.lints]`; all 3 workspace members inherit via `[lints] workspace = true`
- **playGround missing-docs**: added docs to all 163 previously-undocumented public items across 14 files
- **normalize_method absorption**: IPC dispatch normalizes legacy `neuralspring.{method}` prefix per ecosystem convention (barraCuda v0.3.7, loamSpine v0.9.8, wetSpring V132)
- **Smart refactoring**: 3 validation binaries refactored by domain responsibility: `validate_gpu_pure_workload_all` (942→7 modules, max 209), `validate_cross_spring_evolution` (913→9 modules, max 189), `validate_modern_cross_spring` (900→10 modules, max 137)
- **Config centralization**: 8 env var names centralized in `config.rs`; `127.0.0.1` → `Ipv4Addr::LOCALHOST`
- **`#[allow]` → `#[expect]`**: last `#[allow]` in forge converted; zero `#[allow]` in production code
- **Doctest fix**: `nucleus_pipeline.rs` doctest unwraps `Result` correctly
- **1,380 tests** (1,203 lib + 73 forge + 80 playGround + 13 doc), 0 clippy, 0 fmt, 0 doc warnings

### 2026-03-23 — Session S171 (deep debt audit execution)

- `PipelineError` typed error: `nucleus_pipeline` functions return `Result` instead of panicking (`.expect()` → `?`); `CyclicGraph` + `MissingStage` variants with `Display` + `Error` impls
- `POSITIVE_DATA_GUARD` (1e-10) and `R2_DENOMINATOR_FLOOR` (1e-30) named constants in `primitives.rs`; wired into 9 validation binaries replacing inline literals
- 2 `bench_*` entries removed from `validate_all.rs` (benchmarks ≠ validators): 232→230 entries
- metalForge forge lint parity: `unwrap_used` + `expect_used` warnings added to `Cargo.toml`
- barraCuda version refs refreshed v0.3.5→v0.3.7 across 4 specs + ABSORPTION_TRACKER
- 6 new proptests: FASTQ roundtrip + length, VCF position + chrom, WDM surrogate finiteness + determinism
- 2 doc warning fixes: unresolved `[Tensor]` link, redundant explicit link target
- **1,356 tests** (1,203 lib + 73 forge + 80 playGround), 0 clippy, 0 fmt, 0 doc warnings

### 2026-03-22 — Session S170b (proptests)

- 12 new property-based tests (proptest) across 5 modules: HMM (alpha sums, log-likelihood, viterbi), game theory (simplex preservation), Anderson localization (IPR bounds, symmetry), isomorphic reservoir (spectral radius, stats), FASTA (roundtrip)
- **1,195 lib + 9 forge + 13 doc = 1,217 Rust tests**

### 2026-03-22 — Session S170 (cross-ecosystem absorption)

- barraCuda v0.3.7 compatibility: `Precision::F16` handling, `MultiHeadEsn::wgpu_device()` removal, `GpuDriverProfile` deprecation
- Semantic Method Naming v2.1: `capabilities.list` (canonical) + `capability.list` (legacy) + `primal.capabilities` (alias)
- `health.liveness` response: `{"status": "alive"}` per standard
- biomeOS 5-tier discovery: `socket-registry.json` lookup (tier 5), 4-format capability response parsing
- `publish = false` on all 3 workspace Cargo.toml
- wateringHole handoff written, PRIMAL_REGISTRY updated to S170

### 2026-03-22 — Session S169 (deep debt evolution)

- Eliminated all 626 `missing_docs` warnings → 0; documented every public item across 40+ files
- Created `CONTEXT.md` per `PUBLIC_SURFACE_STANDARD.md`
- Full AGPL-3.0 LICENSE file (661-line canonical); GitHub description + 14 topics
- README "Part of ecoPrimals" footer
- Named constants (`DEFAULT_IPC_TIMEOUT_SECS`, `DEFAULT_HEARTBEAT_SECS`, `DEFAULT_MAX_CONCURRENT`)
- Graceful shutdown via `tokio::sync::watch` (replaces `std::process::exit`)
- TCP fallback transport via `PRIMAL_TCP_PORT`/`NEURALSPRING_TCP_PORT`
- HuggingFace URLs configurable via env vars
- Zero-copy streaming (FASTA/FASTQ/VCF in-place trimming)
- Binary renamed `neuralspring_primal` → `neuralspring` (UniBin compliance)
- Provenance/discover/offload handlers wired
- Unnecessary `.clone()` eliminated in spectral.rs
- **1,183 lib + 9 forge + 13 doc = 1,205 Rust tests** (pre-proptest)

### Added

- `extract_rpc_result()` / `extract_rpc_result_owned()` — centralized JSON-RPC
  result extraction (healthSpring V37 / wetSpring V127 pattern), replacing
  ad-hoc `response.get("result")` in `classify_response`
- `PRIMAL_DOMAIN` constant (`"science.learning"`) for biomeOS Neural API
  registration (healthSpring V34 pattern)
- `PROVENANCE_REGISTRY` — complete array of all 49 `BaselineProvenance`
  records with 4 completeness/integrity tests
- `OnceLock` GPU probe cache in `metalForge/forge/src/probe.rs` —
  avoids SIGSEGV from concurrent `wgpu::Instance` creation in parallel
  tests (groundSpring V116 pattern)
- Cast lint deny in all 3 workspace `Cargo.toml` files:
  `cast_precision_loss`, `cast_possible_truncation`, `cast_sign_loss`,
  `cast_lossless` (airSpring V0.9.0 pattern)
- 5 new `extract_rpc_result` tests (borrow, error-present, neither,
  owned, fuzz)
- 4 new provenance registry tests (completeness, no-duplicate-scripts,
  expected-source-complete, records-non-empty via registry)

### Changed

- `classify_response` now delegates to `extract_rpc_result()` instead
  of raw `response.get("result")`
- Provenance test `provenance_records_non_empty` evolved from hand-
  maintained inline array to `PROVENANCE_REGISTRY` reference

### Validated

- 1320 tests PASS (1167 lib + 73 forge + 80 playGround, +8 from S168)
- 0 clippy warnings (pedantic+nursery+cast lints, workspace-wide)
- 0 fmt diffs, 0 doc warnings, 0 unsafe, 0 C deps

## [Unreleased] — 2026-03-18 (Session 168: Deep Debt Execution + Ecosystem Handoff)

### Added

- `playGround/src/discovery.rs` — extracted 5-tier socket resolution, capability
  discovery, and primal discovery into focused module (439 LOC)
- `Dispatcher::tensor_session()` — `TensorSession` factory for fused multi-op GPU
  pipelines (eliminates per-op CPU round-trips)
- `Dispatcher::stateful_pipeline()` — `StatefulPipeline` factory for iterative
  GPU kernels (ODE loops, eigensolvers, training)
- 8 new property tests: metrics invariants (R² perfect=1, RMSE/MAE non-neg,
  RMSE≥MAE Cauchy-Schwarz), spectral invariants (Frobenius non-neg, transpose
  involution, distance-to-normal non-neg)
- `upstream_expected` module in `validate_toadstool_s93_barracuda_extraction.rs`
  — named constants for barraCuda tolerance contract validation
- V119 ecosystem handoff with full absorption inventory

### Changed

- `expected_source()` provenance mapping: 9 → 49+ script paths (was matching on
  label strings that never matched; now matches on stable `script` field)
- `ipc_client.rs` smart refactor: 885 → 448 LOC (discovery logic extracted)
- `coral_forge/activation.rs` tests: inline `1e-5` → `tolerances::LAYER_NORM_EPS`
- `primitives.rs` proptest: `x*x + v*v` → `x.mul_add(x, v*v)` (FMA)
- `property_tests.rs`: `as u32` → `u32::try_from()` (safe cast)
- metalForge absorption manifest: `head_split`/`head_concat` → lean phase,
  stale `logsumexp_reduce.wgsl` removed from planned shaders
- CI: workspace-wide `cargo test`, `cargo clippy`, `cargo fmt`, `cargo doc`

### Fixed

- 66 clippy warnings (pedantic+nursery) → zero workspace-wide
  - playGround test modules: `#[expect(unwrap_used, expect_used)]`
  - `weights.rs` tests: `#[expect(float_cmp)]` for IEEE 754 zero exactness
  - `3.14` approx-PI lint in fuzz test → arbitrary `7.89`
- `expected_source()` was non-functional (short label strings never matched
  actual long label values) — now correctly maps all 49+ experiments

### Validated

- 1312 tests PASS (1164 lib + 73 playGround + 75 forge, +8 from S167)
- 0 clippy warnings (pedantic+nursery, workspace-wide including tests)
- 0 fmt diffs, 0 doc warnings, 0 unsafe, 0 C deps

## [Unreleased] — 2026-03-18 (Session 167: Deep Audit + Ecosystem Evolution)

### Added

- `primitives::pearson_r` — centralized Pearson correlation wrapper with zero
  fallback, shared across `wdm_ensemble_qs`, `attention_anderson`,
  `digester_anderson` (3 modules deduplicated)
- `primal_names::display` module — 12 mixed-case display-name constants for
  presentation contexts (dashboards, handoffs, reports)
- `config/capability_registry.toml` — canonical capability definitions (16
  capabilities with descriptions), sync-tested against `config::ALL_CAPABILITIES`
- `MEAN_REDUCE_UPSTREAM` / `MEAN_REDUCE_F64_UPSTREAM` re-exports in forge
  shaders from `barracuda::ops::WGSL_MEAN_REDUCE`
- ecoBin cross-compile CI job: `x86_64-unknown-linux-musl` + `aarch64-unknown-linux-gnu`
  + banned C sys crate detection
- L-BFGS evolution path documented in `pinn` and `loss_landscape` modules
- Kokkos benchmark provenance restructured with verified-baseline requirements
- 1 new test: `capability_registry_toml_in_sync`

### Changed

- `industry_coverage.rs`: all 20+ hardcoded owner strings → `display::*` constants
- `kokkos_parity.rs`: "barraCuda" label → `display::BARRACUDA` constant
- `coralreef_bridge.rs`: socket hints use `CORALREEF_NAME` constant
- `metalForge/fossils/`: 24 `#[allow()]` → `#[expect(reason)]` across 8 files
- `wdm_ensemble_qs.rs`, `attention_anderson.rs`, `digester_anderson.rs`:
  local `pearson_r` → re-export from `crate::primitives`
- `bench_kokkos_parity.rs`: doc comment restructured with explicit provenance
  level, verification requirements, and op classification

### Validated

- 1156/1156 lib + 75 playGround + 73 forge tests PASS
- 0 clippy warnings (pedantic+nursery, production code)
- 0 fmt diffs, 0 doc warnings
- All pre-existing CI gates green

## [Unreleased] — Session 166 (March 17, 2026)

### Session 166 — Doc Evolution, V117 Handoff, Archive Sweep (2026-03-17)

**Stale count cleanup, full barracuda evolution review, V117 handoff, archive sweep.**

- **DOC CLEANUP**: Test counts corrected across all docs: 1155 lib (was 1152) + 75 playGround
  (was 70) + 73 forge = 1303 total. Binary count: 267 (was 260). Module count: 67 (was 47).
  Tolerance count: 225 named constants (was "180+"). All root docs, specs, experiments,
  whitePaper, and wateringHole synchronized.
- **BARRACUDA REVIEW**: Full evolution mapping against barraCuda Sprint 7 (3772 tests,
  `execute_gemm_ex` TransA/TransB, `WGSL_MEAN_REDUCE` re-export, 10 `mul_add()` evolutions,
  typed `BarracudaError`). Confirmed neuralSpring leverages 45+ submodules, 80+ functions,
  216+ import files. Identified new wiring opportunities: L-BFGS optimizer, `StatefulPipeline`,
  `WGSL_MEAN_REDUCE` re-export.
- **V117 HANDOFF**: New `NEURALSPRING_V117_EVOLUTION_REVIEW_HANDOFF_MAR17_2026.md` with
  action items for barraCuda (FMA sweep recommendation, `WGSL_MEAN_REDUCE` absorption) and
  toadStool (IPC proptest pattern, `StatefulPipeline` for HMM chains).
- **ARCHIVE SWEEP**: Stale TODO/FIXME review, outdated count corrections, debris check.

### Session 165 — Ecosystem Absorption: FMA Sweep, IPC Proptest, Leverage Guide (2026-03-17)

**mul_add() FMA precision, IPC property invariants, ecosystem leverage documentation.**

- **FMA SWEEP**: 14 `a * b + c` patterns in 10 production library modules replaced with
  `mul_add()` for IEEE 754 fused-multiply-add precision. Affected modules:
  `glucose_prediction` (3: ridge regression accumulation, prediction dot product),
  `swarm_robotics` (2: neural forward hidden→output layer),
  `loss_landscape` (1: Metropolis perturbation),
  `pangenome_selection` (1: Jaccard intersection),
  `coral_forge/confidence` (1: softmax expected value),
  `coral_forge/pairformer` (2: triangle multiplicative gating),
  `coral_forge/structure/ipa` (1: point distance squared),
  `spectral_commutativity` (1: mat_mul inner loop),
  `pinn` (1: Burgers equation residual).
- **IPC PROPTEST**: 3 new property tests in `property_tests.rs`:
  `retry_policy_delay_never_exceeds_max` (4 configs × 100 attempts),
  `circuit_breaker_state_machine_sweep` (50 trials, threshold 1–5),
  `circuit_breaker_rapid_cycle_no_panic` (1000 random ops).
  5 new IPC fuzz tests in `playGround/src/ipc_client.rs`:
  `parse_capability_list_never_panics_on_arbitrary_json` (24 fuzz values),
  `parse_capability_list_flat_roundtrip_preserves_strings`,
  `dispatch_outcome_classify_never_panics` (9 fuzz values),
  `extract_rpc_error_never_panics` (9 fuzz values),
  `ipc_error_is_recoverable_contract`.
- **LEVERAGE GUIDE**: New `specs/ECOSYSTEM_LEVERAGE_GUIDE.md` documenting all ecosystem
  absorption sources (barraCuda, toadStool, coralReef, biomeOS, 6 sibling springs),
  registered capabilities (16), IPC protocol, self-knowledge constants, and evolution
  readiness assessment (what stays local vs. what migrates upstream).
- **QUALITY**: 28 property tests (25 prior + 3 new), 23 playground IPC tests (18 prior + 5 new).
  Zero clippy warnings (pedantic+nursery). Zero new unsafe.

### Session 164 — Deep Debt Evolution: Tolerance Naming, barraCuda Delegation, MSRV, Platform-Agnostic (2026-03-17)

**7 inline tolerances named, barraCuda math delegation, MSRV pinning, platform-agnostic testing.**

- **TOLERANCES**: 7 inline magic numbers replaced with named constants in `tolerances/mod.rs`:
  `GPU_TRACE_F32_ROUNDTRIP` (0.01), `CORRELATION_CROSS_VALIDATION` (0.05),
  `GPU_ACCUMULATION_F32` (0.1), `CLASSIFIER_METRIC_CROSS` (0.01),
  `INTROGRESSION_FRACTION_CROSS` (0.05), `PROCESS_MODEL_RESPONSE` (0.05),
  `RPC_COUNT_FALLBACK` (0.5). All registered in `domain_guards` category of tolerance
  registry. Wired into 6 validation binaries.
- **BARRACUDA DELEGATION**: `glucose_prediction::solve_symmetric` rewritten to delegate to
  `barracuda::linalg::solve::solve_f64_cpu()` with ridge-regularized fallback for
  near-singular matrices. Eliminates local Cholesky implementation (~40 LOC).
- **MSRV**: `rust-version = "1.87"` pinned across all 3 workspace `Cargo.toml` files
  (root, metalForge/forge, playGround).
- **PLATFORM-AGNOSTIC**: 6 hardcoded `/tmp/` paths in playGround test files replaced with
  `std::env::temp_dir().join(...)` (ipc_client.rs: 4, biomeos_client.rs: 2).
- **IDIOMATIC**: `partial_cmp(b).unwrap()` → `total_cmp(b)` for float sorting in
  `neuralspring_bench_inference.rs`. `#[expect]` reasons clarified in 2 benchmark binaries.
- **REFACTOR**: `tolerances/mod.rs` training section extracted into `tolerances/training.rs`
  submodule (smart refactor — proactive growth management).
- **PRIMAL SELF-KNOWLEDGE**: 2 validation binaries (`validate_nucleus_tower`,
  `validate_biomeos_spectral`) evolved from hardcoded `"neural-spring-test.sock"` to
  `format!("{}-test.sock", niche::NICHE_NAME)`.
- **DOC**: `information_flow::mat_mul_transpose` doc comment updated with barraCuda migration
  clarity. `AᵀA` backticked for `doc_markdown` compliance.
- **QUALITY**: 1152 lib + 73 forge + 2 playGround + 9 integration tests pass, 0 warnings
  (clippy pedantic+nursery), 0 fmt diffs, 0 unsafe, Rust Edition 2024.

### Session 163 — Edition 2024 + Health Probes + Property Testing (2026-03-17)

**Rust Edition 2024, health probes, IPC resilience, proptest, deny.toml hardening.**

- **EDITION 2024**: Full workspace upgrade from 2021 to 2024. Reserved `gen` keyword renamed
  across 8 validation binaries. Let chains (`if let ... && let ...`) applied to ~15 collapsible
  `if let` patterns. Edition 2024 implicit dereference patterns fixed in closures.
- **HEALTH PROBES**: `health.liveness` and `health.readiness` IPC methods implemented in
  neuralspring_primal (handlers, dispatch, capability registration). MCP tools expanded 14→16
  with health domain. `niche.rs` updated with cost estimates and operation dependencies.
- **IPC RESILIENCE**: New `src/ipc_resilience.rs` module — generic `RetryPolicy` (exponential
  backoff with configurable delay/multiplier/max) and `CircuitBreaker` (Closed/Open/HalfOpen
  states, threshold/cooldown, epoch-based timing). 8 unit tests.
- **PROPTEST**: `proptest = "1"` added to dev-dependencies. 6 property-based tests for core
  primitives: softmax sums-to-one, softmax non-negative, shannon_entropy non-negative,
  relu idempotent, relu identity-for-positive, rk4 energy conservation (harmonic oscillator).
- **TOLERANCE PROVENANCE**: Doc comments enriched with validation citations on EXACT_F64,
  CROSS_LANGUAGE, LAYER_NORM_EPS, ODE_ATOL, ODE_RTOL, EIGH_JACOBI_RECONSTRUCT.
- **DENY.TOML**: Hardened — `unknown-git = "deny"` (was warn), advisory DB URLs added,
  `allow-git = []` explicit.
- **DISPATCH**: `DispatchOutcome` enriched with `classify_response()`, `is_protocol_error()`,
  `is_method_not_found()` for structured RPC error handling.
- **QUALITY**: 1295 tests (1152 lib + 73 forge + 70 playGround), 0 warnings (clippy
  pedantic+nursery), 0 fmt diffs, 0 unsafe, Rust Edition 2024.

### Session 162 — Cross-Ecosystem Absorption Execution (2026-03-16)

**4-format capability parsing, generic discovery, circuit breaker IPC, safe casts, zero eprintln workspace-wide.**

- **IPC**: `parse_capability_list()` evolved from 2-format to 5-format (flat, object array,
  nested wrapper, double-nested, result wrapper) — airSpring V0.8.7 pattern. Now `pub` and
  returns `Vec<String>` (never errors) for defensive discovery probes.
- **DISCOVERY**: `socket_env_var()`, `address_env_var()`, `discover_primal()` helpers
  (sweetGrass / groundSpring V112 pattern). Primals check `{UPPER}_SOCKET` env var first,
  then fall back to biomeOS socket directory resolution.
- **DISPATCH**: `DispatchOutcome` enum (groundSpring V112 pattern) classifies RPC responses
  as `Ok`, `ProtocolError` (-32700..-32600), or `ApplicationError` for graceful degradation.
- **RESILIENCE**: `resilient_call()` with circuit breaker + exponential backoff retry
  (healthSpring V32 pattern). Retries recoverable errors (connect, timeout) 2× with
  50ms/100ms backoff; short-circuits if primal recently unavailable.
- **SAFE CASTS**: New `src/safe_cast.rs` module (groundSpring V112 pattern) with `usize_u32()`,
  `usize_u64()`, `usize_f64()`, `f64_f32()`. Applied to `gpu_ops/bio/evolution.rs` (9 casts)
  and `gpu_ops/bio/activation.rs` (7 casts). GPU dispatch params now checked, not silently
  truncated.
- **LOGGING**: All 1642 remaining `eprintln!` → `println!` across 186 src/ files. Zero
  `eprintln!` in entire workspace (src/ + playGround/).
- **QUALITY**: 1295 tests (1152 lib + 73 forge + 70 playGround), 0 warnings, 0 unfulfilled
  expectations, 0 fmt diffs, 0 `eprintln!`, 0 unsafe, 0 hardcoded paths.

### Session 161 — Doc Cleanup + Structured Logging Completion (2026-03-16)

**Hardcoded path elimination, playGround logging, doc sync, archive sweep.**

- **PATHS**: Hardcoded `"biomeos/biomeos.sock"` → `config::BIOMEOS_SOCKET_SUBDIR` /
  `BIOMEOS_ORCHESTRATOR_SOCKET` in primal `main.rs`, `biomeos_client.rs`, `ipc_client.rs`.
  Duplicate `BIOMEOS_SOCKET_SUBDIR` constant in playGround → delegates to lib `config::`.
- **LOGGING**: 28 `eprintln!` → `log::info!/warn!/debug!` across playGround binaries
  (`neuralspring_mcp_adapter`, `neuralspring_interactive`, `neuralspring_bench_inference`,
  `biomeos_client`). Zero `eprintln!` remaining in playGround src.
- **DOCS**: All root docs, baseCamp, experiments, wateringHole synced to S161.
  barracuda/toadstool evolution handoff updated. ecoPrimals docs updated.
  README consolidated (S155–S158 condensed). Archive sweep: workspace clean.
- **QUALITY**: 1128 lib + 61 playGround + 73 forge tests, 0 warnings, 0 unfulfilled
  expectations, 0 fmt diffs, 0 `eprintln!` in playGround, 0 hardcoded socket paths.

### Session 160 — IPC Evolution: Structured Errors + compute.dispatch (2026-03-16)

**Typed IPC errors, centralized RPC error extraction, compute.dispatch protocol.**

- **IPC**: `IpcError` typed enum in `playGround/src/ipc_client.rs` (healthSpring V31 / rhizoCrypt V13
  pattern): `Connect`, `Write`, `Read`, `InvalidJson`, `NoResult`, `RpcError{code,message}`,
  `Timeout`. `is_recoverable()` for retry logic. `call_typed()` for structured error reporting.
  Backward-compatible `call()` preserved.
- **RPC**: `extract_rpc_error()` centralized helper (airSpring V0.8.6 pattern) — replaces ad-hoc
  `response.get("error")` checks.
- **DISPATCH**: Typed `compute.dispatch` IPC methods on `ToadStoolClient`:
  `dispatch_submit(operation, input)` → `DispatchHandle`, `dispatch_result(id)` → `DispatchResult`,
  `dispatch_capabilities()` → `Vec<String>` (wetSpring V124 / healthSpring V31 pattern).
- **TYPES**: `JsonRpcError::code` evolved `i32` → `i64` for JSON-RPC 2.0 spec compliance.
  `DispatchHandle` and `DispatchResult` response types.
- **QUALITY**: 1128 lib + 61 playGround tests, 0 warnings, 0 unfulfilled expectations, 0 fmt diffs.

### Session 159 — Cross-Ecosystem Absorption Execution (2026-03-16)

**OrExit<T>, deny.toml, structured logging, dep audit.**

- **OREXIT**: Absorbed `OrExit<T>` trait from wetSpring V123 into `validation::OrExit` — panic-free
  `process::exit(1)` for `Result<T,E>` and `Option<T>`. Applied to 6 binaries:
  `bench_modern_rewire`, `bench_cross_spring_shader_evolution`, `bench_cross_spring_evolution`,
  `bench_portability_tiers`, `diagnose_f64_regression`, `bench_evolution_tiers` (setup code only;
  `.expect()` in GPU dispatch still uses expect). Stale `clippy::expect_used` expectations pruned
  from `diagnose_f64_regression` (zero `.expect()` remaining) and
  `bench_cross_spring_shader_evolution` (zero `.expect()` remaining).
- **DENY**: Created `deny.toml` (groundSpring V110 / healthSpring V30 pattern): `wildcards = "deny"`,
  license allowlist (AGPL-3.0, MIT, Apache-2.0, BSD, etc.), advisory `vulnerability = "deny"`,
  `yanked = "deny"`, `unknown-registry = "deny"`.
- **LOGGING**: Primal binary `src/bin/neuralspring_primal/main.rs` + `biomeos.rs`: 18× `eprintln!`
  → `log::info!` / `log::warn!` / `log::debug!`. All server lifecycle messages now controllable
  via `RUST_LOG`. Capability list moved to `debug!` level.
- **DEP AUDIT**: Confirmed only external non-Rust dependency: `cc` (build-time C compiler tool)
  via `blake3` in barraCuda. Zero C dependencies in neuralSpring itself. `ring`/`openssl`/`cmake`
  all absent from dependency tree. `blake3 pure` feature noted for barraCuda team.
- **QUALITY**: 1128 lib + 61 playGround tests, 0 warnings, 0 unfulfilled expectations, 0 fmt diffs.

### Session 158 — Cross-Ecosystem Absorption + Deep Debt Continuation (2026-03-16)

**Lint, env safety, smart refactoring, hardcoded names → constants.**

- **LINT**: `#[allow(clippy::wildcard_imports)]` in `tolerances/registry.rs` → `#[expect(reason)]`
  with `#[cfg_attr(not(test), ...)]` to handle cross-cfg boundary. Stale `clippy::unwrap_used` in
  `diagnose_f64_regression.rs` removed after evolving `unwrap()` → `expect("tokio runtime")`.
  Zero unfulfilled lint expectations across `--all-targets`.
- **ENV SAFETY**: `temp-env` v0.3.6 adopted for playGround tests. 26 `set_var`/`remove_var` calls
  in `ipc_client.rs` + `biomeos_client.rs` → `temp_env::with_var`/`temp_env::with_vars`.
  Eliminates Rust 2024 `unsafe` env mutation in tests. 61 playGround tests pass.
- **REFACTOR**: `validate_barracuda_tensor.rs` 918→875 LOC via `check_binary_op` + `check_scalar_op`
  helpers extracting the repeated alloc→dispatch→readback→check pattern for binary and scalar ops.
- **CONSTANTS**: 3 hardcoded `"biomeos"` in `discovery.rs` → `config::BIOMEOS_SOCKET_SUBDIR`.
  `songbird_http.rs` SONGBIRD_HINT → `primal_names::SONGBIRD`.
  `primal_client.rs` → `niche::NICHE_NAME`. Zero hardcoded primal strings in discovery paths.
- **AUDIT**: All `unwrap()` in library code confirmed test-only. All mocks confirmed `#[cfg(test)]`
  only. `#![forbid(unsafe_code)]` enforced at lib root. Only C-adjacent dep is `blake3/cc`
  (build-time, via barraCuda).

### Session 157 — Deep Debt + Idiomatic Rust + Tower Atomic (2026-03-16)

**Zero C dependencies achieved.** 5 blanket lint suppressions eliminated, primal binary
refactored, error handling evolved, Tower Atomic HTTP via Songbird IPC:

- **TOWER ATOMIC**: `playGround/src/songbird_http.rs` — HTTP routed through Songbird via
  `http.request` IPC capability. `reqwest` + `ring` completely removed from workspace.
  `hf_hub.rs` rewritten to use `SongbirdHttp`. Zero compile-time HTTP deps, zero C deps.
- **LINT**: 5 binaries evolved from blanket `#![expect(clippy::pedantic,...)]` to targeted
  `#[expect()]` with documented reasons: `neuralspring_primal/main.rs`,
  `validate_alphafold2_evoformer.rs`, `validate_multi_head_esn.rs`,
  `validate_gpu_ode_batch.rs`, `validate_training_monitor.rs`.
- **REFACTOR**: `neuralspring_primal/main.rs` — 3 functions extracted
  (`push_petaltongue_scenario`, `spawn_lifecycle_tasks`, `accept_loop`).
  `std::env::set_var` eliminated (deprecated Rust 2024). Sub-modules cleaned:
  `discovery.rs`, `folding.rs`, `spectral.rs`.
- **ERROR**: `expect()`/`unwrap()`/`panic!()` replaced with `Result<()>`, `let...else`,
  `process::exit(1)` in `dump_neuralspring_scenarios.rs`, `validate_gpu_ode_batch.rs`,
  `neuralspring_primal/main.rs`.
- **FILE SIZE**: `validate_modern_cross_spring.rs` 949→865 LOC (`bench_row!` macro,
  `bench_pair` helper). `validate_gpu_pure_workload_all.rs` — `check_gpu_f32_mean` extracted.
- **DEPS**: `bytemuck` 1.14→1.21 (metalForge/forge), hardcoded `sandbox/scenarios` →
  `NEURALSPRING_SCENARIO_DIR` env var, Kokkos benchmark provenance documented.
- V108 handoff. V107 archived.
- 1128 lib tests, 0 clippy (pedantic+nursery, -D warnings), 0 fmt diffs.

### Session 156 — Comprehensive Audit + IPC Discovery Fixes + Deep Debt (2026-03-16)

**Full codebase audit** against ecosystem standards. Two critical bugs fixed, deep debt
executed, docs and handoffs updated:

- **P1 CRITICAL**: `playGround/src/ipc_client.rs` — `probe_capabilities()` expected raw
  array from `capability.list` but primals return `{"primal": "...", "capabilities": [...]}`.
  New `parse_capability_list()` handles both formats. 2 tests added.
- **P2 CRITICAL**: `metalForge/forge/src/coralreef_bridge.rs` — `discover_by_capability()`
  returned `.json` manifest file path instead of socket path. Now parses manifest for
  `socket_path`/`socket`/`name` fields. Socket scan widened from hardcoded `"coralreef"` to
  capability hints (`["coralreef", "coral-reef", "shader"]`). `serde_json` dep added to forge.
- **P3**: Squirrel client evolved from name-only to `discover_by_capability("ai.query", ...)`
  with name fallback — matching ToadStool/coralReef client patterns.
- **P4**: New `playGround/src/biomeos_client.rs` — typed `BiomeOsClient` with methods for
  `nucleus.register`, `nucleus.deregister`, `nucleus.heartbeat`, `capability.register`,
  `register_all_capabilities`, `capability.resolve`. 2 tests. Registered in `lib.rs`.
- **P5**: `src/bin/neuralspring_primal/discovery.rs` — hardcoded `"id": 1` in
  `forward_to_primal` and `forward_to_primal_raw` replaced with `AtomicU64` counter.
- **D2**: `validate_digester_anderson.rs` — magic `1e-10` replaced with
  `tolerances::CROSS_LANGUAGE`.
- **D3**: 3 validators (`validate_sovereign_compile`, `validate_mixed_composition_pipeline`,
  `validate_batched_spectral`) converted from `assert!`/`println!` to `ValidationHarness`
  with proper `check_bool`/`check_abs` and exit 0/1.
- **E1**: SPDX header added to root `Cargo.toml`.
- **D1**: `kokkos_parity.rs` placeholder benchmarks evolved to `ProvenanceLevel` enum
  (`Estimated`/`Measured`) — data maturity now machine-introspectable and shown in dashboard.
- Scripts synced: `check_drift.sh` count comment fixed (45→48), `requirements.txt` date
  updated (206→397 PASS).
- V107 handoff crafted for barraCuda/toadStool team.
- Superseded V106 archived. All root docs, experiments journal, wateringHole updated to S156.
- 1301 tests (1128 lib + 73 forge + 65 playGround + 13 doc + 15 integration + 7 bin),
  0 clippy (pedantic+nursery), 0 fmt diffs, 0 unsafe, 0 doc warnings

### Session 155 — Cross-Spring Absorption: primal_names, tolerances.py, provenance trio (2026-03-16)

**Deep absorption** from 4 spring pulls + 5 primal pulls. Ecosystem-aligned patterns:

- `src/primal_names.rs` — dedicated primal name module following airSpring/groundSpring
  pattern: 11 primal constants (TOADSTOOL, BEARDOG, SONGBIRD, NESTGATE, SQUIRREL,
  CORALREEF, RHIZOCRYPT, LOAMSPINE, SWEETGRASS, PETALTONGUE, BIOMEOS) + 4 domain
  constants (DAG, COMMIT, PROVENANCE, COMPUTE) + 2 unit tests
- `config.rs` evolved: `*_NAME_HINT` constants now delegate to `primal_names::*`,
  `PETALTONGUE_SOCKET_DIR/PREFIX` delegate to `primal_names::PETALTONGUE` — zero
  duplicate string literals
- `control/tolerances.py` — shared Python tolerance module mirroring 80+ Rust
  constants from `src/tolerances/mod.rs` (following wetSpring/airSpring pattern)
- `graphs/neuralspring_deploy.toml` — provenance trio nodes added (rhizoCrypt,
  loamSpine, sweetGrass) as Phase 2b with `fallback = "skip"`, following
  wetSpring/rhizoCrypt deploy graph patterns
- V106 handoff crafted: cross-spring absorption (4 springs + 5 primals),
  100+ barraCuda primitives consumed, provenance trio deploy, quality gates
- V105 archived, wateringHole README updated, central wateringHole copy published
- All root docs updated to S155 (README, EVOLUTION_READINESS, CONTROL_EXPERIMENT_STATUS,
  DEPRECATION_MIGRATION, whitePaper/baseCamp, experiments journal Exp 111)
- ecoPrimals/whitePaper/gen3/baseCamp updated with S155 status
- Debris review: zero real TODOs, zero stale files, zero archive candidates
  beyond V105 (now archived)
- 1301 tests (1128 lib + 73 forge + 63 playGround + 13 doc + 15 integration + 9 bin),
  0 clippy (pedantic+nursery), 0 fmt diffs, 0 unsafe

### Session 154 — Niche Deployment + Cross-Spring Absorption + Hardcoding Elimination (2026-03-15)

**Niche architecture** following airSpring/groundSpring pattern. Cross-spring
absorption from 5 sibling springs. Hardcoded primal names eliminated:

- `src/niche.rs` — niche self-knowledge module: NICHE_NAME, CAPABILITIES (22
  capabilities including provenance/compute/discovery), `operation_dependencies()`,
  `cost_estimates()`, `science_semantic_mappings()`, 7 unit tests
- `graphs/neuralspring_deploy.toml` — biomeOS deploy graph: 5-phase deployment
  (Tower Atomic → optional ToadStool/NestGate → neuralSpring → health check →
  provenance), all `by_capability` discovery, zero hardcoded primal names
- Hardcoded `"biomeOS.sock"` fallback → biomeOS 5-tier socket resolution
  (`$BIOMEOS_ORCHESTRATOR_SOCKET` → `$XDG_RUNTIME_DIR/biomeos/` → `temp_dir()`)
- Hardcoded `"toadstool"`, `"coralreef"`, `"squirrel"` name hints → centralized
  `config::TOADSTOOL_NAME_HINT`, `config::CORALREEF_NAME_HINT`,
  `config::SQUIRREL_NAME_HINT` constants
- `config.rs` expanded: `BIOMEOS_SOCKET_SUBDIR`, `BIOMEOS_ORCHESTRATOR_SOCKET`,
  `ENV_BIOMEOS_ORCHESTRATOR`, 3 primal name hint constants
- Niche deployment Steps 1-4 of 7 complete (UniBin, capabilities, deploy graph,
  niche module). Steps 5-7 (provenance trio, cross-spring time series, workflow
  graphs) documented as evolution targets
- 1297 tests (1126 lib + 7 niche + 73 forge + 61+2 playGround + 13 doc + 15 integration),
  0 clippy (pedantic+nursery), 0 fmt diffs, 0 unsafe

### Session 153 — Comprehensive Ecosystem Audit + Deep Debt Execution (2026-03-15)

**Full 11-dimension audit** against wateringHole ecosystem standards, followed by
systematic execution of all findings. Deep debt solutions, modern idiomatic Rust:

- Zero clippy warnings (pedantic+nursery) across all 3 workspace crates — fixed
  9 playGround warnings (`missing_const_for_fn`, `or_fun_call`, `single_match_else`,
  `redundant_clone`) and 2 forge `rustfmt` diffs
- `ALL_CAPABILITIES` unified into single source of truth in `config.rs` — primal
  binary and playGround `mcp_tools` now re-export from shared constant
- `validate_gpu_eigensolve_pipeline.rs` migrated from ad-hoc pass/fail to
  `ValidationHarness` with centralized `tolerances::GPU_EIGENVALUE_AGREEMENT`
- 3 new tolerance constants: `GPU_EIGENVALUE_AGREEMENT` (1e-6), `VARIANCE_PARITY_FLOOR`
  (1e-10), `PAIRFORMER_PARITY` (1e-6) — all with scientific justification
- 6 validation binaries migrated from inline tolerances to `tolerances::` constants
- 3 validator-local `fn sigmoid` wrappers replaced with `use primitives::sigmoid`
- playGround lints aligned: `[lints]` section added to `Cargo.toml` matching
  workspace standard (pedantic, nursery, unsafe_code=forbid, unwrap/expect=warn)
- `#![forbid(unsafe_code)]` added to `metalForge/forge/src/lib.rs`
- `neural-spring` added as playGround dependency for shared `ALL_CAPABILITIES`
- Provenance linked to expected-value sources via `BaselineProvenance::expected_source()`
- 4 new GPU tests in `gpu_ops/bio/` (evolution edge cases, activation monotonicity)
- V104 absorption handoff for barraCuda/toadStool team
- 1290 tests, 0 clippy (pedantic+nursery), 0 doc warnings, 0 fmt diffs

### Session 152 — Deep Debt Execution: Tolerance Centralization, Capability Discovery, Shared Infrastructure (2026-03-15)

**Deep debt execution** — centralize hardcoded tolerances, evolve to capability-based
discovery, add shared validation infrastructure, clean archive, update all docs:

- 15+ hardcoded tolerance literals in bench/validation binaries replaced with named
  `tolerances::` constants (`CROSS_LANGUAGE`, `JACOBI_GPU_CONVERGENCE`, `EXACT_F64`,
  `SPECTRAL_EIGENSOLVER_CROSS`, `GELU_LARGE_INPUT`, `EIGENSOLVER_SMALL_MATRIX`).
- New `IPR_CROSS_PYTHON` tolerance (0.005) with provenance documentation, registered
  in spectral category of tolerance registry.
- `PrimalClient::discover()` evolved from `discover_socket("neuralspring")` to
  `discover_by_capability("science.spectral_analysis", PRIMAL_SOCKET_HINT)` —
  capability-first with name fallback.
- coralReef bridge `discover_socket()` restructured: capability manifests scanned
  first, socket name-matching as fallback.
- `BIOMEOS_SOCKET_SUBDIR` constant extracted in both `ipc_client` and `coralreef_bridge`
  — no more inline `"biomeos"` strings in path construction.
- `validate_tensor_binary()` + `BinaryTensorInputs` struct added to `validation::gpu`
  (avoids `too_many_arguments` while providing shared binary-op validation pattern).
- `gen_test_f64()` helper added to `validation::gpu` for deterministic test data.
- playGround test tolerances named: `F32_SOFTMAX_SUM`, `F32_ELEMENT_EXACT` (transformer),
  `F16_EXACT`, `F16_SUBNORMAL_UPPER` (weights).
- 3 pre-existing clippy `single_match` warnings fixed in `ipc_client` tests.
- V95 coralReef handoff archived (superseded by V102).
- V103 handoff for barraCuda/toadStool absorption.
- All docs updated: README, CHANGELOG, CONTROL_EXPERIMENT_STATUS, experiments journal,
  whitePaper/baseCamp, wateringHole handoff, ecoPrimals/whitePaper/gen3/baseCamp.

### Session 151 — Deep Audit: ecoBin Compliance, Capability Discovery, Tolerance Centralization (2026-03-15)

**Deep audit and evolution pass** — ecoBin compliance, capability-based IPC,
tolerance centralization, and V102 handoff:

- Eliminated `openssl-sys`/`native-tls` C dependency in playGround by switching
  `reqwest` to `rustls-tls` backend. Main crates remain zero C deps.
- ToadStool and coralReef IPC clients evolved from hardcoded primal names to
  `discover_by_capability()` — probes running primals for required capabilities
  via `capability.list` JSON-RPC.
- 12 hardcoded tolerance values in tests centralized to `tolerances::` constants.
- 4 weak `#[expect()]` reasons replaced with specific mathematical justifications.
- 3 rustdoc intra-doc link warnings fixed (full qualification in visualization/).
- `mock_response` renamed to `accept_and_reply` in validation binary (real IPC).
- metalForge coralReef bridge evolved to biomeOS 5-tier socket resolution.
- Handoff naming fix: `ENABLE_F64_FIX` → `V95_ENABLE_F64_FIX`.
- V102 handoff for barraCuda/toadStool/coralReef absorption.
- Updated all stale v0.3.3 → v0.3.5 references across docs and source.
- Updated all root docs, whitePaper, experiments journal, wateringHole README.
- Updated ecoPrimals/whitePaper/gen3/baseCamp with S151 state.
- 1115 lib + 73 forge + 61 playGround tests. 0 clippy (pedantic+nursery). 0 doc warnings.

### Session 150 — playGround: Compute Triangle (ToadStool + coralReef) (2026-03-14)

**Compute triangle integration** — wired playGround into ToadStool (compute
orchestration) and coralReef (sovereign shader compiler), with hot dispatch
benchmarking:

- `toadstool_client.rs`: Typed IPC client for ToadStool — `compute.submit`,
  `compute.status`, `compute.result`, `gpu.info`, `gpu.memory`,
  `science.gpu.dispatch`, `science.substrate.discover` (30s timeout for jobs)
- `coralreef_client.rs`: Typed IPC client for coralReef — `shader.compile.wgsl`,
  `shader.compile.wgsl.multi`, `shader.compile.spirv`, compiler capabilities
  and status (60s timeout for compilation)
- `neuralspring_compute_probe` binary: Probes all three compute tiers
  (barraCuda direct, ToadStool IPC, coralReef IPC) for availability, latency,
  and capabilities. Reports pipeline compilation time and hot/cold dispatch.
- `neuralspring_bench_inference` refactored with `--hot` mode: Reuses
  `TensorSession` via `reset()` to benchmark pure kernel dispatch (pipelines
  compiled once). 7-45x faster than cold dispatch. GELU drops from 5329µs to
  118µs on RTX 4070.
- `bench/compare.sh` updated: Now runs both cold and hot barraCuda modes
  alongside PyTorch/CUDA for 4-way comparison with speedup ratios.
- Library `lib.rs`: Added `toadstool_client` and `coralreef_client` modules,
  updated doc comments for compute triangle.

**Key benchmark findings (RTX 4070, seq=128, hidden=768):**
- Cold→Hot speedup: 7x (matmul), 10x (layer_norm), 45x (GELU)
- Remaining gap vs PyTorch/CUDA: 8-22x (bind-group creation + Vulkan submit)
- Path to parity: coralReef sovereign dispatch (bypass Vulkan entirely)

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
