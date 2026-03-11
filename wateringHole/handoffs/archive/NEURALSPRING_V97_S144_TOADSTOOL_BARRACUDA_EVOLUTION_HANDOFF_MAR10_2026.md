<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V97 → toadStool/barraCuda Evolution Handoff

| Field | Value |
|-------|-------|
| **Date** | 2026-03-10 |
| **From** | neuralSpring S144 (1112 lib + 73 forge + 9 integration tests, 254 binaries, 0 clippy) |
| **To** | barraCuda team, toadStool team, coralReef team |
| **Supersedes** | V96 (S143 — Axis 2 novel compositions) |
| **Synced against** | barraCuda `83aa08a`, toadStool S142 (`a86bc546`), coralReef Iteration 29 (`2779c88`) |
| **License** | AGPL-3.0-or-later |

---

## Executive Summary

neuralSpring S144 adds **composition visualization** and **NUCLEUS pipeline execution**
on top of the S143 Axis 2 novel compositions. The 5 new experiments are now
fully visualizable via petalTongue, and the execution pipeline is wired through
the biomeOS NUCLEUS Tower→Node→Nest dispatch pattern.

This handoff documents:

1. New petalTongue visualization infrastructure for composition experiments
2. The `composition_pipeline()` DAG added to metalForge (ToadStool-absorbable)
3. The `nucleus_pipeline` executor and its NUCLEUS dispatch pattern
4. Updated BarraCUDA usage metrics (219 import files, up from 209)
5. Carried upstream evolution opportunities from V96

---

## Part 1: petalTongue Composition Visualization

### 5 New Scenario Builders

| Builder | Nodes | Channel Types | BarraCUDA Used (transitive) |
|---------|-------|---------------|---------------------------|
| `digester_anderson_study()` | 3 (community, Anderson coupling, ESN accuracy) | TimeSeries, Gauge | `stats::correlation`, `tensor::Tensor` via digester_anderson |
| `isomorphic_reservoir_study()` | 2 (cross-domain spectra, universality metrics) | Spectrum, Bar, Gauge | `ops::linalg::eigh` via isomorphic_reservoir |
| `wdm_ensemble_qs_study()` | 3 (disagreement, Anderson phase, QS dynamics) | TimeSeries, Gauge | `stats`, `tensor` via wdm_ensemble_qs |
| `introgression_nn_study()` | 2 (NN observations, HMM detection) | TimeSeries, Heatmap, Bar, Gauge | `stats` via introgression_nn |
| `attention_anderson_study()` | 2 (quality sweep, spectral localization) | TimeSeries, Spectrum, Gauge | `ops::linalg::eigh` via attention_anderson |

### Composition Combiner

`composition_study()` merges all 5 experiments into a single graph with 4
cross-experiment edges:

- Anderson coupling ↔ Anderson phase (shared Anderson physics)
- Reservoir spectral ↔ attention quality (shared spectral analysis)
- NN observations → reservoir spectral (NN weights → universality)
- Ensemble disagreement ↔ digester community (variance ↔ diversity)

`full_study()` grows from **16 to 21 tracks** with 5 new cross-track edges
connecting composition experiments to their foundation tracks:

- `anderson_sweep` → `digester_community`
- `spectral_analysis` → `reservoir_spectral`
- `wdm_transport` → `ensemble_disagreement`
- `hmm_forward` → `nn_observations`
- `spectral_analysis` → `attention_quality`

### Visualization Modes

| Mode | Command | Tracks |
|------|---------|--------|
| `--compositions` | `./scripts/visualize.sh --compositions` | 5 composition experiments |
| `--ecosystem` | `./scripts/visualize.sh --ecosystem` | All 21 tracks |
| `--render` | `./scripts/visualize.sh --render` | Complete study + petalTongue launch |

---

## Part 2: metalForge Composition Pipeline DAG

### `composition_pipeline()` — 6 Stages, 6 Edges

```text
eigensolve → digester_anderson → wdm_ensemble_qs → introgression_nn
           ↘ isomorphic_reservoir → attention_anderson ↗
```

| Stage | Capability | Substrate |
|-------|-----------|-----------|
| `eigensolve` | `science.eigensolve` | CpuOnly |
| `digester_anderson` | `science.digester_anderson_coupling` | CpuOnly |
| `isomorphic_reservoir` | `science.isomorphic_reservoir` | CpuOnly |
| `wdm_ensemble_qs` | `science.wdm_ensemble_qs` | CpuOnly |
| `introgression_nn` | `science.introgression_nn` | CpuOnly |
| `attention_anderson` | `science.attention_anderson` | GpuOnly |

**For ToadStool absorption**: This pipeline follows the same `PipelineGraph` /
`StageNode` / `PipelineExecution` pattern that ToadStool already absorbed from
neuralSpring S139 (`pipeline_graph`). The composition pipeline adds 6 new
capability-addressed stages that ToadStool can dispatch via its
`pipeline_graph` engine once the capabilities are registered as primal actions.

---

## Part 3: NUCLEUS Pipeline Executor

### Tower→Node→Nest Dispatch

`nucleus_pipeline.rs` implements the full NUCLEUS atomic pattern:

| Phase | What happens |
|-------|-------------|
| **Tower** (capability discovery) | Resolves `stage.capability` → local function via match table |
| **Node** (compute dispatch) | Calls real neuralSpring module (eigh, digester_anderson, etc.) |
| **Nest** (provenance) | Records `StageResult` with substrate, timing (µs), output (Map/Vector) |

### API

```rust
let report = execute_composition_pipeline();
assert!(report.all_passed());
println!("Total: {:.0}µs across {} stages", report.total_us(), report.total_stages);
```

### Validation

| Test | What it checks |
|------|---------------|
| `composition_pipeline_executes_all_stages` | All 6 stages pass |
| `composition_pipeline_respects_topo_order` | eigensolve before all dependents |
| `composition_pipeline_records_timing` | All stages have `elapsed_us > 0` |
| `composition_pipeline_outputs_are_populated` | Map or Vector outputs non-empty |
| `eigensolve_stage_produces_correct_eigenvalues` | Identity matrix → all eigenvalues = 1.0 |
| `unknown_capability_fails` | Graceful failure for unknown caps |
| `digester_anderson_produces_valid_metrics` | IPR ∈ [0,1], Shannon H > 0 |
| `introgression_detects_anomalous_layers` | TPR > 0.5, accuracy > 0.5 |
| `substrate_provenance_is_recorded` | CpuOnly vs GpuOnly per stage |

**For biomeOS absorption**: The executor pattern can evolve from local dispatch
(current) to JSON-RPC dispatch (`capability.resolve` → primal socket → forward)
once biomeOS NUCLEUS is running. The `dispatch_capability()` match table maps
directly to biomeOS capability strings.

---

## Part 4: Updated BarraCUDA Usage Metrics

| Metric | V96 (S143) | V97 (S144) | Delta |
|--------|-----------|-----------|-------|
| Files with barracuda imports (src/) | 209 | 219 | +10 |
| metalForge barracuda files | ~60 | 64 | +4 |
| Lib tests | 1085 | 1112 | +27 |
| Forge tests | 71 | 73 | +2 |
| Modules | 46 | 47 | +1 (`nucleus_pipeline`) |
| Binaries | 245 | 254 | +9 (5 orphans registered) |
| petalTongue scenario tracks | 16 | 21 | +5 |
| metalForge pipelines | 3 | 4 | +1 (`composition_pipeline`) |

### New BarraCUDA Usage Paths

The S144 scenario builders use barracuda **transitively** — they call
neuralSpring modules that themselves use barracuda. No new direct barracuda
imports were added in S144, but the visualization layer exercises more
barracuda code paths:

- `barracuda::stats::correlation::pearson_correlation` — via digester_anderson, wdm_ensemble_qs
- `barracuda::ops::linalg::eigh_householder_qr` — via isomorphic_reservoir, attention_anderson
- `barracuda::tensor::Tensor` matmul/tanh/add — via digester_anderson ESN

---

## Part 5: Carried Upstream Evolution Opportunities (from V96)

### For BarraCUDA

| Opportunity | Priority | Status |
|------------|----------|--------|
| Batched `eigh_householder_qr` (N matrices in one call) | Medium | Open — isomorphic_reservoir runs 3 serial eigh calls |
| `Tensor::eigh` (GPU-resident eigendecomposition) | High | Open — currently CPU-only via `eigh_householder_qr` |
| HMM Viterbi GPU batch | Medium | Open — introgression_nn runs CPU Viterbi on 50–100 layers |

### For ToadStool Compute Dispatch

| Opportunity | Priority | Status |
|------------|----------|--------|
| Composition pipelines via `pipeline_graph` | High | **NEW**: `composition_pipeline()` ready for absorption |
| Cross-domain spectral dispatch (shared eigensolve) | Medium | Open |
| Mixed-hardware IPR (GPU eigensolve → CPU stats) | Low | Open |

### For metalForge

No new shaders needed. Existing substrate model validated across all 5
composition experiments. `composition_pipeline()` stages annotated with
correct `MixedSubstrate` values.

---

## Part 6: Current neuralSpring State

| Metric | Value |
|--------|-------|
| Lib tests | 1112 |
| Forge tests | 73 |
| Integration tests | 9 |
| Modules | 47 |
| Binaries | 254 |
| Papers reproduced | 27/27 |
| Composition experiments | 5 (Exp 097–101) |
| petalTongue scenario tracks | 21 |
| metalForge pipelines | 4 (spectral, popgen, folding, composition) |
| NUCLEUS pipeline executor | 6-stage, 9 tests |
| Clippy | 0 |
| Unsafe | 0 |
| BarraCUDA imports | 219 files |
| metalForge shaders | 42 |
| GPU dispatch ops | 47 |

### What's Next

| Priority | Item |
|----------|------|
| **Axis 1** | Real data into validated pipelines (pretrained model weights, PDB proteins, soil microbiome) |
| **Axis 2** | Remaining compositions blocked on hotSpring/healthSpring data |
| **Axis 3** | Full NUCLEUS orchestration (JSON-RPC dispatch, biomeOS graph execution) |
| **petalTongue** | Live streaming for composition experiments, Songbird integration |
| **ToadStool** | `composition_pipeline()` absorption, cross-domain dispatch evolution |

---

*neuralSpring V97 S144 — ready for ToadStool composition pipeline absorption
and biomeOS NUCLEUS evolution.*
