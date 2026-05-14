# neuralSpring V158 — Deep Debt Re-Audit + Evolution Sprint

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Session:** S205b (May 13, 2026)
**From:** neuralSpring (syntheticChemistry)
**To:** primalSpring (L2 coordination)
**Prior:** V157 (S205 niche convergence), V156 (S204b deep debt), V155 (S203 Tier 2)

---

## Directive Compliance

Re-audit of all 7 priorities after S205 code additions (NestGate weight
persistence, Squirrel facade completion, `IpcError::Other`, `base64`).

---

## Priority 1: Deep Debt (TODO/FIXME/HACK)

| Item | Count |
|------|-------|
| `TODO` in `src/` | **0** |
| `FIXME` in `src/` | **0** |
| `HACK` in `src/` | **0** |
| `todo!()` in `src/` | **0** |
| `unimplemented!()` in `src/` | **0** |

## Priority 2: Modern Idiomatic Rust

| Item | Status |
|------|--------|
| Edition | **2024** |
| MSRV | **1.87** |
| Clippy pedantic+nursery warnings | **0** |
| Unfulfilled `#[expect]` | **0** |
| `#[allow()]` in active code | **0** |
| `#[expect()]` instances | ~130 (all legitimate — `cast_precision_loss`, test helpers) |

## Priority 3: External Dependencies

| Item | Status |
|------|--------|
| Direct C/FFI crates in `Cargo.toml` | **0** |
| `openssl-sys`, `ring`, `aws-lc-sys` | **absent** |
| Indirect C linkage | `wgpu` → system Vulkan/Mesa, `blake3` → optional ASM (via barraCuda, optional) |
| Pure Rust deps | `safetensors`, `serde`, `base64`, `tokio`, `log`, `thiserror`, `clap`, `toml`, `pollster` |
| Workspace dependency inheritance | **100%** (all deps via `[workspace.dependencies]`) |

## Priority 4: Large Files (>800 LOC)

| Rank | File | Lines | Status |
|------|------|-------|--------|
| 1 | `tolerances/mod.rs` | 776 | Under limit |
| 2 | `validate_petaltongue_scenarios.rs` | 749 | Under limit |
| 3 | `property_tests.rs` | 722 | Under limit |
| 4 | `validate_nucleus_tower.rs` | 718 | Under limit |
| 5 | `ipc/mod.rs` | 710 | Under limit |

**Zero files exceed 800 LOC.**

## Priority 5: Unsafe Code

| Item | Status |
|------|--------|
| `unsafe {}` blocks | **0** |
| `unsafe fn` | **0** |
| `#![forbid(unsafe_code)]` | **Workspace-wide** (`[workspace.lints.rust]`) |

## Priority 6: Hardcoding

| Item | Status |
|------|--------|
| Hardcoded primal names in discovery | **0** — all via `primal_names::*` constants |
| Hardcoded socket paths | **0** — all via `discover_primal_socket()` / `CapabilityRouter` |
| Hardcoded capability strings | **0** — all via `capabilities::*` constants |
| `CAPABILITY_HINTS` entries | **20** (mapping 37 capabilities to 7 primals) |

## Priority 7: Mocks

| Item | Status |
|------|--------|
| Production mock functions | **0** |
| `mock` references in `src/` | 7 files — all inside `#[cfg(test)]` or doc comments |
| `panic!` in production code | **0** — all 10 instances confined to `#[cfg(test)]` |

---

## Audit Question Answers

### 1. Python baselines for barraCuda CPU (Rust) parity?

**Yes — comprehensive coverage.**

- **20 benchmark scripts** (`control/*/bench_*.py`) across 15 domains
- **84 total Python scripts** in `control/`
- **397/397 assertions** pass against Rust implementations
- **38.6× geometric mean** CPU speedup (Python/NumPy single-thread → Rust)
- **Domains covered**: MLP, transformer, Anderson localization, counterdiabatic,
  directed evolution, eco-dynamics, game theory, glucose LSTM, HMM phylo,
  meta-population, pangenome selection, regulatory network (RK4), signal
  integration, spectral commutativity, swarm robotics

**Operations lacking baselines**: None at the CPU level. GPU-specific operations
(WGSL shader dispatch, tensor lifecycle) are validated through Rust integration
tests rather than Python→GPU comparison scripts.

### 2. Industry-standard benchmarks for barraCuda GPU parity?

**Partial coverage.**

| Benchmark Suite | Status | Notes |
|-----------------|--------|-------|
| Python/NumPy vs Rust CPU | **Complete** | 15 domains, 38.6× geomean |
| Kokkos | **Partial** | `bench_kokkos_parity` with estimated baselines (not matched-hardware). barraCuda shipped LAMMPS+SciPy+Kokkos bench scaffolds upstream |
| cuBLAS (GEMM) | **Partial** | `bench_cublas_gemm.py` exists, roofline-model estimates |
| cuFFT | **Partial** | `bench_cufft.py` exists, estimated baselines |
| cuDNN | **Partial** | `bench_cudnn_ops.py` exists, estimated baselines |
| Flash Attention | **Partial** | `bench_flash_attention.py` exists, estimated baselines |
| Galaxy/NAMD/GROMACS | **Not present** | These are HPC application benchmarks, not kernel benchmarks |
| SciPy | **Partial** | barraCuda upstream shipped SciPy parity scaffolds |
| LAMMPS | **Partial** | barraCuda upstream shipped LAMMPS parity scaffolds |

**Key gap**: Industry benchmarks use estimated baselines from published papers,
not matched-hardware runs. Matched-hardware validation is hotSpring's niche
(biomeGate + strandGate compute teams).

### 3. What have we NOT implemented, verified, validated, or tested?

| Item | Status | Owner/Dependency |
|------|--------|------------------|
| Squirrel provider registration (`inference.register_provider`) | Not implemented | Squirrel upstream |
| WGSL tokenization pipeline (coralReef → toadStool → barraCuda) | Not implemented | Cross-primal composition |
| coralReef `shader.compile.wgsl` live validation | Blocked | coralReef deployment |
| Full NUCLEUS composition end-to-end | Holding | Per directive — await Tower+Node+Nest atomic live |
| Matched-hardware GPU benchmarks | Not our niche | hotSpring (biomeGate/strandGate) |
| Ionic bridge pattern | Blocked upstream | healthSpring owns |

### 4. Papers remaining unreviewed from queue?

**0 unreviewed** — all 27 papers in the queue are reproduced:
- Dolson (5 papers): counterdiabatic, directed evolution, eco-dynamics, LTEE
- Liu (3 papers): HMM, phylogenetics
- Anderson/Kachkovskiy (7 papers): localization, spectral
- Waters (5 papers): quorum sensing, ecology
- Barrick (3 papers): LTEE mutation accumulation (B1 DONE)
- Industry (4 papers): game theory, swarm, glucose, pangenome

**Queue for future**: MDA Framework (Hunicke 2004) and Bartle player types
are ludoSpring's domain. LTEE B6-B9 papers for groundSpring when bandwidth.

### 5. Datasets to examine?

| Dataset | Domain | Priority | Notes |
|---------|--------|----------|-------|
| **NOAA GHCND** | Climate/environmental | High | Public CSV, good first NestGate pipeline exercise (per groundSpring directive) |
| **LTEE frozen fossil archive** | Evolutionary biology | High | Barrick Lab UT Austin, already reproduced B1 |
| **UniProt/AlphaFold DB** | Protein structure | Medium | For coral_forge structure predictions |
| **SILVA rRNA database** | Metagenomics | Medium | For wetSpring-adjacent 16S/ITS pipelines |
| **PhyloNet/TreeBASE** | Phylogenetics | Medium | For HMM/phylo validation against real trees |
| **ImageNet/CIFAR** | ML benchmarks | Low | Standard but not our scientific niche |

---

## Quality Gates

| Metric | Value |
|--------|-------|
| Workspace tests (IPC-first) | **910** |
| Workspace tests (barracuda-enabled) | **~1,453** |
| Clippy warnings (new) | **0** |
| `unsafe` blocks | **0** |
| TODO/FIXME/HACK | **0** |
| Files >800 LOC | **0** |
| Production mocks | **0** |
| `#[allow()]` attributes | **0** |
| Unfulfilled `#[expect]` | **0** |
| Python baselines | **397/397** |
| Papers reproduced | **27/27** |
| Capability constants | **37** |
| IPC modules | **7** |
| CAPABILITY_HINTS | **20** |
| Python bench scripts | **20** (15 domains) |

---

## Summary

neuralSpring is at **zero actionable deep debt** across all 7 priority
categories. The codebase has been through 4 deep debt audits (S199, S202c,
S204b, S205b) with consistent zero-debt results. All S205 code additions
(NestGate weight persistence, Squirrel facade, `IpcError::Other`, `base64`)
maintain the zero-debt standard.

Remaining evolution items are all dependency-blocked:
- Squirrel provider registration (upstream Squirrel)
- WGSL tokenization pipeline (coralReef live deployment)
- Full NUCLEUS composition (per directive: hold until atomics prove live)
