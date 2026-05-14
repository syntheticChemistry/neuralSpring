<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V156 — Deep Debt Resolution + Evolution Sprint

**Session**: S204 | **Date**: May 13, 2026 | **Handoff**: V156
**Prior**: V155 (S203–S203b, May 13 — Tier 2 convergence + deep debt)

---

## Summary

Comprehensive deep debt audit executed per primalSpring directive. All seven
priority areas audited. Four code fixes applied. Codebase confirmed clean
across all categories. Upstream pull synced all 13 primals.

---

## 1. Deep Debt Audit Results

### 1.1 TODO/FIXME/HACK/todo!()/unimplemented!()

| Metric | Result |
|--------|--------|
| `TODO` comments | **0** |
| `FIXME` comments | **0** |
| `HACK` comments | **0** |
| `todo!()` macros | **0** |
| `unimplemented!()` macros | **0** |

### 1.2 Clippy (pedantic + nursery + cast deny)

| Metric | Result |
|--------|--------|
| Errors | **0** |
| Unfulfilled `#[expect]` | **0** |
| Remaining warnings | Only `cast_precision_loss` (under `#[expect]` in 3 modules — legitimate `usize→f64` grid math) |

**S204 fixes**: `relu` → `const fn`, `mul_add` in LTEE B1 polyfit/interp/residual, `merge_tracks()` helper (DRY + `too_many_lines`), unneeded `return` removed, 23 doc backtick fixes across `primal_names`, `nestgate`, `ipc/mod`.

### 1.3 External Dependencies (ecoBin Compliance)

| Category | Status |
|----------|--------|
| Direct `-sys`/`bindgen`/`cc` crates | **0** in any `Cargo.toml` |
| Handwritten FFI (`extern "C"`) | **0** in workspace `.rs` |
| Indirect C/native via deps | `wgpu` → `wgpu-hal` (Vulkan/Metal FFI, expected), `blake3` → `cc` (SIMD builds) |
| Workspace inheritance | **100%** — `approx`, `proptest`, `serial_test` moved to `[workspace.dependencies]` (S204) |
| `#![allow(deprecated)]` | **Removed** from `playGround/src/lib.rs` (was suppressing nothing) |

**ecoBin verdict**: No C dependencies in *our* code. GPU backend FFI via `wgpu` is ecosystem-standard and unavoidable for shader compilation.

### 1.4 Large Files (>800 LOC)

| Metric | Result |
|--------|--------|
| Files > 800 LOC | **0** |
| Largest file | `tolerances/mod.rs` (776 LOC) |
| Files 700–800 LOC | 4 (`tolerances/mod.rs`, `validate_petaltongue_scenarios.rs`, `fossils/fused_transformer.rs`, `property_tests.rs`) |

### 1.5 Unsafe Code

| Metric | Result |
|--------|--------|
| `unsafe {}` blocks | **0** |
| `unsafe fn` | **0** |
| `#![forbid(unsafe_code)]` | **All 3 crate roots** (neural-spring, playground, forge) |

### 1.6 Hardcoding

| Pattern | Status |
|---------|--------|
| Hardcoded socket paths | **0** — all use `std::env::temp_dir()` or `$XDG_RUNTIME_DIR` |
| Hardcoded primal names | Only in `primal_names.rs` (canonical barrel), `capabilities.rs` (protocol strings), `config.rs` (self-knowledge), `niche.rs` (self-knowledge). Forge `CORALREEF_NAME` documented as intentional (independent workspace member). |
| Hardcoded IPs/ports | **0** — `Ipv4Addr::LOCALHOST` + env-driven ports |
| Discovery | `CapabilityRouter` with 20 runtime hints, `discover_primal_socket` 5-tier cascade |
| Doc drift fixed | `composition.rs` discovery doc aligned with `temp_dir()` implementation (S204) |

### 1.7 Production Mocks

| Metric | Result |
|--------|--------|
| `mock`/`fake`/`stub` in production | **0** |
| `mock`/`fake` in `#[cfg(test)]` | **Correct** — all test-only |
| `panic!` in production | **0** (8 occurrences, all `#[cfg(test)]`) |
| `.unwrap()` in library code | **0** — only in bench/diagnostic binaries |

---

## 2. Audit Answers

### Python baselines for barraCuda CPU (Rust) parity

**15 domains complete** with Python `bench_*.py` scripts:

HMM forward, NK fitness, Pairwise L2, Eco batch, Hamming distance, Jaccard distance, Replicator dynamics, RK4 GRN, Spectral commutator, Anderson IPR, Hill gate, Multi-objective, Swarm NN, Global FST, LSTM Glucose.

**38.6× honest geomean** across all 15 domains. **397/397 Python baseline checks PASS**.

**Gap**: Paper 027 (digestion prediction / Wang ESN) has correctness validators but no CPU bench script in the 15-domain suite. This is a minor gap — the math is validated via other paths.

### Industry benchmarks for barraCuda GPU parity

| Framework | Status | Location |
|-----------|--------|----------|
| cuBLAS/cuDNN/cuFFT/FlashAttention | **Complete** | `bench_industry_gpu_parity` + `control/industry_gpu/bench_*.py` |
| Kokkos | **Partial** — estimated baselines, not matched-hardware | `bench_kokkos_parity` + upstream barraCuda Kokkos bench scaffolds |
| SciPy | **Reference validation** (chi², Welford, cdist) not a perf benchmark | Via barraCuda upstream `scipy_parity.rs` |
| LAMMPS | **barraCuda upstream** — LJ/Yukawa GPU benchmarks | `barraCuda/benches/lammps_parity.rs` |
| Polybench/oneDNN | **Not present** — upstream barraCuda team scope |
| Galaxy | **N/A** — workflow engine, not kernel comparison |

### What has NOT been implemented/verified/validated/tested?

| Area | Status |
|------|--------|
| NestGate weight loader integration | **WIP** — IPC client done, `weight_loader.rs` still filesystem-only |
| BearDog BTSP sessions | **Pending** — probes work, signed sessions not yet |
| Songbird mesh discovery | **Pending** — solo mode validated by ludoSpring V70 |
| coralReef shader compilation via IPC | **Skip-safe** (honest skip) — awaiting coralReef IPC surface |
| toadStool `compute.dispatch.submit` | **Wired for validate/list** — full submit via IPC pending |
| Squirrel `inference.register_provider` | **Not wired** — awaiting Squirrel provider API |
| barraCuda `plasma_dispersion` feature gate | **Gap 9** — upstream fix needed |
| LTEE B2–B9, E2–E5 | **Queued** — B1 complete |

### Papers remaining from queue

**27/27 papers COMPLETE** (Papers 011–027). Queue CLOSED.

LTEE GuideStone queue (B2–B9, E2–E5) is separate backlog.

### Datasets to examine

| Dataset | Status | Scale |
|---------|--------|-------|
| ERA5 (Open-Meteo) | **Active** — Michigan + multi-city weather | ~GB class |
| MNIST | **Active** — LeNet study | Standard |
| WDM/FPEOS | **Active** — EOS tables | MB class |
| LTEE B1 | **Active** — Barrick 2009 accumulation | KB class |
| FAO-56 | **Active** — ET0 reference | KB class |
| **UniRef90** | **Future** — MSA for coralForge scale | ~100 GB |
| **LTEE SRA reads** | **Future** — structural evolution | ~200 GB |
| **OhioT1DM/OpenAPS** | **Future** — CGM glucose extension | GB class |
| **PDB structures** | **Future** — AlphaFold validation | Multi-GB |

---

## 3. S204 Code Fixes

| Fix | File(s) |
|-----|---------|
| `relu` evolved to `const fn` | `primitives.rs` |
| LTEE B1 `mul_add` evolution (3 sites) | `validate_ltee_b1_mutation_accumulation.rs` |
| Logger extracted to module level | `validate_ltee_b1_mutation_accumulation.rs` |
| `run_checks()` extracted from main | `validate_ltee_b1_mutation_accumulation.rs` |
| `merge_tracks()` DRY helper | `combiners.rs` |
| Unneeded `return` removed | `composition.rs` |
| 23 doc backtick fixes | `primal_names.rs`, `nestgate.rs`, `ipc/mod.rs` |
| `#![allow(deprecated)]` removed | `playGround/src/lib.rs` |
| Dev-deps workspace-inherited | `Cargo.toml` (`approx`, `proptest`, `serial_test`) |
| Discovery doc aligned with `temp_dir()` | `composition.rs` |
| BENCHMARK_ANALYSIS 14→15 domains fix | `specs/BENCHMARK_ANALYSIS.md` |
| Kokkos status corrected | `specs/BENCHMARK_ANALYSIS.md` |

---

## 4. Quality Gates

| Metric | Value |
|--------|-------|
| Workspace tests (IPC-first) | **907** (731 lib + 11 integration + 73 forge + 80 playGround + 12 exp094) |
| Certification tests | **19** (guideStone L0–L5) |
| Python baselines | **397/397 PASS** |
| Total validation checks | **4,900+** |
| Capabilities | **37** |
| IPC modules | **7** |
| Binaries | **269** (244 validate, 18 bench, 7 other) |
| Clippy | **0 errors** |
| `#[allow()]` | **0** |
| `unsafe` | **0** (`forbid` workspace-wide) |
| Production mocks | **0** |
| Production panics | **0** |
| TODO/FIXME/HACK | **0** |
| Files >800 LOC | **0** (max 776) |
| C dependencies | **0** (indirect via `wgpu` only) |

---

## 5. Upstream Gaps — For Primal Teams

### primalSpring

- **Gap 11 stale** — Layer 3 table still says "18 RPC methods" for neuralSpring. Resolved S201b. Flagged in V153, V154, V155, V156.
- **LIVE_SCIENCE_API.md** — `barracuda.precision.route` still listed as "NOT IMPLEMENTED". It has been implemented since barraCuda v0.4.0 (649 tests).

### barraCuda

- **Gap 9** — `plasma_dispersion` unconditional import behind `domain-lattice` feature. Workaround active (feature enabled). Fix is upstream.

### Ecosystem

- **Kokkos matched-hardware** — scaffold exists (`bench_kokkos_parity`), needs real Kokkos-CUDA numbers on same GPU.
- **Polybench/oneDNN** — not present, scope for barraCuda/toadStool teams.

---

*neuralSpring V156 | Session S204 | May 13, 2026*
