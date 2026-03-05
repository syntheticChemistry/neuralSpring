<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/barraCuda Handoff V83 — wgpu 28 + BarraCUDA v0.3.3 + ToadStool S94b Sync

**Date**: March 5, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/barraCuda team
**License**: AGPL-3.0-or-later
**Covers**: Session 125 — wgpu 22→28 migration, BarraCUDA v0.3.1→v0.3.3 sync, ToadStool S87→S94b pin update
**Supersedes**: V82 (S122–S124 naming rewire + HMM absorption + Paper 026)
**barraCuda**: v0.3.3 standalone (`../barraCuda/crates/barracuda`)
**ToadStool HEAD**: `9d359814` (S94b)

---

## Executive Summary

- **wgpu 22 → 28 migration**: ~70 API call sites updated across neuralSpring `src/` and `metalForge/forge/`. Six distinct API changes applied mechanically. Zero code logic changes.
- **BarraCUDA v0.3.1 → v0.3.3**: Removed `unidirectional` feature (dropped in v0.3.2). neuralSpring now consumes the wgpu 28 stack, `GuardedDeviceHandle`, fused reduction shaders, DF64 precision tier, and subgroup capability detection.
- **ToadStool S87 → S94b**: 9 upstream commits reviewed. `BarraCUDA` extracted to standalone primal, D-SOV resolved, `NpuDispatch` added, REST removed.
- **Quality gates**: fmt ✓ · clippy ✓ (0 warnings, pedantic+nursery) · test 871/880 lib (9 GPU SIGSEGV — upstream) · doc ✓

---

## Part 1: wgpu 22 → 28 Migration

### API Changes Applied

| Change | Sites | Pattern |
|--------|-------|---------|
| `Maintain::Wait` → `PollType::Wait` | 13 | `wgpu::PollType::Wait { submission_index: None, timeout: None }` |
| `push_constant_ranges: &[]` → `immediate_size: 0` | 19 | `PipelineLayoutDescriptor` field rename |
| `entry_point: "name"` → `entry_point: Some("name")` | 19 | `ComputePipelineDescriptor` field type change |
| `set_bind_group(0, &bg, &[])` → `set_bind_group(0, Some(&bg), &[])` | 17 | Second parameter wrapped in `Option` |
| `Instance::new(owned)` → `Instance::new(&ref)` | 1 | `gpu.rs` + `probe.rs` |
| `enumerate_adapters()` sync → async | 2 | `.await` added in `gpu.rs` |
| `DeviceDescriptor` new fields | 1 | `experimental_features` + `trace` added |
| `from_existing(Arc, Arc, info)` → `from_existing(owned, owned, info)` | 1 | wgpu 28 removes `Arc` wrapper for Device/Queue |

### Files Touched

- **Library**: `gpu.rs`, `bench.rs`, `gpu_shader_validation.rs`, `gpu_ops/bio/hmm.rs`, `gpu_ops/bio/evolution.rs`, `gpu_ops/bio/activation.rs`, `gpu_ops/eigensolver.rs`, `gpu_ops/ode_batch.rs`
- **Binaries**: 17 `validate_gpu_*.rs` + `validate_mha_gpu.rs` + `bench_modern_rewire.rs` + `validate_upstream_*.rs`
- **metalForge**: `forge/src/probe.rs` (Instance::new + enumerate_adapters via pollster)

### Dependencies Updated

| Dependency | Before | After |
|-----------|--------|-------|
| `wgpu` | 22 | 28 |
| `tokio` | 1.35 | 1.49 |
| `barracuda` | v0.3.1 (+ `unidirectional`) | v0.3.3 (no features) |
| `pollster` | — | 0.4 (metalForge/forge only) |

---

## Part 2: BarraCUDA v0.3.3 Capabilities Now Available

| Capability | Version | Status in neuralSpring |
|-----------|---------|----------------------|
| `GuardedDeviceHandle` (RAII encoder barriers) | v0.3.2 | Available, not yet used |
| Fused mean+variance shaders (f64 and DF64) | unreleased | Available via `VarianceF64::mean_variance()` |
| `CorrelationResult` + fused correlation shaders | unreleased | Available, not yet used |
| `ComputeDispatch::df64()` DF64 shader path | unreleased | Available for consumer GPU precision |
| Subgroup capability detection | unreleased | Available via `DeviceCapabilities` |
| Three-tier precision model (f32/DF64/f64) | unreleased | Available, not yet used |
| Workgroup size constants (`WORKGROUP_SIZE_1D = 256`) | v0.3.3 | Available |

### Absorption Opportunities

| Item | What | Effort |
|------|------|--------|
| `VarianceF64::mean_variance()` | Replace separate mean + variance GPU calls with fused shader | Low |
| `CorrelationResult` | Replace manual Pearson GPU implementation | Low |
| `ComputeDispatch::df64()` | Use DF64 path on consumer GPUs for better throughput | Medium |
| `GuardedDeviceHandle` | Replace manual device polling with RAII barriers | Low |

---

## Part 3: ToadStool S87 → S94b Review

| Session | Key Change | neuralSpring Impact |
|---------|-----------|-------------------|
| S88 | Cross-spring absorption, shader evolution | None — already consumed via BarraCUDA |
| S89 | BarraCUDA extraction to standalone primal | Already rewired in S118 |
| S90-92 | REST removal, sovereignty deprecation, coverage | None — neuralSpring uses JSON-RPC |
| S93 | D-DF64 transfer to BarraCUDA, debris cleanup | DF64 now available in BarraCUDA |
| S94b | NpuDispatch, GpuAdapterInfo, D-SOV resolved | New APIs available for future use |

### New ToadStool APIs (Not Yet Used by neuralSpring)

- `NpuDispatch` / `AkidaNpuDispatch` — NPU inference dispatch
- `NpuParameterController` — NPU-driven parameter tuning
- `GpuAdapterInfo` / `GpuDeviceType` — structured adapter metadata
- `get_socket_path_for_capability()` — capability-based discovery (D-SOV resolved)

---

## Part 4: Known Issues

### GPU Tensor SIGSEGV (9 tests)

9 GPU Tensor-based tests fail with SIGSEGV after the wgpu 28 migration:

```
gpu_ops::tests_ops::gpu_variance_known
gpu_ops::tests_ops::gpu_chi_squared_known
gpu_ops::tests_ops::gpu_pearson_perfect_correlation
gpu_ops::tests_ops::gpu_matrix_correlation_self
gpu_ops::tests_ops::gpu_thermal_diversity_basic
gpu_dispatch::tests_gpu::gpu_chi_squared
gpu_dispatch::tests_gpu::gpu_matrix_correlation
gpu_dispatch::tests_gpu::gpu_pearson_correlation
gpu_dispatch::tests_gpu::gpu_thermal_diversity_correlation
```

**Root cause**: Upstream — barracuda's own variance tests also SIGSEGV on this hardware (Linux, llvmpipe/NVK). The `Tensor::from_data` → GPU reduction pipeline crashes at the driver level with wgpu 28. This is a barracuda/wgpu 28 runtime issue, not a neuralSpring code issue.

**Recommendation**: Track in BarraCUDA issue tracker. These tests passed with wgpu 22.

---

## Part 5: Lint Evolution

4 `#[expect]` attributes became unfulfilled with Rust 1.93 / clippy updates:

| Location | Lint | Resolution |
|----------|------|------------|
| `anderson_localization.rs` tests | `float_cmp` | Removed (no longer triggered) |
| `gpu_dispatch/basecamp.rs` tests | `float_cmp` | Removed (no longer triggered) |
| `tolerances/registry.rs` | `wildcard_imports` | Changed to `#[allow]` (still needed but not warned) |
| `weight_loader.rs` test | `cast_possible_truncation` | Removed (no longer triggered) |

---

## Quality Gates (S125)

| Gate | Result |
|------|--------|
| `cargo fmt -- --check` | Clean |
| `cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery` | 0 warnings |
| `cargo test --lib` | 871/880 PASS (9 GPU SIGSEGV — upstream) |
| `cargo doc --no-deps` | 0 warnings |
| wgpu API sites migrated | ~70 |
| Files modified | ~40 |

---

## Counts

| Metric | Value |
|--------|-------|
| Library tests | 880 (871 PASS, 9 GPU upstream SIGSEGV) |
| Validation/bench binaries | 238 |
| `validate_all` | 217/217 |
| ToadStool HEAD | `9d359814` (S94b) |
| BarraCUDA version | v0.3.3 |
| wgpu version | 28 |
| clippy warnings | 0 (pedantic+nursery) |
| doc warnings | 0 |

---

*V83 — neuralSpring Session 125 (March 5, 2026)*
