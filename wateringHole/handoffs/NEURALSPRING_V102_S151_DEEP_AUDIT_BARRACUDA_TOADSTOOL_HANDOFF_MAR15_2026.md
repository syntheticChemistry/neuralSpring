# neuralSpring → barraCuda/toadStool: Deep Audit & Evolution Handoff

**Date**: 2026-03-15
**From**: neuralSpring V102 / Session 151
**To**: barraCuda team, toadStool team, coralReef team
**License**: AGPL-3.0-or-later
**Covers**: Sessions 147–151 evolution (debt resolution, ecoBin compliance, capability discovery)
**Supersedes**: V101 S150 Compute Triangle Handoff (Mar 14, 2026)
**Pins**: barraCuda v0.3.5 at `0649cd0`, toadStool S146 at `751b3849`, coralReef Iteration 33

---

## Executive Summary

- **ecoBin compliance**: Eliminated `openssl-sys`/`native-tls` C dependency in playGround
  (`reqwest` default → `rustls-tls`). Main crates (`neural-spring`, `neural-spring-forge`)
  remain zero C deps. `ring` via `rustls` remains as transitional dep in playGround only —
  evolution path is Songbird for external HTTP/TLS.
- **Capability-based discovery**: ToadStool and coralReef IPC clients evolved from hardcoded
  primal names to `discover_by_capability()` — probes running primals for required capabilities
  (`compute.submit`, `shader.compile.wgsl`) via `capability.list` JSON-RPC. Fallback to
  name-based discovery preserved for backward compatibility.
- **Tolerance centralization**: 12 hardcoded tolerance values in test code migrated to named
  `tolerances::` constants. Zero inline magic numbers in any code path.
- **Code quality**: 0 clippy warnings (pedantic+nursery), 0 doc warnings (was 3), 0 unsafe,
  all `#[expect()]` with specific reasons, 1115+73+61 tests passing.
- **4 local-only WGSL shaders** remain without upstream equivalents — candidates for absorption.

---

## Part 1: Absorption Status

### Fully Lean (46 upstream rewires, 219 import files)

neuralSpring consumes barraCuda across 45+ submodules. All 17 shortcomings (S-01 through S-17)
are resolved upstream. Zero duplicate math in the crate.

| Namespace | Functions Used | Status |
|-----------|---------------|--------|
| `device` | `WgpuDevice`, `GpuDriverProfile`, `Fp64Strategy`, `PrecisionRoutingAdvice` | Lean |
| `tensor` | `Tensor` (84+ sites) | Lean |
| `ops::bio` | 18+ GPU bio kernels (BatchFitness, HMM, Pairwise, Wright-Fisher, etc.) | Lean |
| `ops::fft` | `Fft1D`, `Rfft`, `Fft1DF64`, `Ifft1D` | Lean |
| `ops::mha` | `MultiHeadAttention` | Lean |
| `ops::linalg` | `BatchedEighGpu` | Lean |
| `ops::logsumexp` | `LogSumExp` | Lean |
| `ops::fused_*` | `fused_chi_squared_f64`, `fused_kl_divergence_f64` | Lean |
| `dispatch` | `dispatch_for`, `DispatchTarget` | Lean |
| `stats` | Variance, Pearson, Shannon, ESD, Marchenko-Pastur | Lean |
| `spectral` | `BatchIprGpu`, `tridiag_eigenvectors` | Lean |
| `nn` | `SimpleMlp`, `DenseLayer`, `Activation` | Lean |
| `esn_v2` | `MultiHeadEsn`, ESN config | Lean |
| `nautilus` | `DriftMonitor`, `NautilusBrain`, `BetaObservation` | Lean |
| `staging` | `StatefulPipeline`, `KernelDispatch` | Lean |
| `special` | Chi-squared, gamma, erf | Lean |
| `shaders::provenance` | Shader provenance metadata | Lean |

### 4 Local-Only Shaders (Absorption Candidates)

| Shader | metalForge Path | Purpose | Upstream Equivalent |
|--------|----------------|---------|---------------------|
| `HEAD_SPLIT` | `shaders/head_split.wgsl` | MHA head decomposition (S-03b workaround) | None — upstream MHA uses fused path |
| `HEAD_CONCAT` | `shaders/head_concat.wgsl` | MHA head reassembly | None — pair with HEAD_SPLIT |
| `XOSHIRO128SS` | `shaders/xoshiro128ss.wgsl` | GPU PRNG (f32) | `barracuda::ops::prng` (different API) |
| `SWARM_NN_SCORES` | `shaders/swarm_nn_scores.wgsl` | Multi-objective swarm scoring | None |

**Recommendation**: If barraCuda's MHA remains fused, HEAD_SPLIT/CONCAT can stay local. XOSHIRO128SS
should converge when `barracuda::ops::prng` exposes a f32 uniform API. SWARM_NN_SCORES is
domain-specific and may stay in metalForge.

---

## Part 2: Upstream Requests

### P0 — Critical

| Request | Context | Suggested Fix |
|---------|---------|---------------|
| `enable f64;` PTXAS silent-zero regression | Ada Lovelace (SM89, RTX 40xx) + Vulkan proprietary driver. WGSL shaders with `enable f64;` compile and dispatch without error but produce all-zero outputs. | **Strip `enable f64;`** from WGSL source before compilation — naga resolves f64 from device caps. neuralSpring has local fix in metalForge. See `NEURALSPRING_V95_ENABLE_F64_FIX_HANDOFF_MAR10_2026.md` for full details. |

### P1 — Important

| Request | Context |
|---------|---------|
| `TensorSession` documentation & examples for multi-op fusion | neuralSpring playGround uses `TensorSession` for hot dispatch (7–45× speedup). Main library still uses per-op `Tensor` dispatch. Examples showing HMM chains, ODE loops, and attention pipelines via `TensorSession` would accelerate adoption. |
| `ops::prng` f32 uniform API | neuralSpring maintains local `xoshiro128ss.wgsl` because `barracuda::ops::prng` uses a different binding layout. A public `uniform_f32(seed, count) → Tensor` would allow absorption. |

### P2 — Stretch

| Request | Context |
|---------|---------|
| `WGSL_MEAN_REDUCE` public constant | metalForge keeps a local copy of `mean_reduce.wgsl` because upstream exposes it via `LazyLock<String>`. A `pub const` re-export would clean up the dependency. |
| Hamming distance f64 performance | `PairwiseHammingGpu` 200×500 f64 is 20.85× slower than f32 equivalent. Size-based f32/f64 routing or exposing a public f32 Hamming constant would help. |

---

## Part 3: Capability-Based Discovery Evolution

neuralSpring S151 evolved all IPC clients from hardcoded primal names to
capability-based discovery. This pattern is recommended for all springs:

```
// Old: hardcoded primal name
ipc_client::discover_socket("toadstool")

// New: capability-based (probes all sockets via `capability.list`)
ipc_client::discover_by_capability("compute.submit", "toadstool")
```

The new `discover_by_capability()` function:
1. Scans the biomeOS socket directory for all `.sock` files
2. Sends `capability.list` JSON-RPC to each
3. Matches the first primal advertising the required capability
4. Falls back to name-based discovery if no capability probe succeeds

**toadStool action**: Ensure `capability.list` returns all advertised capabilities
including `compute.submit`, `compute.capabilities`, `gpu.info`, `science.gpu.dispatch`,
`science.substrate.discover`.

**coralReef action**: Ensure `capability.list` returns `shader.compile.wgsl`,
`shader.compile.spirv`, `shader.compile.capabilities`, `shader.compile.status`.

---

## Part 4: ecoBin Compliance Observations

neuralSpring's main crates (`neural-spring`, `neural-spring-forge`) are fully ecoBin
compliant: zero C dependencies, cross-compilation ready.

The `playGround` workspace member has `ring` via `reqwest → rustls`. This is the
standard Rust TLS stack and compiles from vendored source (no system library required),
but `ring` is explicitly listed in the ecoBin exclusion list. The evolution path:

1. **Short term**: `ring` via `rustls` is acceptable for development tooling
2. **Medium term**: Route external HTTP through Songbird (the network primal)
3. **Long term**: All springs use Songbird for TLS; `ring` removed from dependency tree

**biomeOS action**: Songbird should expose a `http.fetch` capability via JSON-RPC
for springs that need external HTTP access (HuggingFace Hub, dataset downloads).

---

## Part 5: Validation Metrics (S151)

| Metric | Count |
|--------|-------|
| Python baselines | 397/397 PASS |
| Rust lib tests | 1115 PASS |
| Forge tests | 73 PASS |
| playGround tests | 61 PASS |
| Integration tests | 13 PASS |
| validate_all | 220/220 PASS |
| barraCuda CPU | 272/272 PASS |
| barraCuda CPU ports | 203/203 (24/25 papers) |
| GPU tensor | 98+ checks (23/25 papers) |
| GPU shader | 108/108 PASS |
| Cross-dispatch | 49/49 PASS |
| Dispatch parity | 30/30 (CPU↔GPU identical) |
| Multi-GPU | 384/384 bit-identical |
| Coverage | 91.66% (llvm-cov, --lib) |
| Clippy | 0 warnings (pedantic+nursery) |
| Doc warnings | 0 |
| Unsafe | 0 (forbidden) |
| `#[allow()]` in production | 1 (tolerances/registry.rs, justified) |
| Files > 1000 LOC | 0 |
| SPDX headers | 425/425 |

---

## Part 6: Spring Guidance for Niche Deployment

neuralSpring's evolution from V1 to V102 provides a template for new springs:

1. **Start with Python baselines** — reproducible, deterministic, seed-fixed
2. **Port to Rust** — delegate all math to barraCuda from day one
3. **Centralize tolerances** — named constants with mathematical justification
4. **Provenance everything** — every hardcoded value traces to script + commit + date
5. **hotSpring validation pattern** — `ValidationHarness`, exit 0/1, `check_abs`/`check_rel`
6. **Capability-based discovery** — never hardcode primal names in production
7. **ecoBin from day one** — zero C deps in application code
8. **metalForge for local shaders** — Write → Validate → Handoff → barraCuda Absorbs → Lean

---

## Action Items

### barraCuda team
- [ ] Absorb `enable f64;` strip fix (P0 — affects all springs on Ada Lovelace)
- [ ] Consider `ops::prng` public f32 uniform API for XOSHIRO128SS absorption
- [ ] Consider `WGSL_MEAN_REDUCE` public constant export
- [ ] Document `TensorSession` patterns for multi-op fusion

### toadStool team
- [ ] Ensure `capability.list` RPC returns all advertised capabilities
- [ ] Consider Songbird `http.fetch` capability for spring HTTP needs

### coralReef team
- [ ] Ensure `capability.list` RPC returns shader compilation capabilities

### neuralSpring (next session)
- [ ] Evaluate `TensorSession` for HMM chains, ODE loops in main library
- [ ] Track XOSHIRO128SS upstream convergence
- [ ] Track HEAD_SPLIT/CONCAT relevance as barraCuda MHA evolves
