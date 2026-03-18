# Ecosystem Leverage Guide — neuralSpring

**Last Updated**: March 17, 2026 (Session 165 — FMA sweep, IPC proptest, ecosystem absorption)
**Purpose**: Map what neuralSpring absorbs from the ecoPrimals ecosystem and how other
components can compose with neuralSpring.

---

## What We Absorb

### barraCuda (Pure Math Engine)

| Category | What We Use | Where |
|----------|------------|-------|
| GPU device | `WgpuDevice`, `new()`, `new_cpu_relaxed()`, `new_gpu()` | `gpu.rs` |
| Precision routing | `Precision` (F32/F64/Df64), `GpuDriverProfile`, `Fp64Strategy` | `gpu.rs`, `gpu_dispatch/` |
| Shader compilation | `compile_shader`, `compile_shader_f64`, `compile_shader_df64` | `gpu.rs` |
| Statistics | `stats::correlation::variance`, `pearson_correlation`, `rank_*` | 14+ modules |
| Linear algebra | `linalg::solve::solve_f64_cpu`, `linalg::tridiag_eigen` | `glucose_prediction.rs`, `eigh.rs` |
| Tensor ops | `tensor::create_tensor_f32`, GPU matmul, reductions | `gpu_ops/`, validation binaries |
| Dispatch | `dispatch::matmul_dispatch`, `compute_dispatch` | `gpu_dispatch/` |
| Neural ops | `nn::simple_mlp`, activations, `esn_v2` | `wdm_esn/`, `lenet.rs` |
| Bio ops | 18+ GPU kernels: HMM, eco, population, game theory | `gpu_ops/bio/` |
| FFT | `ops::fft` GPU pipeline | `fft.rs`, validation binaries |
| Special functions | `special::log_sum_exp`, Gaussian CDF | `hmm.rs`, `neural_pgm.rs` |
| Numerical | `numerical::rk4`, ODE solvers | `primitives.rs` |

**Version**: v0.3.5 at `0649cd0` (path dep `../barraCuda/crates/barracuda`)
**Evolution**: We delegate to barraCuda when it has the primitive. Local code stays only
for (a) tiny matrices where dispatch overhead dominates, or (b) pending absorption.

### toadStool (Hardware Orchestration)

| What | How | Status |
|------|-----|--------|
| `ComputeDispatch` | Via JSON-RPC IPC | Wired, capability-probed |
| Hardware discovery | `unified_hardware::BandwidthTier` | Via barraCuda re-export |
| PCIe transport | Tensor transfer validation | Validation binaries |
| Pipeline DAG | Absorbed from neuralSpring S134 | Upstreamed |

**Discovery**: `discover_primal("toadstool")` or `discover_by_capability("compute.submit", "toadstool")`

### coralReef (Sovereign Shader Compiler)

| What | How | Status |
|------|-----|--------|
| Sovereign compile | `compile_shader_df64` path | Via barraCuda's compiler bridge |
| ILP optimizer | Instruction-level parallelism | Transparent via barraCuda |
| Multi-GPU | NVIDIA + NVK bit-identical | Validated in cross-spring binaries |

**Discovery**: Transparent through barraCuda — no direct IPC needed.

### biomeOS (Orchestration Platform)

| What | How | Status |
|------|-----|--------|
| Socket resolution | 5-tier: `BIOMEOS_SOCKET_DIR` → `XDG_RUNTIME_DIR` → `/run/user/{uid}` → `temp_dir()` | `ipc_client.rs` |
| FAMILY_ID routing | `{name}-{family_id}.sock` for multi-instance | `discover_socket()` |
| Capability discovery | `capability.list` JSON-RPC probe | `discover_by_capability()` |

### Patterns Absorbed from Sibling Springs

| Pattern | Origin | Where Applied |
|---------|--------|--------------|
| `ValidationHarness` (pass/fail, exit 0/1) | hotSpring | 260 validation binaries |
| `OnceLock` GPU probe caching | hotSpring/toadStool | `gpu.rs` test module |
| `mul_add()` FMA precision | wetSpring V0.5.0 | 14 sites across 10 library modules |
| `total_cmp()` float sorting | wetSpring | All eigenvalue sorts |
| `DispatchOutcome` classification | groundSpring V112 | `ipc_client.rs` |
| `IpcError` typed phases | healthSpring V31 | `ipc_client.rs` |
| `RetryPolicy` + `CircuitBreaker` | healthSpring V32 | `ipc_resilience.rs`, `ipc_client.rs` |
| `extract_rpc_error` centralised | airSpring V0.8.6 | `ipc_client.rs` |
| `parse_capability_list` (5 formats) | airSpring V0.8.7 | `ipc_client.rs` |
| FAMILY_ID multi-family discovery | groundSpring V112 | `ipc_client.rs` |
| `proptest` IPC invariants | groundSpring V113 | `property_tests.rs` |
| Platform-agnostic paths | ecosystem-wide | `std::env::temp_dir()` everywhere |
| `#[expect()]` with reasons | ecosystem-wide | Zero `#[allow()]` in production |

---

## How Others Compose with neuralSpring

### Registered Capabilities

neuralSpring advertises via `capability.list`:

| Capability | Description |
|-----------|-------------|
| `health.liveness` | Health check — returns `"ok"` |
| `health.readiness` | Readiness check with GPU/module probe |
| `science.ipr` | Inverse participation ratio calculation |
| `science.spectral_analysis` | Attention matrix spectral analysis |
| `science.anderson_localization` | Anderson disorder diagnostics |
| `science.depth_scale` | Signal propagation depth scale |
| `science.gate_disorder` | LSTM gate disorder parameter |
| `science.information_flow` | Full information flow analysis |
| `compute.softmax` | Softmax computation |
| `compute.relu` | ReLU activation |
| `compute.sigmoid` | Sigmoid activation |
| `compute.rk4` | Runge-Kutta 4th order integration |
| `compute.shannon_entropy` | Shannon entropy calculation |
| `compute.fft` | Fast Fourier transform |
| `visualization.push` | Push visualization data to petalTongue |
| `provenance.register` | Register experiment provenance |

### IPC Protocol

- **Transport**: JSON-RPC 2.0 over Unix domain sockets
- **Socket name**: `neuralspring-{FAMILY_ID}.sock` (or `neuralspring.sock` default)
- **Timeout**: 5s default (`PRIMAL_IPC_TIMEOUT_SECS` env override)
- **Error handling**: Typed `IpcError` with `is_recoverable()` for retry logic

### Self-Knowledge

| Constant | Value | Location |
|----------|-------|----------|
| `NICHE_NAME` | `"neuralspring"` | `src/niche.rs` |
| `NICHE_DOMAIN` | `"ML primitives + scholarly reproduction"` | `src/niche.rs` |
| `FAMILY_ID` | Runtime from `FAMILY_ID` env | `ipc_client.rs` |

---

## Evolution Readiness

### What Stays Local (by Design)

- `softmax_rows`: f64 validation reference (barraCuda GPU softmax is f32)
- `mat_mul_transpose`: n=4..8 Jacobian matrices (dispatch overhead > compute)
- `eigh_householder_qr`: Validation reference independent of upstream

### What Should Migrate Upstream

- Local WGSL shaders in `metalForge/forge/shaders/` → barraCuda absorption
- `GpuSoftmaxF64` pipeline → barraCuda `TensorSession` when f64 softmax matures
- Validation harness patterns → potential `primalSpring` extraction
