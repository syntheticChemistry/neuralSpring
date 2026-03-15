# neuralSpring V101 S150 → barraCuda / toadStool / coralReef: Compute Triangle Integration

**Date:** March 14, 2026
**From:** neuralSpring S150 (1115 lib + 63 playGround unit + 13 integration tests, 260 binaries, 0 clippy)
**To:** barraCuda, toadStool, coralReef teams
**Scope:** Compute triangle integration, hot/cold dispatch benchmarks, typed IPC clients, live primal coordination
**Supersedes:** V100 S147 Deep Debt Evolution Handoff (Mar 14, 2026)
**Pins:** barraCuda v0.3.5 at `0649cd0`, toadStool S146+, coralReef Phase 10
**License:** AGPL-3.0-or-later

---

## Executive Summary

- **Compute triangle proven live** — neuralSpring now has typed IPC clients for ToadStool and coralReef, with live coordination verified against a running ToadStool server
- **Hot/cold dispatch quantified** — pipeline compilation overhead measured: matmul 7×, layer_norm 10×, GELU 45× speedup from cold→hot dispatch (session reuse vs. per-call session creation)
- **Remaining PyTorch/CUDA gap identified** — 8–22× slower than PyTorch/CUDA at hot dispatch; bind-group creation + Vulkan submit overhead dominates
- **63 unit + 13 integration tests** across playGround modules (MCP tools, IPC client, HuggingFace Hub, model config, weights, transformer, secrets)
- **End-to-end GPT-2 forward pass on barraCuda** — download from HuggingFace, load safetensors, run transformer inference entirely on GPU

---

## Part 1: Compute Triangle Architecture (What We Built)

```
WGSL source → coralReef (compile) → ToadStool (orchestrate) → barraCuda (execute)
```

### New IPC Clients

| Client | Socket | Methods | Status |
|--------|--------|---------|--------|
| `ToadStoolClient` | `toadstool.jsonrpc.sock` | `health()`, `capabilities()`, `gpu_info()` | Live verified |
| `CoralReefClient` | `coralreef.jsonrpc.sock` | `compile_wgsl()`, `compile_wgsl_multi()`, `compile_spirv()`, `status()`, `capabilities()` | Schema defined, awaiting daemon |

Both clients use `ipc_client::discover_socket()` with biomeOS 5-tier socket resolution and explicit JSON-RPC socket preference.

### Diagnostic Binary

`neuralspring_compute_probe` — probes all three compute tiers:

| Tier | Source | What It Checks |
|------|--------|----------------|
| Tier 0 | barraCuda (local wgpu) | `WgpuDevice::new()`, adapter info, pipeline dispatch latency |
| Tier 1 | ToadStool (IPC) | Socket discovery, health, capabilities, GPU devices |
| Tier 2 | coralReef (IPC) | Socket discovery, status, compiler capabilities |

---

## Part 2: Dispatch Overhead Analysis (Critical for barraCuda)

### Cold vs. Hot Dispatch (RTX 4070, 512 seq_len, 768 hidden)

| Operation | Cold (ms) | Hot (ms) | Speedup | Bottleneck |
|-----------|-----------|----------|---------|------------|
| matmul | 6.86 | 0.93 | 7.4× | Pipeline compilation in cold |
| layer_norm | 5.58 | 0.54 | 10.3× | Pipeline compilation in cold |
| softmax | 5.72 | 0.57 | 10.0× | Pipeline compilation in cold |
| GELU | 5.41 | 0.12 | 45.1× | Pipeline compilation dominates cold |

**Key finding:** Per-op dispatch overhead (~200 µs per `queue.submit()`) dominates small workloads. Pipeline reuse via `TensorSession::reset()` eliminates recompilation but bind-group recreation remains.

### Comparison to PyTorch/CUDA

| Operation | PyTorch/CUDA (ms) | barraCuda hot (ms) | Gap |
|-----------|-------------------|-------------------|-----|
| matmul | 0.11 | 0.93 | 8.5× |
| layer_norm | 0.03 | 0.54 | 18× |
| softmax | 0.03 | 0.57 | 19× |
| GELU | 0.005 | 0.12 | 24× |

**Root causes (for barraCuda team):**
1. **Bind-group creation** — wgpu creates new bind groups per dispatch even with pipeline reuse
2. **Vulkan submit overhead** — wgpu's `queue.submit()` has ~200 µs fixed cost vs. CUDA's ~5 µs kernel launch
3. **No persistent buffer binding** — CUDA reuses buffer addresses; wgpu invalidates bind groups on each submit
4. **No command batching** — single-encoder-per-submit vs. CUDA's stream model

### barraCuda action: Priority dispatch evolution

1. **P0 — Bind-group caching**: Cache bind groups keyed on `(pipeline_id, buffer_address_set)` — eliminates per-dispatch allocation
2. **P0 — Command batching**: Batch multiple shader dispatches into a single `queue.submit()` — amortizes 200 µs overhead across N ops
3. **P1 — Pre-allocated buffer pools**: Reuse GPU buffers by shape — eliminates per-inference allocation
4. **P2 — CoralReef sovereign dispatch**: Bypass Vulkan entirely using `coral-driver` DRM ioctl dispatch — eliminates wgpu overhead (requires coralReef integration)

---

## Part 3: ToadStool Schema Discovery

We reverse-engineered the ToadStool JSON-RPC schema by interrogating a live server:

### `compute.health` Response

```json
{
  "healthy": true,
  "uptime_seconds": 42,
  "active_jobs": 0,
  "total_jobs_completed": 0,
  "version": "0.5.0"
}
```

### `compute.capabilities` Response

```json
{
  "supported_workload_types": ["gpu_compute", "shader_compile", ...],
  "max_concurrent_jobs": 8,
  "gpu_available": true,
  "features": ["wgsl_compute", "f32_compute", ...]
}
```

### `gpu.info` Response

```json
{
  "devices": [
    {
      "name": "NVIDIA GeForce RTX 4070",
      "vendor": 4318,
      "device_type": "DiscreteGpu",
      "backend": "Vulkan",
      "driver": "560.35.03",
      "driver_info": "560.35.03"
    }
  ]
}
```

**toadStool action:** Consider publishing JSON Schema or OpenAPI definitions for your JSON-RPC API to prevent client drift.

---

## Part 4: Safetensors Alignment Fix (For barraCuda Awareness)

We discovered that `safetensors` file data is **not guaranteed 4-byte aligned**. `bytemuck::cast_slice::<u8, f32>()` panics on unaligned data.

**Fix:** `safe_cast_slice<T: Pod>()` — copies bytes into an aligned `Vec<T>` using `memcpy` semantics:

```rust
fn safe_cast_slice<T: bytemuck::Pod>(data: &[u8]) -> Vec<T> {
    let size = std::mem::size_of::<T>();
    let count = data.len() / size;
    let mut out = vec![T::zeroed(); count];
    let dst = bytemuck::cast_slice_mut::<T, u8>(&mut out);
    dst.copy_from_slice(&data[..count * size]);
    out
}
```

**barraCuda action:** If barraCuda ever loads safetensors directly, it must handle unaligned data. Consider adding `Tensor::from_unaligned_bytes()` to the public API.

---

## Part 5: Test Coverage (playGround)

| Module | Unit Tests | Integration Tests | Coverage |
|--------|-----------|-------------------|----------|
| `mcp_tools` | 6 | — | Tool registry, schema, serialization |
| `ipc_client` | 7 | — | Timeout, socket resolution, family ID, discovery |
| `hf_hub` | 8 | 2 | Cache dirs, model info, completeness, API calls |
| `model_config` | 9 | — | GPT-2/Llama/Phi parsing, defaults, memory estimation |
| `inference/weights` | 16 | 1 | f16/bf16 conversion, layer extraction, alignment |
| `inference/transformer` | 10 | 1 | top_k, softmax, end-to-end GPT-2 forward pass |
| `secrets` | 3 | — | Token loading, env var fallback |
| `toadstool_client` | — | 4 | Health, capabilities, GPU info, discovery |
| `coralreef_client` | — | 3 | Status, capabilities, shader compilation |
| `barracuda_gpu` | — | 2 | Basic session, pipeline reuse |
| **Total** | **63** | **13** | |

### Live Verification Results (ToadStool running)

10/13 integration tests pass. 3 failures are coralReef tests (daemon not running). All ToadStool, GPU, HuggingFace, and E2E tests pass.

---

## Part 6: What toadStool/barraCuda/coralReef Should Consider

### For barraCuda

1. **Bind-group caching** — highest-impact optimization for closing the PyTorch/CUDA gap
2. **`Tensor::from_unaligned_bytes()`** — safe loading from external formats (safetensors, Arrow IPC, etc.)
3. **`TensorSession` hot-path documentation** — document `reset()` semantics and pipeline reuse guarantees
4. **Command batching API** — allow batching N dispatches into one submit for small-kernel workloads

### For toadStool

1. **JSON-RPC schema publication** — prevents client-side struct drift
2. **Socket naming convention** — `*.jsonrpc.sock` vs `*.sock` dual-socket pattern needs documentation
3. **Pipeline cache integration** — expose persistent compiled pipeline cache to reduce cold-start overhead

### For coralReef

1. **Native dispatch path** — sovereign dispatch bypassing Vulkan is the long-term answer to the 8–22× gap
2. **Shader pre-compilation API** — allow springs to pre-compile WGSL at build time and cache native binaries

### Carried Forward from V100

All P0/P1/P2 items from V100 (softmax dispatch overhead, RFFT structural gap, MHA fused kernel, logsumexp, pairwise_distance, StatefulPipeline) remain valid targets.

---

## Part 7: Code Health

| Metric | Value |
|--------|-------|
| Library tests | 1115 pass |
| playGround unit tests | 63 pass |
| playGround integration tests | 13 (10 pass, 3 need coralReef) |
| Forge tests | 73 pass |
| Validation binaries | 260 |
| Clippy warnings | 0 (pedantic + nursery) |
| Doc warnings | 0 |
| Unsafe code | 0 (`#![forbid(unsafe_code)]`) |
| Max file size (library) | 812 LOC |
| External deps | All pure Rust |
| License | AGPL-3.0-or-later |
