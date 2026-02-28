# NUCLEUS Tower Mode Integration Plan

**Date**: February 28, 2026
**Purpose**: Wire neuralSpring into biomeOS NUCLEUS local Tower mode
**Current State**: `neuralspring_primal.rs` exists with 7 science capabilities,
JSON-RPC 2.0 over Unix sockets, biomeOS 5-tier socket resolution
**Target**: Full Tower mode on Eastgate — automated startup, health monitoring,
capability registration with biomeOS discovery, cross-primal communication

---

## What Exists

### neuralSpring Primal Binary

`src/bin/neuralspring_primal.rs` — a working JSON-RPC 2.0 server:

| Component | Status | Notes |
|-----------|--------|-------|
| Unix socket listener | Implemented | 5-tier biomeOS resolution |
| JSON-RPC dispatch | Implemented | 7 methods + health |
| Concurrency control | Implemented | Semaphore(4) |
| Family ID support | Implemented | `FAMILY_ID` / `BIOMEOS_FAMILY_ID` |
| Feature gate | Implemented | `--features primal` |

### Current Capabilities (7 methods)

| Method | Module | Compute Profile |
|--------|--------|----------------|
| `science.ipr` | `anderson_localization` | CPU, O(N) |
| `science.disorder_sweep` | `anderson_localization` | CPU, O(N² × W) |
| `science.spectral_analysis` | `eigh` + `weight_spectral` | CPU, O(N³) |
| `science.anderson_localization` | `anderson_localization` | CPU, O(N³ × W) |
| `science.hessian_eigen` | `eigh` + `primitives` | CPU, O(N³) |
| `science.agent_coordination` | `agent_coordination` | CPU, O(N² × W) |
| `science.training_trajectory` | `eigh` + `primitives` | CPU, O(N³ × E) |

### biomeOS SDK Integration

| Component | Status | Location |
|-----------|--------|----------|
| `biomeos-primal-sdk` dependency | Optional (`primal` feature) | `Cargo.toml` |
| `PrimalCapability::science()` | Mapped | `biomeos-types` |
| Provider mapping | `neuralspring` registered for `science` | `discovery.rs` |
| `UniversalPrimalService` trait | Available | `biomeos-types` |

---

## What Needs Building

### Step 1: Sovereign Folding Capabilities (expose nF-01/02 pipeline)

The primal currently only exposes baseCamp spectral analysis. Session 90
validated a full AlphaFold2 Evoformer + Structure Module pipeline — these
should be available over JSON-RPC for cross-primal use.

**New methods to add to `neuralspring_primal.rs`**:

| Method | Handler | Input | Output |
|--------|---------|-------|--------|
| `science.evoformer_block` | Run one Evoformer block iteration | `{n_seq, n_res, c_msa, c_pair, seed}` | MSA/pair tensors, tri_attn scores |
| `science.structure_module` | Run one Structure Module step | `{n_res, c_single, c_pair, seed}` | IPA scores, backbone, torsions |
| `science.folding_health` | Report folding primitive availability | — | Primitive count, GPU availability |
| `science.gpu_dispatch` | Run a Dispatcher operation on GPU | `{op, params}` | Result tensor |

**Implementation approach**: Thin wrappers around existing functions in
`src/sovereign_folding/` and `src/gpu_dispatch/`. The handlers serialize
inputs/outputs as JSON arrays (same pattern as the existing handlers).

### Step 2: GPU-Aware Health Check

The current `health` handler reports capabilities but not hardware status.
Tower mode needs GPU availability reporting for NUCLEUS compute routing.

**Enhanced health response**:

```json
{
  "status": "healthy",
  "primal": "neuralspring",
  "version": "0.7.0",
  "capabilities": ["science.spectral_analysis", "..."],
  "hardware": {
    "gpu_available": true,
    "gpu_name": "NVIDIA RTX 4070",
    "gpu_vram_mb": 12288,
    "gpu_driver": "wgpu/vulkan",
    "fp64_strategy": "Hybrid",
    "cpu_cores": 16,
    "ram_mb": 32768
  },
  "stats": {
    "requests_served": 42,
    "uptime_seconds": 3600,
    "active_connections": 1
  }
}
```

**Implementation**: Query `wgpu::Adapter::get_info()` at startup, cache in
a static. Increment counters per request.

### Step 3: biomeOS Registration Protocol

When the primal starts, it should announce itself to the local biomeOS
orchestrator (if running) via the NUCLEUS 5-layer discovery protocol.

**Startup sequence**:

1. Resolve socket path (existing)
2. Bind listener (existing)
3. **NEW**: Probe for biomeOS orchestrator at `$XDG_RUNTIME_DIR/biomeos/biomeOS.sock`
4. **NEW**: If found, send registration: `{ "jsonrpc": "2.0", "method": "nucleus.register", "params": { "primal": "neuralspring", "socket": "<path>", "capabilities": [...], "hardware": {...} } }`
5. **NEW**: Start heartbeat loop (every 30s): `{ "method": "nucleus.heartbeat", "params": { "primal": "neuralspring", "load": 0.15, "active_requests": 1 } }`
6. **NEW**: On shutdown (SIGTERM), send deregistration: `{ "method": "nucleus.deregister", "params": { "primal": "neuralspring" } }`

**Graceful degradation**: If biomeOS is not running, the primal operates
standalone (current behavior). Registration is best-effort.

### Step 4: Cross-Primal Request Forwarding

In Tower mode, neuralSpring may receive requests that require other primals
(e.g., `data.ncbi_search` needs NestGate). The primal should be able to
forward requests to sibling primals via socket discovery.

**Implementation**:

```rust
async fn forward_to_primal(primal: &str, method: &str, params: Value) -> Result<Value> {
    let socket = discover_primal_socket(primal)?;
    let client = UnixStream::connect(socket).await?;
    // ... send JSON-RPC, read response
}
```

**Discovery**: Use `biomeos-primal-sdk::PrimalDiscovery` to find sibling
primal sockets. Falls back to scanning `$XDG_RUNTIME_DIR/biomeos/*.sock`.

### Step 5: Systemd Integration (optional)

For persistent Tower mode on Eastgate, create a systemd user service:

```ini
[Unit]
Description=neuralSpring biomeOS Primal
After=network.target

[Service]
Type=simple
ExecStart=%h/Development/ecoPrimals/neuralSpring/target/release/neuralspring_primal
Environment=RUST_LOG=info
Environment=FAMILY_ID=eastgate
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

**Path**: `~/.config/systemd/user/neuralspring-primal.service`

---

## Capability Expansion Roadmap

### Tower Mode (Eastgate only — immediate)

```
neuralSpring primal
  ├── science.spectral_analysis      (existing)
  ├── science.anderson_localization  (existing)
  ├── science.hessian_eigen          (existing)
  ├── science.agent_coordination     (existing)
  ├── science.ipr                    (existing)
  ├── science.disorder_sweep         (existing)
  ├── science.training_trajectory    (existing)
  ├── science.evoformer_block        (NEW — Step 1)
  ├── science.structure_module       (NEW — Step 1)
  ├── science.folding_health         (NEW — Step 1)
  └── science.gpu_dispatch           (NEW — Step 1)
```

### Node Mode (+ ToadStool — next)

When ToadStool primal is also running on Eastgate, neuralSpring can
route heavy GPU workloads through ToadStool's streaming dispatch:

```
neuralSpring ──JSON-RPC──→ ToadStool (GPU compute)
     │                          │
     └── science.* request      └── WGSL shader execution
                                    (unidirectional streaming)
```

ToadStool socket: `$XDG_RUNTIME_DIR/biomeos/toadstool-{family}.sock`

The primal already has `Dispatcher` with GPU fallback. Node mode adds
the option of routing through ToadStool for fused pipeline execution
(46-78x over per-op dispatch).

### Nest Mode (+ NestGate — after providers built)

NestGate provides data for folding pipeline:

```
neuralSpring ──"data.pdb_fetch"──→ NestGate (PDB provider)
neuralSpring ──"data.msa_build"──→ NestGate (UniRef90 + MMseqs2)
```

This enables the full sovereign folding pipeline:
1. Query sequence → NestGate (MSA search) → MSA
2. MSA → neuralSpring (Evoformer) → pair representation
3. Pair → neuralSpring (Structure Module / Diffusion) → 3D coordinates
4. Store result → NestGate (ZFS provenance)

---

## Validation Plan

| Test | Checks | Description |
|------|--------|-------------|
| Socket bind + health | 2 | Start primal, query health, verify capabilities |
| Evoformer RPC | 3 | Send evoformer_block request, validate output shapes |
| Structure Module RPC | 3 | Send structure_module request, validate IPA scores |
| GPU health report | 2 | Verify GPU info in health response |
| Registration probe | 2 | Verify graceful degradation without biomeOS |
| Cross-primal forward | 3 | Forward to NestGate (mock), verify round-trip |
| Concurrent requests | 2 | 4 simultaneous requests within semaphore |
| Shutdown cleanup | 1 | Verify socket removed on SIGTERM |

**Target**: `validate_nucleus_tower.rs` — 18/18 checks

---

## Build Order

1. **Step 1**: Add folding capability handlers to `neuralspring_primal.rs`
   - `science.evoformer_block`, `science.structure_module`
   - `science.folding_health`, `science.gpu_dispatch`
   - Update `health` response with new capabilities list

2. **Step 2**: Add GPU-aware health check
   - Query `wgpu::Adapter` at startup
   - Add hardware info to health response
   - Add request counters and uptime

3. **Step 3**: Add biomeOS registration
   - Probe for orchestrator socket on startup
   - Send registration if found
   - Start heartbeat loop
   - Handle SIGTERM for deregistration

4. **Step 4**: Add cross-primal forwarding
   - Socket discovery via `biomeos-primal-sdk`
   - `forward_to_primal()` helper
   - Route `data.*` methods to NestGate

5. **Step 5**: Validator binary
   - `validate_nucleus_tower.rs` — 18/18 checks
   - Add to `validate_all.rs` and `Cargo.toml`

6. **Step 6** (optional): systemd user service
   - Service file + enable/start
   - Log rotation via journald

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| JSON serialization overhead for large tensors | Medium | Low | Binary protocol for large payloads (future) |
| Socket permission issues | Low | Low | biomeOS 5-tier fallback |
| biomeOS orchestrator not running | Expected | None | Graceful degradation (standalone mode) |
| GPU initialization delay on first request | Medium | Low | Warm up GPU at startup |
| Concurrent GPU contention | Low | Medium | Semaphore already limits to 4 |

---

*Tower mode on Eastgate can begin immediately. Steps 1-2 are purely within
neuralSpring. Steps 3-4 require biomeOS orchestrator to be running but
degrade gracefully. The folding pipeline (Steps 1-2) is the highest-value
addition — it exposes Session 90's validated Evoformer + Structure Module
to the entire primal ecosystem.*
