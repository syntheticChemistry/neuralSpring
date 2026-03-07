# neuralSpring → biomeOS Handoff V1 — NUCLEUS Local Integration + LAN Scaling

**Date**: March 1, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: biomeOS/NUCLEUS team
**License**: AGPL-3.0-or-later
**Covers**: Session 99 — NUCLEUS local deployment, primal registration, metalForge↔NUCLEUS alignment, LAN multi-gate roadmap

---

## Executive Summary

- neuralSpring primal binary **already integrates with biomeOS** — registers capabilities via `lifecycle.register` + `capability.register`, sends 30s heartbeats, forwards `data.*` to NestGate
- metalForge substrate model (substrate/inventory/dispatch) **aligns with NUCLEUS Tower/Node/Nest** — validated at **47/47 mixed-hardware** + **41/41 metalForge NUCLEUS** checks
- **11 science capabilities** registered: spectral analysis, Anderson localization, coralForge folding, GPU dispatch
- **LAN roadmap**: 10 towers on 10GbE with biomeOS Plasmodium coordinating MSA (Strandgate EPYC) → GPU inference (Northgate 5090) → cold storage (Westgate 76TB)
- **What we need**: reliable `biomeos nucleus start --mode full` with all 5 primals, capability-based workload routing across gates

---

## Part 1: Current neuralSpring ↔ biomeOS Integration

### Primal Binary

`src/bin/neuralspring_primal/main.rs` — JSON-RPC 2.0 server with biomeOS integration:

**Startup sequence**:
1. Parse args (family-id, socket path)
2. Bind Unix socket at `$XDG_RUNTIME_DIR/biomeos/neuralspring-{family_id}.sock`
3. Probe for biomeOS orchestrator at `biomeOS.sock`
4. If found: `lifecycle.register` + `capability.register` for all 11 capabilities
5. Start 30s heartbeat loop via `lifecycle.status`
6. Serve JSON-RPC requests

**Capabilities registered**:

| Capability | Domain | What It Does |
|------------|--------|-------------|
| `science.ipr` | nS-01 (Weight Hamiltonians) | Inverse participation ratio of weight matrices |
| `science.disorder_sweep` | nS-01, nS-05 | Anderson disorder parameter sweep |
| `science.spectral_analysis` | nS-01 | Full weight spectral analysis (ESD, LSR, MP) |
| `science.anderson_localization` | nS-01, nS-05 | Anderson localization diagnostics |
| `science.hessian_eigen` | nS-03 (Loss Landscapes) | Hessian eigendecomposition for saddle detection |
| `science.agent_coordination` | nS-05 (Multi-Agent QS) | Game theory + QS coordination |
| `science.training_trajectory` | nS-01 | Training checkpoint spectral tracking |
| `science.evoformer_block` | coralForge (nF-01/02) | Evoformer attention + triangle multiply |
| `science.structure_module` | coralForge (nF-01) | IPA + backbone generation |
| `science.folding_health` | coralForge | Pipeline health check |
| `science.gpu_dispatch` | All | Route arbitrary GPU Dispatcher operations |

### Cross-Primal Forwarding

neuralSpring forwards `data.*` calls to NestGate and `primal.forward` to any named primal:

```rust
fn discover_primal_socket(primal_name: &str) -> Result<PathBuf> {
    // 1. $XDG_RUNTIME_DIR/biomeos/{primal}-{family_id}.sock
    // 2. $XDG_RUNTIME_DIR/biomeos/{primal}.sock
    // 3. Scan socket dir for {primal}*.sock
}
```

---

## Part 2: metalForge ↔ NUCLEUS Alignment

neuralSpring's metalForge (`metalForge/forge/`) mirrors NUCLEUS atomic patterns:

| metalForge Concept | NUCLEUS Equivalent | Implementation |
|-------------------|-------------------|----------------|
| `Substrate` (GPU/CPU/NPU + capabilities) | Tower hardware inventory | `forge/src/substrate.rs` |
| `discover()` → `Vec<Substrate>` | Tower hardware discovery | `forge/src/inventory.rs` |
| `dispatch(workload, substrate)` | Node compute routing | `forge/src/dispatch.rs` |
| `Provenance` (lineage tracking) | Nest metadata storage | `forge/src/workloads.rs` |
| PCIe bridge cost model | Plasmodium inter-gate routing | `forge/src/pcie_bridge.rs` |

**Validation coverage**:
- `validate_mixed_hardware_dispatch`: **47/47 PASS** — substrate routing, PCIe bridge, NUCLEUS atomics
- `validate_metalforge_wdm_coral`: **41/41 PASS** — Tower discovery, Node GPU dispatch, Nest provenance

The metalForge substrate model is a **local single-gate** version of what NUCLEUS Plasmodium provides across gates. The bridge: metalForge dispatches within a gate, Plasmodium dispatches across gates.

---

## Part 3: Binary Discovery for NUCLEUS

### Current State

`biomeos nucleus start` discovers primal binaries via (in order):
1. `livespore-usb/{ARCH}/primals/{primal}`
2. `livespore-usb/primals/{primal}`
3. `plasmidBin/{primal}` (root of plasmidBin)
4. `plasmidBin/optimized/{ARCH}/{primal}`
5. `target/release/{primal}`
6. `$PATH`

### neuralSpring's Binary

neuralSpring's primal binary is `neuralspring_primal` (built via `cargo build --release --features primal --bin neuralspring_primal` in neuralSpring repo). It's not a standard phase1 primal — it's a **science primal** that registers with NUCLEUS as a capability provider.

**Request**: biomeOS should support discovering science primals alongside infrastructure primals. Either:
- Add neuralSpring's `target/release/neuralspring_primal` to the discovery path
- Allow `nucleus start` to accept additional primal paths via config
- Use `$PATH` (simplest — symlink `neuralspring_primal` into a shared bin dir)

---

## Part 4: LAN Multi-Gate Roadmap

### Hardware Inventory (from gen3/about/HARDWARE.md)

| Gate | Role | Key Resource | NUCLEUS Mode |
|------|------|-------------|-------------|
| **Eastgate** | Primary dev + neuromorphic | RTX 4070, AKD1000 NPU | Node (Tower + ToadStool + neuralSpring) |
| **Strandgate** | Bioinformatics | Dual EPYC 64c, 256GB ECC, 20TB+ | Node (heavy CPU) |
| **Northgate** | Flagship GPU | RTX 5090 (32GB VRAM), 192GB DDR5 | Node (heavy GPU) |
| **Westgate** | Cold storage | 76TB ZFS | Nest (Tower + NestGate) |
| **biomeGate** | Multi-GPU | 3090 + Titan V, 256GB | Node (multi-GPU) |

### Workload Routing Vision

```
neuralSpring experiment request
        │
        ▼
  biomeOS Plasmodium (capability router)
        │
        ├── "compute.msa" → Strandgate (64 EPYC cores, 256GB)
        ├── "compute.gpu.matmul" → Northgate (5090, 32GB VRAM)
        ├── "compute.gpu.eigendecomp" → Eastgate (4070) or Northgate (5090)
        ├── "storage.archive" → Westgate (76TB ZFS)
        ├── "compute.npu.classify" → Eastgate (AKD1000)
        └── "data.ncbi_fetch" → any Nest with NestGate
```

**What biomeOS needs to enable this**:
1. **Plasmodium workload routing** based on capability + resource requirements
2. **Inter-gate data transfer** — weights uploaded to Northgate GPU, results back to Eastgate
3. **Job orchestration** — multi-step pipelines (MSA → structure prediction → confidence scoring) across gates
4. **Health-aware routing** — if Northgate is busy, route GPU work to biomeGate Titan V

---

## Part 5: What neuralSpring Validates for biomeOS

### Patterns Proven in metalForge

| Pattern | Checks | What It Proves |
|---------|--------|----------------|
| Hardware discovery (GPU, CPU, NPU) | 2 | `discover()` finds all substrates |
| Capability-based dispatch | 8 | Workloads route to correct substrate |
| Provenance tracking | 4 | Metadata lineage through pipeline stages |
| Mixed routing scenarios | 10 | Small/large workloads, realtime folding, heterogeneous pipeline |
| PCIe bypass costs | 5 | Inter-device transfer cost modeling |
| NUCLEUS coordination | 12 | Tower+Node+Nest atomic patterns |

### Key Findings for biomeOS

1. **GPU dispatch overhead is ~186µs per submit** — motivates batching everything into single-submit pipelines. biomeOS job orchestration should batch related operations before dispatching to a gate.

2. **CPU→GPU crossover at ~1946µs** — small workloads are faster on CPU. Plasmodium should have a size threshold for routing to GPU gates.

3. **PCIe bypass (NPU→GPU direct) saves ~2× vs CPU roundtrip** — relevant for Eastgate's AKD1000 feeding the RTX 4070. biomeOS should model inter-device transfer costs within a gate.

4. **Multi-GPU bit-identical** — RTX 4070 and TITAN V produce identical results (384/384). biomeOS can safely load-balance across GPU gates.

---

## Part 6: Priority Actions for biomeOS

| Priority | Action | Impact |
|----------|--------|--------|
| **P1** | Verify `nucleus start --mode tower` works on Eastgate (BearDog + Songbird) | Foundation for all integration |
| **P2** | Add neuralSpring primal to NUCLEUS discovery (science primal pattern) | Enables capability registration |
| **P3** | Verify `nucleus start --mode full` with all 5 primals + neuralSpring | Full local atomic |
| **P4** | Plasmodium multi-gate with capability-based routing | Enables LAN workload distribution |
| **P5** | Inter-gate data transfer for pipeline stages | Enables MSA→inference→storage pipelines |
| **P6** | Job orchestration for multi-step experiments | Enables automated experiment pipelines |

---

## Handoff Lineage

| Version | Session | Focus |
|---------|---------|-------|
| **V1** | **S99** | **NUCLEUS local integration, metalForge↔NUCLEUS alignment, LAN multi-gate roadmap** |

---

*neuralSpring → biomeOS V1 handoff — March 1, 2026. Session 99. 11 science capabilities registered, 88/88 metalForge NUCLEUS checks, primal binary operational. Priority: verify NUCLEUS local, enable science primal discovery, build toward Plasmodium multi-gate routing.*
