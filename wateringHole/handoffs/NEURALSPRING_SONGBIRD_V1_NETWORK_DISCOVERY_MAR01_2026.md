# neuralSpring → Songbird Handoff V1 — Network Discovery + LAN Multi-Gate Vision

**Date**: March 1, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: Songbird team
**License**: AGPL-3.0-or-later
**Covers**: Session 99 — Socket discovery patterns, LAN multi-gate networking, inter-gate data transfer for science workloads

---

## Executive Summary

- neuralSpring connects to Songbird **indirectly** via biomeOS NUCLEUS (Tower layer: BearDog crypto + Songbird networking)
- neuralSpring's primal binary discovers other primals via **Unix domain sockets** in `$XDG_RUNTIME_DIR/biomeos/` — Songbird manages the socket namespace and TLS for remote connections
- **LAN vision**: 10 towers on 10GbE, Songbird handles inter-gate discovery (mDNS), connection management (TLS), and data routing between gates
- neuralSpring's metalForge **PCIe bypass cost model** translates to inter-gate networking: direct GPU→GPU transfer via RDMA is analogous to NPU→GPU PCIe bypass
- **What we need**: reliable socket discovery, multi-gate primal resolution, bandwidth-aware routing for large tensor transfers

---

## Part 1: Current Socket Discovery Pattern

neuralSpring discovers primals via a 3-step fallback:

```rust
fn discover_primal_socket(primal_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();  // $XDG_RUNTIME_DIR/biomeos/
    let family_id = get_family_id();

    // 1. Family-scoped socket
    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() { return Ok(with_family); }

    // 2. Global socket
    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() { return Ok(without_family); }

    // 3. Glob scan for any matching socket
    for entry in read_dir(&socket_dir) {
        if filename starts with primal_name { return Ok(entry.path()); }
    }

    bail!("No socket found for {primal_name}")
}
```

This works for **single-gate local** deployment. For multi-gate LAN:
- Step 1/2 find local primals
- Step 3 needs extension: if not found locally, query Songbird for remote primals

---

## Part 2: What Songbird Needs for neuralSpring

### Multi-Gate Primal Discovery

When neuralSpring on Eastgate needs NestGate running on Westgate:

```
neuralSpring (Eastgate)
    → discover_primal_socket("nestgate")
    → not found locally
    → query Songbird: "where is nestgate?"
    → Songbird: "nestgate is on Westgate at 10.0.0.5:9443"
    → connect via Songbird-managed TLS tunnel
```

**What Songbird needs**:
1. **Primal registry** — which gate runs which primals, with capabilities
2. **mDNS gate discovery** — find other gates on the 10GbE LAN
3. **TLS tunnel management** — secure primal-to-primal communication across gates
4. **Socket proxy** — remote primal looks like a local socket to the caller

### Bandwidth-Aware Routing

neuralSpring's metalForge models inter-device transfer costs:

| Transfer | Cost Model | Equivalent Gate Transfer |
|----------|-----------|------------------------|
| CPU → GPU (same gate) | ~10µs + size/bandwidth | Same gate, different substrate |
| GPU → NPU (PCIe bypass) | ~5µs + size/bandwidth | Same gate, PCIe direct |
| CPU → CPU (different gate) | Network RTT + size/bandwidth | Cross-gate, 10GbE |
| GPU → GPU (different gate) | Network RTT + size/bandwidth | Cross-gate, RDMA if available |

For large tensor transfers (MSA results from Strandgate → Northgate for GPU inference), bandwidth matters:
- 10GbE: ~1.2 GB/s theoretical → ~100MB MSA output transfers in ~80ms
- PCIe 4.0 x16: ~25 GB/s → same transfer in ~4ms

Songbird should expose bandwidth metrics so biomeOS Plasmodium can make routing decisions.

---

## Part 3: LAN Topology

```
┌─────────────┐     10GbE      ┌──────────────┐
│  Eastgate   │◄──────────────►│  Strandgate  │
│  (dev+NPU)  │                │  (EPYC 64c)  │
│  RTX 4070   │     10GbE      │  256GB ECC   │
│  AKD1000    │◄──────┐       │  20TB+       │
└─────────────┘       │       └──────────────┘
                      │
                ┌─────┴─────┐
                │  Switch   │
                │  (10GbE)  │
                └─────┬─────┘
                      │
┌─────────────┐       │       ┌──────────────┐
│  Northgate  │◄──────┘       │   Westgate   │
│  (flagship) │     10GbE     │ (cold store) │
│  RTX 5090   │◄─────────────►│  76TB ZFS    │
│  192GB DDR5 │               │              │
└─────────────┘               └──────────────┘
```

**Songbird's role**: Each gate runs a Songbird instance (Tower atomic). Songbird instances discover each other via mDNS on the 10GbE subnet, establish TLS channels, and maintain a primal registry.

---

## Part 4: Science Workload Patterns

Typical neuralSpring multi-gate pipeline:

**coralForge structure prediction**:
1. NestGate (Westgate): fetch PDB sequence → stream to Strandgate
2. CPU heavy (Strandgate): MSA generation (JackHMMer, 64 EPYC cores)
3. GPU heavy (Northgate): Evoformer + IPA + diffusion (5090, 32GB VRAM)
4. Storage (Westgate): archive predicted structure + confidence scores

**nS-01 batch weight spectral**:
1. Data (any gate): load pretrained model weights
2. GPU heavy (Northgate): `BatchedEighGpu` for large weight matrices
3. Analysis (Eastgate): spectral comparison + provenance tracking

Each step produces data that flows to the next gate. Songbird manages the connections; biomeOS Plasmodium orchestrates the pipeline.

---

## Part 5: Lessons from metalForge for Songbird

1. **Dispatch overhead matters** — metalForge found ~186µs GPU dispatch overhead. For cross-gate routing, Songbird's connection setup + TLS overhead should be characterized similarly. If it's >1ms, batch requests.

2. **Crossover points exist** — metalForge found CPU→GPU crossover at ~1946µs. Similarly, there's a workload-size crossover below which local computation beats remote dispatch. Songbird (or Plasmodium) should know these thresholds.

3. **Bit-identical across substrates** — metalForge proved multi-GPU bit-identical (384/384). Cross-gate routing should preserve this guarantee.

4. **Health checks prevent silent failure** — metalForge's substrate `probe()` detects unhealthy devices. Songbird's gate health monitoring should expose the same.

---

## Part 6: Priority Actions for Songbird

| Priority | Action | Impact |
|----------|--------|--------|
| **P1** | Reliable local socket namespace management | Foundation for all primal communication |
| **P2** | mDNS gate discovery on 10GbE subnet | Enables multi-gate awareness |
| **P3** | TLS tunnel for cross-gate primal communication | Secure inter-gate data flow |
| **P4** | Primal registry (which gate has which capabilities) | Enables remote primal discovery |
| **P5** | Bandwidth metrics exposure | Enables routing optimization |
| **P6** | Socket proxy (remote primal appears as local socket) | Transparent multi-gate for callers |

---

## Handoff Lineage

| Version | Session | Focus |
|---------|---------|-------|
| **V1** | **S99** | **Socket discovery, LAN multi-gate networking, bandwidth-aware routing, science workload patterns** |

---

*neuralSpring → Songbird V1 handoff — March 1, 2026. Session 99. Socket discovery operational for single-gate. LAN requires mDNS gate discovery, TLS tunnels, primal registry, bandwidth-aware routing. 10 towers on 10GbE, 4 Akida NPUs, 176GB GPU VRAM aggregate.*
