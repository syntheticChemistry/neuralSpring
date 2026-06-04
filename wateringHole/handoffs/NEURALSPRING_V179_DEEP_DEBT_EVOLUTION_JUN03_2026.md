# neuralSpring V179 — Deep Debt Evolution Handoff

**Spring:** neuralSpring (southGate)
**Session:** S223 | **Date:** 2026-06-03
**Gate:** southGate (AMD Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04)
**Supersedes:** V178 (S222 — Wave 76 parity alignment)

---

## Mission

S223 deep debt evolution: production stubs evolved to real IPC forwarding,
capability-first tower discovery, error type modernization, visibility
tightening, dead code removal, documentation reconciliation.

## Test Results

| Metric | Value |
|--------|-------|
| `cargo test --workspace` | **930 passed, 0 failed, 0 warnings** |
| Lib tests (neural-spring) | 754 passed |
| Playground tests | 80 passed |
| Forge tests | 73 passed |
| Integration tests | 11 passed |
| Exp094 tests | 12 passed |
| Clippy (workspace, all-features) | **0 warnings** |
| Fmt | **clean** |
| Doc tests | **clean** |

## Changes — S223 Deep Debt Evolution

### Production Stub → Real IPC (handlers.rs)

4 JSON-RPC handlers evolved from local acknowledgment stubs to real IPC
forwarding with capability-based discovery and graceful fallback:

| Handler | Forwards to | Fallback |
|---------|-------------|----------|
| `handle_provenance` | biomeOS / rhizoCrypt | Local acknowledgment |
| `handle_compute_offload` | toadStool | Local dispatcher readiness |
| `handle_security_audit_log` | skunkBat | Local audit log |
| `handle_method_register` | biomeOS | Local acknowledgment |

New `try_discover_and_call()` helper for generic capability-based
primal discovery and IPC forwarding, used by all evolved handlers
plus existing inference handlers.

### Capability-First Tower Discovery (tower.rs)

BearDog and Songbird probing rewritten to prioritize capability-based
discovery (`crypto.btsp_handshake`, `security.audit_log`, `discovery.peers`,
`mesh.init`) with name-based socket lookup as fallback.

### Error Type Evolution

- `weight_loader.rs`: 5 functions migrated `Result<T, String>` →
  `crate::error::Result<T>` (typed errors with proper `From` impls)
- Error `source()` chains implemented for `FastqError`, `FastaError`,
  `VcfError`, `PipelineError`, `PushError`

### Visibility Tightening

18 internal modules narrowed from `pub` to `pub(crate)` in `lib.rs`:
`gpu`, `gpu_dispatch`, `gpu_ops`, `gpu_shader_validation`, `bench`,
`evolved`, `loss_landscape`, `nautilus_bridge`, `nucleus_pipeline`,
`provenance_dispatch`, `training_monitor`, `wdm_esn`, `wdm_sqw`,
`wdm_surrogate`, `wdm_transport`, `weight_spectral`, `certification`,
`rpc_service`.

3 functions in `provenance_dispatch.rs` narrowed `pub` → `pub(crate)`.

### Dead Code Removal

- `src/bin/neuralspring_primal/inference.rs`: deleted (129 lines of
  orphan stub handlers, never wired into `main.rs`)

### Clippy Cleanup

- `fasta.rs`: wildcard match → explicit `InvalidHeader` variant
- LTEE validators: `#[expect]` attributes for acceptable test patterns
  (`too_many_lines`, `expect_used`, `unwrap_used`)

### Documentation Reconciliation (S224 doc sweep)

Session stamps reconciled across 15+ documents from S221/V177 → S223/V179:
README.md, CONTEXT.md, CONTROL_EXPERIMENT_STATUS.md, EVOLUTION_READINESS.md,
DEPRECATION_MIGRATION.md, experiments/README.md, sporeprint/validation-summary.md,
docs/DEGRADATION_BEHAVIOR.md, docs/FOUNDATION_SEEDING.md,
docs/GUIDESTONE_PROPERTIES.md, graphs/neuralspring_deploy.toml,
6x experiments/results/*.json.

## Files Changed

| File | Change |
|------|--------|
| `src/bin/neuralspring_primal/inference.rs` | **Deleted** (dead code) |
| `src/bin/neuralspring_primal/handlers.rs` | 4 stubs → real IPC + `try_discover_and_call` |
| `src/bin/neuralspring_primal/tower.rs` | Capability-first discovery |
| `src/weight_loader.rs` | `Result<T, String>` → typed errors |
| `src/streaming/{fastq,fasta,vcf}.rs` | Error `source()` chains |
| `src/nucleus_pipeline/error.rs` | Error `source()` |
| `src/visualization/ipc_push.rs` | Error `source()` |
| `src/lib.rs` | 18 modules `pub` → `pub(crate)` |
| `src/provenance_dispatch.rs` | 3 fns `pub` → `pub(crate)` |
| `src/bin/validate_ltee_b{3,4}_*.rs` | `#[expect]` for test patterns |
| `CHANGELOG.md` | S223 entry |
| `README.md` | S223/V179 stamps |
| `CONTEXT.md` | S223 stamp |
| `CONTROL_EXPERIMENT_STATUS.md` | S223/V179 stamps |
| `EVOLUTION_READINESS.md` | S223/V179 stamps |
| `DEPRECATION_MIGRATION.md` | S223/V179 stamps |
| `experiments/README.md` | S223/V179 stamps |
| `docs/*.md` (3 files) | S223 stamps |
| `sporeprint/validation-summary.md` | S223/V179 stamps |
| `graphs/neuralspring_deploy.toml` | S223/V179 status |
| `experiments/results/*.json` (6 files) | S223 session |
| `docs/PRIMAL_GAPS.md` | S223/V179 (updated in S223) |
| `experiments/results/gap-status.json` | S223 (updated in S223) |

## Upstream Gaps for Primal Teams

| # | Gap | Owner | Priority | Notes |
|---|-----|-------|----------|-------|
| G-01 | Discovery triplication: `validation/composition.rs`, `primal/discovery.rs`, `playGround/discovery.rs` have overlapping socket resolution | neuralSpring | P2 | Consolidation target: shared library, thin async/sync wrappers |
| G-02 | Capability probe order divergence: `composition.rs` tries `capability.list` first, `primal/discovery.rs` tries `capabilities.list` first | neuralSpring / biomeOS | P2 | Behavioral inconsistency across primals |
| G-03 | barraCuda 18 surface gaps (Gap 11) | barraCuda | P3 | Tracked in PRIMAL_GAPS.md |

## ACK

neuralSpring on southGate: S223 deep debt evolution complete. 930 tests,
0 warnings, 0 failures. Production stubs evolved to real IPC. Capability-first
discovery. 15 modules tightened. Error types modernized. Documentation
reconciled. Ready for primalSpring audit.
