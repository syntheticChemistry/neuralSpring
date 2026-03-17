# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V114 — Session 163

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V114 S163** | `handoffs/NEURALSPRING_V114_S163_EDITION2024_HEALTH_PROBES_HANDOFF_MAR17_2026.md` | Mar 17, 2026 | Edition 2024, health probes (health.liveness, health.readiness), ipc_resilience (RetryPolicy, CircuitBreaker), 6 proptest invariants, MCP 14→16, deny.toml hardened. Supersedes V113 S162 |
| **V114 bC/tS** | `handoffs/NEURALSPRING_V114_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR17_2026.md` | Mar 17, 2026 | Edition 2024, health probes, IPC resilience patterns, proptest invariants, DispatchOutcome enrichment, evolution opps. Supersedes V113 bC/tS |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V113 + V95 + NestGate V1 + biomeOS V1 + Songbird V1).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
