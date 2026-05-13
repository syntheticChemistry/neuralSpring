# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V155 — Sessions S203–S203b (Tier 2 Convergence + Deep Debt)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V155** | `handoffs/NEURALSPRING_V155_TIER2_CONVERGENCE_HANDOFF_MAY13_2026.md` | May 13, 2026 | Tier 2 complete: `barracuda.precision.route` wired (v0.4.0). 7 stale lint expects pruned. LTEE B1 tolerances.toml. 907 IPC-first tests. 37 capabilities. 20 CAPABILITY_HINTS. |

### V154 — Session S202c (Tier 2 Wiring + Deep Debt Audit)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V154** | `handoffs/NEURALSPRING_V154_TIER2_DEEP_DEBT_HANDOFF_MAY12_2026.md` | May 12, 2026 | `toadstool.validate` + `toadstool.list_workloads` wired. Deep debt audit all-clear. B1 NUCLEUS workload. 892 tests. 36 capabilities. |

### V153 — Sessions S202–S202b (River Delta Downstream Seeding)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V153** | `handoffs/NEURALSPRING_V153_RIVER_DELTA_DOWNSTREAM_SEEDING_HANDOFF_MAY12_2026.md` | May 12, 2026 | `--format json`, CapabilityRouter, Thread 5, NestGate. 888 tests. |

### V152 (archived)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V152** | `handoffs/archive/NEURALSPRING_V152_TIER4_IPC_FIRST_HANDOFF_MAY11_2026.md` | May 11, 2026 | Tier 4 IPC-first. |
| **V152** (companion) | `handoffs/archive/NEURALSPRING_V152_PRIMAL_EVOLUTION_UPSTREAM_HANDOFF_MAY11_2026.md` | May 11, 2026 | Upstream primal handoff. |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V151 + NestGate V1 + biomeOS V1 + Songbird V1 + barraCuda evolution requests).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
