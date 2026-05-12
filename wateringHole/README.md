# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V154 — Session S202c (Tier 2 Wiring + Deep Debt Audit)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V154** | `handoffs/NEURALSPRING_V154_TIER2_DEEP_DEBT_HANDOFF_MAY12_2026.md` | May 12, 2026 | Tier 2 convergence: `toadstool.validate` + `toadstool.list_workloads` wired. Deep debt audit all-clear (zero unsafe, zero mocks, zero panics, zero TODO). B1 NUCLEUS workload TOML. Foundation Thread 5 ML_SURROGATES wired. 892 IPC-first workspace tests. 36 capabilities. |

### V153 — Sessions S202–S202b (River Delta Downstream Seeding + Audit Response)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V153** | `handoffs/NEURALSPRING_V153_RIVER_DELTA_DOWNSTREAM_SEEDING_HANDOFF_MAY12_2026.md` | May 12, 2026 | River Delta downstream seeding: `--format json` all validation binaries, `CapabilityRouter` IPC, foundation Thread 5, NestGate IPC, Gap 11 drift flagged. 888 IPC-first workspace tests. |

### V152 (archived)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V152** | `handoffs/archive/NEURALSPRING_V152_TIER4_IPC_FIRST_HANDOFF_MAY11_2026.md` | May 11, 2026 | Tier 4 IPC-first, V152 handoff. |
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
