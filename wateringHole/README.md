# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V159 — Session S205c (Primal Evolution + Composition Patterns Upstream)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V159** | `handoffs/NEURALSPRING_V159_PRIMAL_EVOLUTION_UPSTREAM_HANDOFF_MAY13_2026.md` | May 13, 2026 | Primal usage map (7 IPC modules, 37 caps), composition patterns (CapabilityRouter, NestGate weights, Squirrel pipeline, agent-driven), evolution opportunities for upstream primal teams, NUCLEUS deployment patterns, learnings. |

### V158 — Session S205b (Deep Debt Re-Audit + Evolution Sprint)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V158** | `handoffs/NEURALSPRING_V158_DEEP_DEBT_REAUDIT_HANDOFF_MAY13_2026.md` | May 13, 2026 | Full deep debt re-audit: 0 across all 7 categories. Answers all 5 audit questions. 20 bench scripts, 397/397 baselines, 27/27 papers, 15 domains. 910 tests. |

### V157 — Session S205 (Niche Convergence → Atomic Deployment)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V157** | `handoffs/NEURALSPRING_V157_NICHE_CONVERGENCE_ATOMIC_DEPLOYMENT_HANDOFF_MAY13_2026.md` | May 13, 2026 | NestGate weight persistence wired (store/load safetensors via BLAKE3). Squirrel inference_models + has_squirrel. Wire hygiene verified. plasmidBin ready. 910 tests. |

### V156 — Session S204b (Deep Debt Resolution + Evolution Sprint)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V156** | `handoffs/NEURALSPRING_V156_DEEP_DEBT_RESOLUTION_HANDOFF_MAY13_2026.md` | May 13, 2026 | Full deep debt audit: 0 TODO/unsafe/mocks/panics/allow/files>800L. `relu` const fn, LTEE B1 mul_add, merge_tracks DRY, allow(deprecated) removed, dev-deps workspace-inherited. 15/15 CPU baselines, 27/27 papers. 907 tests. |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V155 + NestGate V1 + biomeOS V1 + Songbird V1 + barraCuda evolution requests).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
