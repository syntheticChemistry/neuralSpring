# wateringHole — neuralSpring Cross-Project Handoffs

Formal handoff documents between neuralSpring and ecoPrimals primals.
Following the wetSpring/hotSpring pattern: unidirectional Spring → primal flow.

## Active Handoffs

### V167 — Session S211 (GPU Parity + Compute Dispatch Evolution)

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V167** | `handoffs/NEURALSPRING_V167_GPU_PARITY_EVOLUTION_MAY17_2026.md` | May 17, 2026 | 6/6 GPU dispatch (4 stages CpuOnly→GpuPreferred). PCIe P2P in Dispatcher. Typed toadStool workload submission. s_gpu_parity scenario 10/10. node.compute in live executor. |

### Central wateringHole Copies

Active handoffs also published to `ecoPrimals/wateringHole/handoffs/` for cross-project visibility.

## Archive

Superseded handoffs: `handoffs/archive/` (V1–V166 + NestGate V1 + biomeOS V1 + Songbird V1 + barraCuda evolution requests).

## Conventions

- Naming: `NEURALSPRING_V{NN}_{TOPIC}_HANDOFF_{DATE}.md`
- Direction: neuralSpring → primal (never the reverse)
- On supersede: move to `handoffs/archive/`
- Max file size: 1000 lines
- License: AGPL-3.0-or-later (all handoffs)
