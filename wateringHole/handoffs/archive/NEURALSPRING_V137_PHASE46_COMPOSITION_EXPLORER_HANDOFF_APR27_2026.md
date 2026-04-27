# neuralSpring V137 — Phase 46 Composition Explorer Handoff

**Date**: April 27, 2026
**Session**: S186 (V137)
**From**: neuralSpring
**For**: primalSpring, Squirrel, upstream primal teams
**primalSpring**: v0.9.17+ (Phase 45c BTSP default + Phase 46 composition library)
**guideStone**: v0.3.0 / standard v1.2.0 / Level 3

---

## Summary

neuralSpring has absorbed the Phase 46 composition library and implemented its
assigned lane: **Agent-Driven Composition + AI Feedback Loops**. This handoff
documents the agentic composition patterns discovered, inference integration
findings, AI provenance schema, and gaps for upstream teams.

### What was done

1. **Copied composition tools** from `primalSpring/tools/` into `neuralSpring/tools/`:
   - `nucleus_composition_lib.sh` (41 functions: discovery, JSON-RPC, petalTongue, DAG, ledger, braids, sensors)
   - `composition_template.sh` (domain starter with hook pattern)
   - `composition_nucleus.sh` (parameterized NUCLEUS launcher)

2. **Created `tools/neural_composition.sh`** — domain composition implementing:
   - Squirrel-mediated `inference.complete` / `inference.embed` via IPC
   - DAG branching for AI decisions with structured metadata
   - Braid provenance audit trail (`application/x-neuralspring-agent`)
   - Closed-loop feedback (act → observe → adjust via `domain_on_tick` + `check_proprioception`)
   - petalTongue dashboard showing agent state, inference count, DAG depth, last result
   - Interactive triggers: I=infer, E=embed, R=reason, A=auto-toggle, Q=quit
   - Auto-reason mode: periodic autonomous inference at configurable interval

3. **Updated Rust-side awareness** — guidestone doc comment references Phase 46 tooling;
   PRIMAL_GAPS Gap 14 documents all composition findings

4. **Phase 45c BTSP default** — auto-absorbed via `primalspring` path dependency.
   BTSP now mandatory for all 13 capabilities; cleartext connections FAIL

---

## Agentic Composition Patterns Discovered

### Pattern 1: Squirrel-Mediated Inference via Composition Library

```bash
socket=$(cap_socket "ai")
result=$(send_rpc "$socket" "inference.complete" \
    '{"prompt":"...","model":"default","max_tokens":64}')
```

- Squirrel must be in `REQUIRED_CAPS` or `OPTIONAL_CAPS` for discovery
- `cap_socket "ai"` resolves family-aware sockets (`ai-{family}.sock`)
- Falls back gracefully when Squirrel unavailable (degraded mode)
- BTSP handshake required (Phase 45c) — `is_skip_error` handles failures

### Pattern 2: DAG Branching for AI Decisions

```bash
dag_append_event "$COMPOSITION_NAME" "inference" "$agent_state" \
    "[{\"key\":\"prompt\",\"value\":\"...\"},
     {\"key\":\"result\",\"value\":\"$result\"},
     {\"key\":\"model\",\"value\":\"default\"}]" "agent" "0"
```

- Each inference call becomes a DAG event with causal ordering
- Multi-step reasoning creates branching DAG paths
- DAG depth tracks agent decision history
- Requires `dag` capability (rhizoCrypt / Nest atomic)

### Pattern 3: Braid Provenance Audit Trail

```bash
braid_record "inference" "application/x-neuralspring-agent" "$agent_state" \
    "{\"action\":\"inference\",\"detail\":\"$result\",\"tick\":$TICK_COUNT}" \
    "agent" "0"
```

- Content-type `application/x-neuralspring-agent` tags all agent decisions
- Braids carry full payload (prompt, result, confidence, tick)
- DAG provides structure; braids provide searchable content
- Together they answer: what did the agent decide, when, why, with what input?

### Pattern 4: Closed-Loop Feedback (Act → Observe → Adjust)

```bash
domain_on_tick() {
    TICK_COUNT=$((TICK_COUNT + 1))
    check_proprioception                   # observe
    if (( AUTO_REASON_INTERVAL > 0 )); then
        do_reason                           # act + adjust
    fi
}
```

- `check_proprioception` reads sensor state for the "observe" phase
- Auto-reason at configurable interval enables autonomous agent behavior
- Reasoning step: inference → parse decision → execute (infer/embed/wait)
- Sensor streams provide real-time feedback for adjustment

---

## Inference Integration Findings

| Method | Status | Notes |
|--------|--------|-------|
| `inference.complete` | Works | JSON-RPC via Squirrel. Params: `prompt`, `model`, `max_tokens`. Response: `text` or `completion` field |
| `inference.embed` | Works | JSON-RPC via Squirrel. Params: `text`, `model`. Response: `embedding` or `embeddings` field |
| `inference.models` | Untested | Not exercised by composition script; should enumerate available models |
| `inference.register_provider` | Unknown | neuralSpring cannot self-register as inference backend; method existence unverified |

---

## AI Provenance Schema

Agent decisions are recorded with dual structures:

**DAG event** (causal ordering):
```json
{
  "session": "neuralspring",
  "action": "inference_complete",
  "state": "inferring",
  "metadata": [
    {"key": "prompt", "value": "..."},
    {"key": "result", "value": "..."},
    {"key": "model", "value": "default"},
    {"key": "inference_count", "value": "5"},
    {"key": "embed_count", "value": "2"}
  ],
  "input_type": "agent",
  "hover": "0"
}
```

**Braid record** (searchable payload):
```json
{
  "strand_type": "inference_complete",
  "content_type": "application/x-neuralspring-agent",
  "label": "ready",
  "payload": {
    "action": "inference_complete",
    "detail": "model response text...",
    "tick": 42
  },
  "input_type": "agent",
  "hover": "0"
}
```

---

## PRIMAL_GAPS Entries (Gap 14)

| Finding | Impact | Hand back to |
|---------|--------|--------------|
| Squirrel not in default `PRIMAL_LIST` | Must manually add for AI compositions | primalSpring |
| `inference.register_provider` wire unknown | Cannot self-register as inference backend | Squirrel |
| No `inference.models` via composition lib | Cannot enumerate available models | Squirrel |
| DAG session requires `dag` capability | Full Nest atomic needed for provenance | primalSpring |
| Braid query latency uncharacterized | Audit trail retrieval may bottleneck | loamSpine |
| Sensor stream polling interval fixed | No adaptive polling for high-frequency loops | primalSpring |

### Recommendation

`composition_nucleus.sh` should support `EXTRA_PRIMALS` env var so domain
compositions can request Squirrel without forking the launcher:

```bash
EXTRA_PRIMALS="squirrel" COMPOSITION_NAME=neuralspring ./composition_nucleus.sh start
```

---

## Primal Use & Evolution Patterns

### Dependency Profile (S186)

| Primal | Usage | Integration |
|--------|-------|-------------|
| barraCuda | 250+ import files, 32 JSON-RPC methods | Library (Tier 1/2), IPC (Tier 3+) |
| toadStool | Hardware discovery, compute dispatch | IPC via `compute.dispatch` |
| coralReef | Shader compilation | IPC via `shader.compile.wgsl` |
| BearDog | Crypto (BLAKE3, Ed25519) | IPC via `crypto.hash`, BTSP |
| Songbird | Discovery mesh | IPC via capability resolution |
| NestGate | Weight storage (future) | Not yet wired |
| Squirrel | AI inference | IPC via `inference.complete`/`embed` + composition shell |
| petalTongue | Visualization + sensor streams | IPC via scene graph + sensors |
| rhizoCrypt | DAG provenance | IPC via `dag.*` (composition shell) |
| loamSpine | Braid audit trail | IPC via `braid.*` (composition shell) |

### NUCLEUS Composition Pattern

neuralSpring validates against a proto-nucleate (pure primal NUCLEUS) externally.
The composition script deploys NUCLEUS from plasmidBin and exercises the full
stack. The guideStone binary validates domain science parity against Python baselines.

### neuralAPI / biomeOS Deployment

neuralSpring deploys as a NUCLEUS node providing `inference.*` capabilities.
Squirrel discovers neuralSpring as a provider. biomeOS orchestrates the deploy
graph (`graphs/neuralspring_deploy.toml`). The cell graph
(`plasmidBin/cells/neuralspring_cell.toml`) defines the full cellular deployment.

---

## Ecosystem Learnings

1. **Composition library is production-ready**: 41 functions cover the full NUCLEUS
   interaction surface. Domain scripts need only implement 5 hooks.

2. **Squirrel integration requires explicit opt-in**: Not in default `PRIMAL_LIST`.
   AI compositions must request it. Consider making it optional-by-default.

3. **DAG + braid duality is powerful for AI audit**: DAG provides causal structure
   (decision tree), braids provide content-addressed payload (what was decided).
   Together they form a complete provenance chain for AI decisions.

4. **BTSP is transparent**: Phase 45c BTSP default required zero code changes in
   neuralSpring — the `primalspring` crate path dependency auto-absorbs it.

5. **Closed-loop feedback works**: The act→observe→adjust cycle through sensor
   streams is composable and extensible. Future work: adaptive polling intervals.

6. **petalTongue scene graph is adequate for dashboards**: The text-node-based
   scene graph handles agent status display well. Interactive buttons via
   hit-testing work for basic UI.

7. **`composition_nucleus.sh` launcher is generic**: Works for any domain with
   only `COMPOSITION_NAME` customization. Adding primals requires editing
   `PRIMAL_LIST` or the recommended `EXTRA_PRIMALS` pattern.

8. **Family-aware discovery + BTSP work end-to-end**: `FAMILY_ID` env propagates
   correctly through all composition library functions.

9. **AI provenance schema is reusable**: The `application/x-{spring}-agent`
   content-type pattern + structured DAG metadata is applicable to any spring
   that integrates Squirrel.

---

## For Upstream Primal Teams

### For Squirrel

- **`inference.register_provider`**: Does this method exist? neuralSpring needs
  it to self-register as an inference backend in NUCLEUS compositions
- **`inference.models`**: Useful for pre-flight model enumeration before inference

### For primalSpring

- **`EXTRA_PRIMALS` env var in `composition_nucleus.sh`**: Allow domain compositions
  to request additional primals (especially Squirrel) without forking the launcher
- **Squirrel in optional PRIMAL_LIST**: Consider adding Squirrel to an "extended"
  primal list for compositions that need AI capabilities
- **Adaptive sensor polling**: Current fixed interval may not suit high-frequency
  agent loops; consider parameterizing poll interval in the library

### For loamSpine

- **Braid query performance**: Agent compositions generate braids at high frequency;
  characterize retrieval latency for audit trail queries

### For rhizoCrypt

- **DAG session for agent compositions**: Agent compositions create DAG sessions
  with potentially deep branching; ensure performance at depth

---

## Files Changed (S186)

| File | Change |
|------|--------|
| `tools/nucleus_composition_lib.sh` | NEW — copied from primalSpring |
| `tools/composition_template.sh` | NEW — copied from primalSpring |
| `tools/composition_nucleus.sh` | NEW — copied from primalSpring |
| `tools/neural_composition.sh` | NEW — agent-driven composition script |
| `src/bin/neuralspring_guidestone.rs` | Doc comment updated (Phase 46) |
| `docs/PRIMAL_GAPS.md` | Gap 14 (composition findings) added |
| `CHANGELOG.md` | S186 entry |
| `README.md` | Session header, tools/ tree, handoff refs |
| `EVOLUTION_READINESS.md` | S186 composition explorer status |
| `docs/GUIDESTONE_PROPERTIES.md` | Phase 45c BTSP note |
| `whitePaper/README.md` | Session update |
| `whitePaper/baseCamp/README.md` | Session update |
| `experiments/README.md` | Exp 133 (Phase 46 composition) |
| `specs/README.md` | Session update |
| `graphs/neuralspring_deploy.toml` | V137/S186 |
| `src/bin/validate_composition_evolution.rs` | Doc ref V137/S186 |
| `wateringHole/handoffs/archive/V136*.md` | Archived |

---

*V137 / S186 — neuralSpring Phase 46 Composition Explorer — April 27, 2026*
