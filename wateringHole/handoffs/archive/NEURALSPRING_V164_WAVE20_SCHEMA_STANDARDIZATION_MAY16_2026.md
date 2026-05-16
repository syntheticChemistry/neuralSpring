# neuralSpring V164 — Wave 20 Schema Standardization

**From:** neuralSpring (S208)
**To:** primalSpring (coordination), all spring teams
**Date:** 2026-05-16
**Session:** S208 — Wave 20 schema standardization absorption

---

## Wave 20 Checklist Status

| Item | Status | Evidence |
|------|--------|----------|
| `capability.list` canonical envelope | **DONE** | Added `"count": ALL_CAPABILITIES.len()` to response. Shape is now `{ "capabilities": [...], "count": 35, "primal": "neural-spring" }` |
| `count` field | **DONE** | Single-line addition to `handle_capability_list` in `handlers.rs` |
| `primal.list` registry sync | **N/A** | biomeOS-served method — not spring-side. `primal.list` not yet in primalSpring's `capability_registry.toml` (documented in narrative docs only). neuralSpring's registry cross-test is substring-based, will auto-pass when upstream TOML updated. |
| Registry sync target | **452** | Doc comment in `registry_methods_in_primalspring_canonical` updated from stale "413" to "452". Test itself is substring-based, not count-based — no assertion to bump. |
| `nest.commit` signal dispatch | **CANDIDATE** | Documented in PRIMAL_GAPS.md Gap 15. Not a Wave 20 blocker. Relevant when training loop orchestration matures. |
| Schema validation scenario | **CANDIDATE** | Documented in PRIMAL_GAPS.md Gap 16. Optional — add when CI schema drift is a concern. |
| `--provenance-dir` | **CANDIDATE** | `neuralspring_unibin validate` supports `--format json`. `--provenance-dir` can be added when foundation workloads call the binary. |

---

## Changes Made

### `src/bin/neuralspring_primal/handlers.rs`

`handle_capability_list` response now includes `"count"`:

```rust
serde_json::json!({
    "primal": PRIMAL_NAME,
    "capabilities": ALL_CAPABILITIES,
    "count": ALL_CAPABILITIES.len(),
})
```

### `src/config.rs`

Registry cross-test doc comment updated: "413-method" → "452-method".

### `docs/PRIMAL_GAPS.md`

- Gap 15: `nest.commit` glacial candidate — training session finalization signal
- Gap 16: Schema validation scenario — optional

---

## Observations for Upstream

1. **`primal.list` not in primalSpring TOML**: The Wave 20 audit says "452 methods" with `primal.list` added, but `primalSpring/config/capability_registry.toml` `[primal]` section still lists only `primal.announce`, `primal.capabilities`, `primal.info`. The method is documented in narrative docs (CONTEXT.md, wateringHole/README.md) but hasn't landed in the canonical TOML. Springs doing count-based assertions would fail until this lands.

2. **neuralSpring's registry test is resilient**: Our `registry_methods_in_primalspring_canonical` test uses substring matching against the primalSpring TOML, not a fixed count assertion. It will automatically pick up `primal.list` when the upstream TOML is updated.

3. **`nest.commit` adoption path**: neuralSpring's weight persistence is currently single-shot (`nest.store`). The signal becomes relevant when we implement training loop orchestration or multi-checkpoint provenance. primalSpring's `s_nest_commit_live` scenario and `graphs/signals/nest_commit.toml` provide the reference implementation.

---

## Current State

| Metric | Value |
|--------|-------|
| Session | S208 |
| Workspace tests | 910 |
| Clippy errors | 0 |
| Capabilities | 35 |
| Validation scenarios | 7 |
| `capability.list` shape | Canonical (`capabilities` + `count` + `primal`) |
| Registry sync | 452 (doc), substring-based (test) |
| Handoff | V164 |
| Wave | 20 absorbed |
