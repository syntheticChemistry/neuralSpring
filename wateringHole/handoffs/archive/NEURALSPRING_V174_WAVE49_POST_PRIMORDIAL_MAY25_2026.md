# neuralSpring V174 — Wave 49 Post-Primordial Deployment

**Date:** 2026-05-25
**Session:** S218
**Gate:** southGate (Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04)
**Upstream:** primalSpring Wave 49 — Post-Primordial Deployment + Covalent Mesh

---

## Summary

Absorbed primalSpring Wave 49 audit. Cut all primordial binary-sourcing
patterns from `composition_nucleus.sh`, implemented plasmidBin auto-detect
matching upstream `nucleus_launcher.sh`, and verified LAN federation on
`0.0.0.0:7700`.

## What changed

### 1. Primordial patterns cut

`find_binary()` rewritten to error hard if binary not in plasmidBin.
Three fallback paths removed:

- `$ECO_ROOT/primals/$name/target/release/$name` (direct source build)
- CamelCase primal directory scan (`primals/*/target/release/`)
- `which $name` PATH lookup

### 2. plasmidBin auto-detect

Added `detect_host_triple()` + `detect_bin_dir()` functions matching
primalSpring's launcher:

1. Git checkout: `infra/plasmidBin/primals/{triple}/`
2. Git checkout root: `infra/plasmidBin/primals/`
3. XDG fallback: `$XDG_DATA_HOME/ecoPrimals/plasmidBin/primals/{triple}/`

### 3. Federation bind

- `SONGBIRD_FEDERATION_BIND` env var documented and wired.
- `--bind` flag conditionally passed (feature-detected; Songbird v0.2.1
  lacks it, `--port` alone already binds to `*`).

### 4. PATH cleanup

Removed 5 stale echo-only stubs from `~/.local/bin/`:
beardog, songbird, toadstool, nestgate, squirrel.

## Deployment results

| Metric | Value |
|--------|-------|
| Primals started | 12/13 (loamSpine: upstream Tokio panic) |
| UDS sockets | 9/13 responsive |
| Federation | `*:7700` (all interfaces) |
| TCP health | `{"status":"healthy"}` |
| eastGate cross-gate | Unreachable (different subnet) |

## Cross-gate mesh gap

southGate LAN IP: `192.168.4.29`
eastGate LAN IP: `192.168.1.144`

Different subnets — direct mesh requires either subnet routing or
cellMembrane TURN relay. Local federation is correctly configured.

## Known pipeline debt (from Wave 49 audit)

| Issue | Status |
|-------|--------|
| petalTongue musl binary rejects `--family-id` | Handled via `FAMILY_ID` env |
| petalTongue stale socket on restart | Cleaned before restart |
| loamSpine Tokio runtime-in-runtime panic | Upstream bug, does not block mesh |
| Songbird sled DB corruption | Cleaned before restart |
| Songbird `--bind` flag (v0.2.1) | Feature-guarded; `--port` alone works |

## Test results

- `cargo test --lib --no-default-features`: 754 tests, 0 failures
- guideStone validation: 30/37 PASS, 7 partial (known limitations)

## Files changed

- `tools/composition_nucleus.sh` — `find_binary()` rewrite, auto-detect, bind guard
- `graphs/neuralspring_deploy.toml` — V174/S218
- `docs/PRIMAL_GAPS.md` — Gap 31 added
- `CHANGELOG.md` — S218 entry
- All doc headers synced to S218/V174
