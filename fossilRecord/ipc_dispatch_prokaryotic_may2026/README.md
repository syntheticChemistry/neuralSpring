# Fossil Record: ipc_dispatch (Prokaryotic)

**What:** Monolithic `src/ipc_dispatch.rs` — 401-line `IpcMathClient` handling
barraCuda, toadStool, BearDog, and Squirrel IPC in a single file.

**When:** May 2026, Interstadial Primordial Extinction event.

**Why:** Eukaryotic evolution graduated the IPC surface into a per-primal
`src/ipc/` tree with dedicated modules for each primal:
- `ipc/barracuda.rs` — tensor lifecycle, core math, ML ops
- `ipc/toadstool.rs` — compute dispatch
- `ipc/beardog.rs` — crypto operations
- `ipc/squirrel.rs` — inference routing
- `ipc/coralreef.rs` — shader compilation (new)

The `IpcMathClient` facade remains in `ipc/mod.rs` and delegates to
per-primal functions.

**Superseded by:** `src/ipc/` module tree.

**Provenance:** Session S193, neuralSpring V143.
