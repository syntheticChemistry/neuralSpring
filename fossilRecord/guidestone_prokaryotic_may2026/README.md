# Fossil Record: neuralspring_guidestone (Prokaryotic)

**What:** Standalone `neuralspring_guidestone` binary — 580-line monolithic
guidestone with 4-layer certification (bare, discovery, parity, nucleus).

**When:** May 2026, Interstadial Primordial Extinction event.

**Why:** Eukaryotic evolution absorbed the guidestone logic into a
`src/certification/` library organelle with 4 submodules (`bare.rs`,
`discovery.rs`, `parity.rs`, `nucleus.rs`). The `certify` subcommand
in the UniBin now delegates to `certification::certify(max_layer)`.

**Superseded by:**
- `src/certification/mod.rs` — library organelle
- `neuralspring-unibin certify` — UniBin subcommand

**Provenance:** Session S193, neuralSpring V143.
