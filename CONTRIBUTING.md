# Contributing to neuralSpring

**License**: AGPL-3.0-or-later (scyBorg trio: AGPL + ORC + CC-BY-SA)

neuralSpring is part of the [ecoPrimals](https://github.com/ecoPrimals) ecosystem.
Contributions that strengthen validation fidelity, improve barraCuda absorption,
or advance GPU evolution are welcome.

## Prerequisites

- **Rust 1.87+** (edition 2024)
- **barraCuda** checkout at `../../primals/barraCuda` (path dependency)
- GPU with Vulkan support (optional — CI validates on llvmpipe fallback)
- Python 3.11+ with NumPy (for baseline regeneration)

## Quality Standards

Every change must maintain these invariants:

| Standard | Requirement |
|----------|-------------|
| `cargo clippy --workspace --all-features` | Zero warnings (pedantic + nursery) |
| `cargo fmt --check` | Zero formatting drift |
| `cargo doc --workspace --no-deps` | Zero warnings (`-D warnings`) |
| `cargo test --workspace --lib` | All pass |
| `cargo llvm-cov --workspace --lib` | ≥ 90% line coverage |
| `cargo deny check` | Zero advisories, license, or source violations |
| `#![forbid(unsafe_code)]` | Zero unsafe in application code |
| `#[allow()]` | Zero in production — use `#[expect(lint, reason = "...")]` |
| File size | All `.rs` files under 1,000 lines |
| SPDX | Every `.rs` file starts with `// SPDX-License-Identifier: AGPL-3.0-or-later` |

## Tolerance Policy

All numeric thresholds live in `src/tolerances/`. No ad-hoc magic numbers in
validation binaries. Each tolerance constant must have:

- A descriptive name
- A doc comment explaining the mathematical justification
- Registration in `src/tolerances/registry.rs`

## Validation Binaries

Follow the hotSpring pattern:

1. Use `ValidationHarness` for all checks
2. Hardcode expected values with provenance (script, commit, date)
3. Exit 0 on all-pass, exit 1 on any failure
4. Reference tolerances from `src/tolerances/`

## barraCuda Evolution

When adding GPU compute:

1. Check if barraCuda already provides the primitive (`barracuda::ops`, `stats`, `linalg`)
2. If yes: delegate, don't duplicate
3. If no: implement locally in `metalForge/`, validate, then hand off via wateringHole
4. Track in `specs/EVOLUTION_MAPPING.md` and `specs/BARRACUDA_USAGE.md`

## IPC / Primal Coordination

Primal code has **self-knowledge only**. Never hardcode another primal's name,
socket path, or capability set. Discover at runtime via:

- `BIOMEOS_SOCKET_DIR` / `XDG_RUNTIME_DIR` socket scanning
- `capability.list` / `capability.resolve` JSON-RPC probes
- Graceful degradation when peers are unavailable

## Commit Conventions

- Keep a Changelog format in `CHANGELOG.md`
- Handoffs go to `wateringHole/handoffs/` with naming convention:
  `NEURALSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

## Running the Full Suite

```bash
just check          # clippy + fmt + doc
just test           # Python + Rust tests
just validate-all   # all 234 validation binaries
just coverage       # llvm-cov with 90% gate
```
