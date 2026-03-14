# playGround — neuralSpring Application Sandbox

**The sandbox where science meets application.** While the main neuralSpring
library validates Python baselines and proves barraCuda primitives produce
correct learning, playGround takes those validated capabilities and wires
them into real applications: MCP tool integration, AI-driven experiment
analysis, and interactive scientific computing.

## What's Here

### Binaries

| Binary | Purpose |
|--------|---------|
| `neuralspring_mcp_adapter` | Bridges Squirrel MCP and the neuralSpring primal — registers 14 `science.*` capabilities as MCP tools, forwards `tool.execute` calls |
| `neuralspring_interactive` | AI-driven interactive experiment runner — combines neuralSpring science with Squirrel AI for conversational analysis |

### Library Modules

| Module | Purpose |
|--------|---------|
| `ipc_client` | Reusable JSON-RPC 2.0 client over Unix domain sockets with biomeOS 5-tier discovery |
| `squirrel_client` | Typed Squirrel MCP client: `ai.query`, `tool.execute`, `capability.announce` |
| `primal_client` | Typed neuralSpring primal client: all 14 `science.*` capabilities |
| `mcp_tools` | MCP tool definitions (JSON Schema) for Squirrel registration |

## Architecture

```
Squirrel (ai.query, tool.execute)
    ↕  JSON-RPC / Unix socket
neuralspring_mcp_adapter
    ↕  JSON-RPC / Unix socket
neuralspring_primal (14 science.* capabilities)
    ↕  library calls
neural-spring lib (1115 tests, barraCuda GPU math)
```

The adapter is a pure bridge — no science logic, just protocol translation.
The interactive runner connects to both Squirrel (AI) and the primal (science)
for a combined experience.

## Usage

```bash
# Start the neuralSpring primal (prerequisite)
cargo run --bin neuralspring_primal --features primal -- serve

# Start the MCP adapter (bridges to Squirrel)
cargo run --bin neuralspring_mcp_adapter

# Start the interactive runner
cargo run --bin neuralspring_interactive

# With explicit socket paths
cargo run --bin neuralspring_mcp_adapter -- \
    --primal-socket /run/user/1000/biomeos/neuralspring-default.sock \
    --squirrel-socket /run/user/1000/biomeos/squirrel-default.sock
```

## Node Atomic Deployment

When running on a Node Atomic (BearDog + Songbird + ToadStool), all primals
are automatically discovered via biomeOS socket scanning. The adapter bridges
neuralSpring science capabilities to Squirrel MCP, enabling AI-driven
scientific computing on sovereign local hardware.

## Lysogeny Protocol Awareness

neuralSpring is assigned **cross-domain validation** for three Lysogeny targets:

| Target | Assignment | Module |
|--------|-----------|--------|
| Usurper | Evolutionary game theory | `game_theory.rs` |
| Symbiont | Multi-agent cooperation | `agent_coordination.rs`, `eco_dynamics.rs` |
| Pathogen | Reward prediction error | (future playGround experiment) |

playGround applications inherit the 7-link Lysogeny provenance chain:
published paper → barraCuda primitive → spring experiment → cross-domain
validation → vocabulary mapping → AGPL-3.0-or-later → wateringHole catalog.

## scyBorg Licensing

All playGround code is licensed under **AGPL-3.0-or-later** (code layer).
Documentation and creative content fall under **CC-BY-SA 4.0** (creative layer).
If game mechanics are involved (Pathogen anti-pattern analysis): **ORC**
(mechanics layer).

## Design Principles

1. **Socket-decoupled**: playGround talks to the primal via Unix sockets,
   not library imports. This keeps the sandbox independent and testable
   without compiling the full neural-spring library.

2. **Discovery-based**: All primal connections use the biomeOS 5-tier
   socket resolution. No hardcoded paths.

3. **Graceful degradation**: If Squirrel is unavailable, the adapter runs
   in standalone mode. If the primal is unavailable, binaries report clearly.

4. **Evolve, don't plan**: This is a sandbox. Build, test, break, fix, evolve.
   Less specification, more iteration.
