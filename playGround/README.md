# playGround — neuralSpring Application Sandbox

**The sandbox where science meets application.** While the main neuralSpring
library validates Python baselines and proves barraCuda primitives produce
correct learning, playGround takes those validated capabilities and wires
them into real applications: MCP tool integration, AI-driven experiment
analysis, GPU inference via the compute triangle, and benchmarking against
PyTorch/CUDA.

## Compute Triangle

playGround leverages the full ecoPrimals GPU stack:

```
                    ┌──────────────┐
                    │  neuralSpring │  (inference, benchmarks)
                    │   playGround  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ barraCuda │ │ToadStool │ │coralReef │
        │  (math)   │ │(compute) │ │(compiler)│
        └──────────┘ └──────────┘ └──────────┘
        719 WGSL      Job queue    WGSL → SASS
        TensorSession Pipeline $   DRM ioctl
        wgpu/Vulkan   Routing      No Vulkan
```

| Tier | Primal | Role | Dispatch path |
|------|--------|------|---------------|
| 0 | **barraCuda** | Math engine, Tensor API | WGSL → wgpu → Vulkan → GPU |
| 1 | **ToadStool** | Orchestration, job queue | JSON-RPC → barraCuda |
| 2 | **coralReef** | Sovereign compiler | WGSL → native SASS/GFX → DRM ioctl |

Tier 2 bypasses the entire wgpu/Vulkan stack — critical for closing the
PyTorch/CUDA dispatch latency gap.

## What's Here

### Binaries

| Binary | Purpose |
|--------|---------|
| `neuralspring_mcp_adapter` | Bridges Squirrel MCP ↔ neuralSpring primal (14 `science.*` tools) |
| `neuralspring_interactive` | AI-driven interactive experiment runner (Squirrel + science) |
| `neuralspring_model_lab` | HuggingFace model exploration: download, inspect, load, forward pass |
| `neuralspring_bench_inference` | Benchmark: barraCuda/WGSL vs PyTorch/CUDA (cold + hot dispatch) |
| `neuralspring_compute_probe` | Probe all three compute triangle tiers for availability and latency |

### Library Modules

| Module | Purpose |
|--------|---------|
| `ipc_client` | Reusable JSON-RPC 2.0 client, biomeOS 5-tier socket discovery |
| `squirrel_client` | Typed Squirrel MCP client: `ai.query`, `tool.execute`, `capability.announce` |
| `primal_client` | Typed neuralSpring primal client: all 14 `science.*` capabilities |
| `toadstool_client` | Typed ToadStool client: `compute.submit`, `gpu.dispatch`, `gpu.info` |
| `coralreef_client` | Typed coralReef client: `shader.compile.wgsl`, compiler capabilities |
| `mcp_tools` | MCP tool definitions (JSON Schema) for Squirrel registration |
| `secrets` | API key loading from `testing-secrets/api-keys.toml` |
| `hf_hub` | HuggingFace Hub download client (safetensors, config.json, tokenizer) |
| `model_config` | HF model config parser → unified `TransformerConfig` |
| `inference` | GPU inference engine: weight loading + transformer forward pass |

## Benchmarking

```bash
# Probe available compute tiers
cargo run --release --bin neuralspring_compute_probe

# Benchmark ops — cold dispatch (new TensorSession per call)
cargo run --release --bin neuralspring_bench_inference -- --ops-only

# Benchmark ops — hot dispatch (reused TensorSession, pre-compiled pipelines)
cargo run --release --bin neuralspring_bench_inference -- --ops-only --hot

# Full comparison with PyTorch/CUDA
./playGround/bench/compare.sh --ops-only

# JSON output for CI
cargo run --release --bin neuralspring_bench_inference -- --hot --json
```

### What cold vs hot means

- **Cold**: Each iteration creates a new `TensorSession`, compiling 17 shader
  pipelines (WGSL → SPIR-V → Vulkan compute pipeline). ~771ms first time.
- **Hot**: Reuses a single `TensorSession` with `reset()`. Compiled pipelines
  persist — only bind-group creation + dispatch encoding per call.

PyTorch/CUDA is always "hot" — cuBLAS kernels are compiled once by the driver
and cached for the process lifetime. `--hot` is the fair comparison.

## Architecture

```
Squirrel (ai.query, tool.execute)
    ↕  JSON-RPC / Unix socket
neuralspring_mcp_adapter
    ↕  JSON-RPC / Unix socket
neuralspring_primal (14 science.* capabilities)
    ↕  library calls
neural-spring lib (1115 tests, barraCuda GPU math)
    ↕  TensorSession / wgpu
barraCuda (719 WGSL shaders)
    ↕  wgpu / Vulkan  OR  coralReef / DRM ioctl
GPU hardware
```

## Usage

```bash
# Probe compute triangle availability
cargo run --release --bin neuralspring_compute_probe

# Explore a HuggingFace model
cargo run --release --bin neuralspring_model_lab -- info openai-community/gpt2
cargo run --release --bin neuralspring_model_lab -- download openai-community/gpt2
cargo run --release --bin neuralspring_model_lab -- forward openai-community/gpt2 --tokens "1,2,3,4"

# Start the MCP adapter (bridges to Squirrel)
cargo run --bin neuralspring_mcp_adapter

# Start the interactive runner
cargo run --bin neuralspring_interactive
```

## Node Atomic Deployment

When running on a Node Atomic (BearDog + Songbird + ToadStool), all primals
are automatically discovered via biomeOS socket scanning. The adapter bridges
neuralSpring science capabilities to Squirrel MCP, enabling AI-driven
scientific computing on sovereign local hardware.

With ToadStool running, playGround can route GPU workloads through the
orchestration layer for persistent sessions and workload batching. With
coralReef running, WGSL shaders are compiled to native GPU binaries for
minimal-latency dispatch.

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

1. **Socket-decoupled**: playGround talks to primals via Unix sockets,
   not library imports. The sandbox is independent and testable without
   compiling the full neural-spring library (except barraCuda for direct GPU).

2. **Discovery-based**: All primal connections use biomeOS 5-tier socket
   resolution. No hardcoded paths.

3. **Graceful degradation**: If ToadStool/coralReef are unavailable, fall
   back to direct barraCuda. If Squirrel is unavailable, the adapter runs
   standalone. Each tier reports its status clearly.

4. **Evolve, don't plan**: This is a sandbox. Build, test, break, fix, evolve.
   Less specification, more iteration.
