# neuralSpring — Evolution Mapping: Rust Module → WGSL Shader → Pipeline Stage

**Last Updated**: February 18, 2026
**Purpose**: Concrete mapping from Phase 0 Python → Phase 1 Rust → Phase 2 GPU

---

## Tier Classification

| Tier | Meaning | Criteria |
|------|---------|----------|
| **A** (rewire) | Direct port — pure math, no framework dependencies | NumPy-only implementations, analytical known-values |
| **B** (adapt) | Needs adaptation — training loops, data dependencies | PyTorch training, real data, stochastic |
| **C** (new) | New implementation — no Python equivalent | GPU-specific (flash attention, fused kernels) |

---

## Module-by-Module Mapping

### Tier A — Direct Rewire (ready for Rust port)

| Python Module | Rust Module | WGSL Shader | Pipeline Stage | Blocker |
|---------------|-------------|-------------|----------------|---------|
| `transformer/` softmax | `transformer::softmax` | `attention.wgsl` (softmax stage) | Inference | **None** — implemented |
| `transformer/` GELU | `transformer::gelu` | elementwise | Inference | **None** — implemented |
| `transformer/` LayerNorm | `transformer::layer_norm` (stub) | `layer_norm.wgsl` | Inference | Implement norm |
| `transformer/` SDPA | `transformer::sdpa` (stub) | `attention.wgsl` | Inference | Implement QKV matmul |
| `surrogate/` Rastrigin | `surrogate::rastrigin_2d` | N/A (test function) | Validation | **None** — implemented |
| `surrogate/` Rosenbrock | `surrogate::rosenbrock_2d` | N/A (test function) | Validation | **None** — implemented |
| `surrogate/` Ackley | `surrogate::ackley_2d` | N/A (test function) | Validation | **None** — implemented |
| `surrogate/` R²/RMSE/MAE | `metrics::*` | `FusedMapReduceF64` | Validation | **None** — implemented |

### Tier B — Adapt (needs training infrastructure)

| Python Module | Rust Module | WGSL Shader | Pipeline Stage | Blocker |
|---------------|-------------|-------------|----------------|---------|
| `surrogate/` MLP forward | `surrogate::mlp_forward` (stub) | `gemm_f64.wgsl` + `nn::ReLU` | Inference | BarraCUDA `nn::Layer` |
| `surrogate/` MLP training | `surrogate::mlp_train` (stub) | `gemm_f64.wgsl` + `nn::Optimizer::Adam` | Training | BarraCUDA autograd |
| `sequence/` LSTM cell | — | `lstm_cell.wgsl` | Inference | BarraCUDA LSTM primitive |
| `sequence/` GRU cell | — | `gru_cell.wgsl` | Inference | BarraCUDA GRU primitive |
| `pinn/` autograd | — | `fd_gradient_f64.wgsl` | Training | Reverse-mode AD in BarraCUDA |
| `lenet/` Conv2d | — | `conv2d.wgsl` | Inference | BarraCUDA Conv2d |
| `lenet/` MaxPool | — | `max_pool2d.wgsl` | Inference | BarraCUDA pooling |
| `deeponet/` Branch-Trunk | — | `gemm_f64.wgsl` × 2 | Inference | Compose from MLP |
| `quantized/` INT8 GEMV | — | `gemv_q8.wgsl` | Deployment | BarraCUDA Q8 kernels |
| `quantized/` INT4 GEMV | — | `gemv_q4.wgsl` | Deployment | BarraCUDA Q4 kernels |
| `transfer/` freeze+finetune | — | selective gradient | Training | BarraCUDA param freeze |

### Tier C — New (GPU-specific, no Python equivalent)

| Capability | WGSL Shader | Pipeline Stage | Blocker |
|------------|-------------|----------------|---------|
| Flash attention | `flash_attention.wgsl` | Inference | Algorithm implementation |
| Fused LayerNorm+GELU | fused kernel | Inference | Kernel fusion framework |
| Batched GEMM | `gemm_f64.wgsl` (batched) | Training | Batch dispatch |
| Population fitness eval | `gemm_f64.wgsl` + selection | Evolution (Dolson) | GA/ES framework |
| HMM Viterbi | log-sum-exp + traceback | Genomics (Liu) | New primitive |
| Gillespie SSA | GPU PRNG + exp sampling | Biology (Waters) | New primitive |

---

## Promotion Checklist

For each Rust module → GPU promotion:

- [ ] Python baseline passes with documented provenance
- [ ] Rust implementation matches Python to documented tolerance
- [ ] WGSL shader exists in BarraCUDA or is planned
- [ ] Validation binary follows hotSpring pattern (exit 0/1)
- [ ] Performance meets or exceeds Python baseline
- [ ] Test coverage ≥ 90% (analytical + round-trip + determinism)

---

## Current Status (February 2026)

| Phase | Status | Coverage |
|-------|--------|----------|
| Phase 0 (Python baselines) | **75/75 PASS** | 10 experiments |
| Phase 1 (Rust validation) | **Scaffolded** | 3 modules, 10 tests, 2 validation binaries |
| Phase 2 (GPU shaders) | **Planned** | Mapping documented above |
| Phase 3 (Sovereign pipeline) | **Planned** | Depends on Phase 2 |
