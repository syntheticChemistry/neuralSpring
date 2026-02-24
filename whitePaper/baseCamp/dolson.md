# Emily Dolson — Evolutionary Computation

**Institution**: Michigan State University
**Track**: Evolutionary computation, digital evolution, open-ended evolution
**Papers**: 5 (011–015)
**Total Checks**: 50
**Domains**: NK fitness landscapes, MODES metrics, eco-dynamics, directed evolution, swarm robotics

## Connection to neuralSpring

Dolson's work provides the evolutionary substrate: fitness evaluation, selection
algorithms, population dynamics. These are the same primitives that drive
evolutionary hyperparameter search, neural architecture search, and evolutionary
strategies in ML. neuralSpring validates that BarraCUDA's batch GEMM and
reduction ops reproduce these algorithms exactly.

## Papers

| # | Citation | Rust Module | Checks | Status |
|---|----------|-------------|--------|--------|
| 011 | Iram, Dolson et al. (2020) *Controlling evolution with counterdiabatic driving*. Nature Physics. | `counterdiabatic.rs` | 19 | **ALL TIERS PASS** |
| 012 | Dolson et al. (2019) *The MODES Toolbox*. Artificial Life. | `modes.rs` | 9 | **ALL TIERS PASS** |
| 013 | Dolson & Ofria (2018) *Ecological Theory Provides Insights about EC*. GECCO. | `eco_dynamics.rs` | 7 | **ALL TIERS PASS** |
| 014 | Dolson, Banzhaf, Ofria (2022) *Artificial selection methods from EC*. eLife. | `directed_evolution.rs` | 7 | **ALL TIERS PASS** |
| 015 | Foreback & Dolson (2025) *Heterogeneous swarm controllers*. IEEE. | `swarm_robotics.rs` | 7 | **ALL TIERS PASS** |

## Evolution Path

| Tier | Status | Key Primitive |
|------|--------|---------------|
| Python (Py) | 5/5 PASS | NumPy random, fitness evaluation |
| Rust (Rs) | 5/5 PASS | `rng::Rng`, flat `Vec<f64>` layouts |
| BarraCUDA CPU (bC) | 5/5 PASS | `FusedMapReduceF64`, `BatchFitnessGpu` |
| GPU Tensor (gT) | 5/5 PASS | `Tensor::matmul`, `BatchedOdeRK4F64` |
| metalForge (mF) | 5/5 PASS | `batch_fitness_eval.wgsl`, `multi_obj_fitness.wgsl`, `swarm_nn_forward.wgsl` |
| GPU Pipeline (gP) | 3/5 PASS | `batch_fitness → mean_reduce`, `multi_obj → mean_reduce` |
| Cross-dispatch (xD) | 5/5 PASS | `DispatchConfig` routing |
