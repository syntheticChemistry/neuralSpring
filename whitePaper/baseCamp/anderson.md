# R. Anderson / Campbell — Population Genetics

**Institution**: Various
**Track**: Pangenome selection, meta-population dynamics
**Papers**: 2 (024–025)
**Total Checks**: 16
**Domains**: Pangenome-scale selection coefficients, Wright-Fisher meta-population dynamics

## Connection to neuralSpring

Population genetics models use the same stochastic simulation primitives
(random number generation, fitness evaluation, selection-mutation balance) that
appear in evolutionary ML algorithms. neuralSpring validates that BarraCUDA's
Wright-Fisher GPU kernel and batch fitness evaluation reproduce these population
dynamics faithfully, providing the stochastic substrate for evolutionary NAS
and hyperparameter tuning.

### gen3 baseCamp Cross-References

Anderson/Campbell Papers 024-025 connect to gen3 Sub-02 (LTEE Extensions):
pangenome selection and meta-population dynamics model the convergent pathway
evolution predicted across LTEE replicates. Wright-Fisher GPU kernels validated
here serve both neuralSpring Sub-05 (agent populations) and gen3 Sub-02
(evolutionary dynamics).

## Papers

| # | Citation | Rust Module | Checks | Status |
|---|----------|-------------|--------|--------|
| 024 | Anderson & Campbell (2022) *Pangenome-scale selection coefficients*. G3. | `pangenome_fst.rs` | 8 | **ALL TIERS PASS** |
| 025 | Campbell et al. (2023) *Meta-population dynamics with gene flow*. Genetics. | `meta_population.rs` | 8 | **ALL TIERS PASS** |

## Evolution Path

| Tier | Status | Key Primitive |
|------|--------|---------------|
| Python (Py) | 2/2 PASS | NumPy random, scipy.stats |
| Rust (Rs) | 2/2 PASS | `rng::Rng` (Xoshiro256**), `fst_weir_cockerham` |
| BarraCUDA CPU (bC) | 2/2 PASS | `WrightFisherGpu`, `FusedMapReduceF64` |
| GPU Tensor (gT) | 2/2 PASS | `Tensor::matmul`, `WrightFisherF64` |
| metalForge (mF) | 2/2 PASS | `wright_fisher.wgsl`, `batch_fitness_eval.wgsl` |
| GPU Pipeline (gP) | 2/2 PASS | `wright_fisher → reduce → fst` chain |
| Cross-dispatch (xD) | 2/2 PASS | `DispatchConfig` population routing |
