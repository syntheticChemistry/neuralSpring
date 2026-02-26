# Kevin Liu — Phylogenetics / HMM

**Institution**: Michigan State University
**Track**: Phylogenetic inference, hidden Markov models, statistical alignment
**Papers**: 3 (016–018)
**Total Checks**: 38
**Domains**: HMM forward/backward/Viterbi, SATé progressive alignment, introgression detection

## Connection to neuralSpring

Liu's HMM and alignment methods share computational DNA with sequence models
(LSTM, transformer attention). The forward algorithm is matrix chain
multiplication in log-space — identical to the matmul primitive that drives
transformer inference. neuralSpring validates that BarraCUDA's GEMM chain
reproduces these algorithms in both CPU and GPU paths.

### gen3 baseCamp Cross-References

Liu Papers 016-018 connect to gen3 Sub-02 (LTEE Extensions): HMM/PhyloNet-HMM
for introgression detection in LTEE genomes, and transfer learning for
cross-environment adaptation. Also connect to gen3 Sub-05 (Cross-species
Signaling): HMM and introgression models for comparative genomics signal
detection across symbiotic species.

## Papers

| # | Citation | Rust Module | Checks | Status |
|---|----------|-------------|--------|--------|
| 016 | Liu et al. (2014) *Coalescent methods for phylogenetic trees*. PLoS Comp Bio. | `hmm.rs` | 17 | **ALL TIERS PASS** |
| 017 | Liu et al. (2009) *SATé: Simultaneous Alignment and Tree Estimation*. Science. | `sate_alignment.rs` | 8 | **ALL TIERS PASS** |
| 018 | Liu et al. (2015) *Introgression detection with PhyloNet-HMM*. PNAS. | `introgression.rs` | 13 | **ALL TIERS PASS** |

## Evolution Path

| Tier | Status | Key Primitive |
|------|--------|---------------|
| Python (Py) | 3/3 PASS | NumPy matmul, log-domain arithmetic |
| Rust (Rs) | 3/3 PASS | Flat `Vec<f64>`, `Hmm::from_flat()` |
| BarraCUDA CPU (bC) | 3/3 PASS | `HmmBatchForwardF64`, `eigh_f64` |
| GPU Tensor (gT) | 3/3 PASS | `Tensor::matmul` chain, `LogSumExpF64` |
| metalForge (mF) | 3/3 PASS | `hmm_forward_log.wgsl`, `pairwise_hamming.wgsl` |
| GPU Pipeline (gP) | 3/3 PASS | `hmm_forward → mean_reduce` |
| Cross-dispatch (xD) | 3/3 PASS | `DispatchConfig` HMM routing |
