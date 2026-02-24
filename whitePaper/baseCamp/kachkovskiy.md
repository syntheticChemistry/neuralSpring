# Ilya Kachkovskiy — Spectral Theory

**Institution**: Michigan State University
**Track**: Spectral theory, Anderson localization, operator theory
**Papers**: 2 (022–023)
**Total Checks**: 16
**Domains**: Spectral commutativity, Anderson localization, eigenvalue distribution

## Connection to neuralSpring

Kachkovskiy's spectral theory work validates the eigenvalue decomposition
primitives that are foundational to principal component analysis, spectral
clustering, and dimensionality reduction in ML. neuralSpring's `eigh_f64`
(Householder tridiagonalisation → implicit QR) is the exact algorithm needed
for both Anderson localization and spectral feature extraction.

## Papers

| # | Citation | Rust Module | Checks | Status |
|---|----------|-------------|--------|--------|
| 022 | Kachkovskiy & Safarov (2016) *Spectral theory of one-dimensional operators*. J Spectral Theory. | `spectral_commutativity.rs` | 8 | **ALL TIERS PASS** |
| 023 | Kachkovskiy (2024) *Anderson localization for quasi-periodic operators*. J Funct Anal. | `anderson_localization.rs` | 8 | **ALL TIERS PASS** |

## Evolution Path

| Tier | Status | Key Primitive |
|------|--------|---------------|
| Python (Py) | 2/2 PASS | NumPy `linalg.eigh`, SciPy sparse |
| Rust (Rs) | 2/2 PASS | `eigh_f64`, `spectral_commutator` |
| BarraCUDA CPU (bC) | 2/2 PASS | `eigh_f64` (Householder+QR) |
| GPU Tensor (gT) | 1/2 PASS | `Tensor::matmul` chain (commutator only) |
| metalForge (mF) | 2/2 PASS | `stencil_cooperation.wgsl` (Anderson 1D lattice) |
| GPU Pipeline (gP) | 2/2 PASS | `anderson → eigh` chain |
| Cross-dispatch (xD) | 2/2 PASS | `eigh` GPU vs CPU Sturm+bisection parity |
