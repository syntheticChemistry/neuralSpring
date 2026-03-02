// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-language reference arrays and analytical reference sources.

// ═══════════════════════════════════════════════════════════════════
// Analytical reference sources
// ═══════════════════════════════════════════════════════════════════

/// `BarraCUDA` validation expected values are analytically derived — no Python
/// dependency.  Provenance is mathematical: NIST DLMF, IEEE 754, and textbook
/// formulas.
pub const BARRACUDA_ANALYTICAL_REFS: &str = "Analytical (IEEE 754, NIST DLMF, textbook formulas)";

/// Chi-squared distribution reference values.
///
/// PDF/CDF validated against `SciPy` 1.15.3 `scipy.stats.chi2`.
/// Moments and test statistic are analytically derived.
///
/// Provenance:
/// ```text
/// python3 -c "from scipy.stats import chi2; print(chi2.pdf(2,3), chi2.pdf(0,3), chi2.pdf(5,1))"
/// python3 -c "from scipy.stats import chi2; print(chi2.cdf(3.84,1), chi2.cdf(5.99,2), chi2.cdf(0,5))"
/// ```
/// Environment: `SciPy` 1.15.3, Python 3.10.12, 2026-02-16
pub const CHI_SQUARED_REFS: &str = "SciPy 1.15.3 chi2 + analytical moments (Pearson 1900)";

/// FFT validation: analytical DFT pairs + Parseval's theorem.
///
/// No Python dependency — all expected values derive from the definition of
/// the Discrete Fourier Transform (Cooley & Tukey, 1965; FFTW docs).
pub const FFT_ANALYTICAL_REFS: &str =
    "Analytical (DFT definition, Parseval's theorem, Cooley-Tukey 1965)";

// ═══════════════════════════════════════════════════════════════════
// Cross-language reference values (Python-computed, hardcoded in Rust)
// ═══════════════════════════════════════════════════════════════════

/// Softmax of `[1,2,3,4,5]` computed by `NumPy` 2.2.6.
///
/// Provenance: `python3 -c "import numpy as np; x=np.array([1.,2.,3.,4.,5.]); e=np.exp(x-x.max()); print(e/e.sum())"`
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
/// Commit: `BASELINE_COMMIT` (`f9ad0268`), Date: `BASELINE_DATE` (2026-02-16).
pub const SOFTMAX_1_TO_5: [f64; 5] = [
    1.165_623_095_603_961e-2,
    3.168_492_079_612_427e-2,
    8.612_854_443_626_87e-2,
    2.341_216_572_527_366e-1,
    6.364_086_465_588_308e-1,
];

/// GELU reference values at selected points, computed by `NumPy` 2.2.6.
///
/// Format: (input, `expected_output`)
/// Provenance: `python3 -c "import numpy as np; gelu=lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))); [print(x,gelu(x)) for x in [-2,-1,0,0.5,1,3]]"`
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
/// Commit: `BASELINE_COMMIT` (`f9ad0268`), Date: `BASELINE_DATE` (2026-02-16).
pub const GELU_REFERENCE: [(f64, f64); 6] = [
    (-2.0, -4.540_230_591_222_494e-2),
    (-1.0, -1.588_080_093_917_233e-1),
    (0.0, 0.0),
    (0.5, 3.457_140_098_251_439e-1),
    (1.0, 8.411_919_906_082_768e-1),
    (3.0, 2.996_362_607_918_227),
];

/// Rastrigin 2D reference values at non-trivial points, computed by `NumPy` 2.2.6.
///
/// Provenance: `python3 control/surrogate/surrogate_validation.py` (`rastrigin_2d`).
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
/// Commit: `BASELINE_COMMIT` (`f9ad0268`), Date: `BASELINE_DATE` (2026-02-16).
pub const RASTRIGIN_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 2.0),
    (2.5, -1.3, 4.103_016_994_374_947e1),
    (0.5, 0.5, 4.05e1),
    (-3.0, 2.0, 13.0),
];

/// Rosenbrock 2D reference values, computed by `NumPy` 2.2.6.
///
/// Provenance: `python3 -c "f=lambda x,y: (1-x)**2 + 100*(y-x**2)**2; [print(x,y,f(x,y)) for x,y in [(1,1),(2.5,-1.3),(0.5,0.5),(-3,2)]]"`
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
/// Commit: `BASELINE_COMMIT` (`f9ad0268`), Date: `BASELINE_DATE` (2026-02-16).
pub const ROSENBROCK_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 0.0),
    (2.5, -1.3, 5702.5),
    (0.5, 0.5, 6.5),
    (-3.0, 2.0, 4916.0),
];

/// Ackley 2D reference values, computed by `NumPy` 2.2.6.
///
/// Provenance: `python3 -c "import numpy as np; a=lambda x,y: -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.e + 20; [print(x,y,a(x,y)) for x,y in [(1,1),(2.5,-1.3),(0.5,0.5),(-3,2)]]"`
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
/// Commit: `BASELINE_COMMIT` (`f9ad0268`), Date: `BASELINE_DATE` (2026-02-16).
pub const ACKLEY_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 3.625_384_938_440_363),
    (2.5, -1.3, 8.772_020_879_614_113),
    (0.5, 0.5, 4.253_654_026_568_412),
    (-3.0, 2.0, 7.988_910_810_518_7),
];

/// Analytical reference source for benchmark functions.
pub const BENCHMARK_REFS: &str = "Analytical global minima + NumPy 2.2.6 cross-validation";

/// Analytical reference source for transformer primitives.
pub const TRANSFORMER_REFS: &str = "NumPy 2.2.6 transformer_inference.py (softmax, gelu_numpy)";

/// Analytical reference source for statistical metrics.
pub const METRICS_REFS: &str = "Analytical (pure arithmetic on known arrays)";
