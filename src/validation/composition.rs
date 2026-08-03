// SPDX-License-Identifier: AGPL-3.0-or-later

//! Composition validation for NUCLEUS proto-nucleate patterns.
//!
//! Re-exports discovery, JSON-RPC probes, proto-nucleate graph definitions,
//! and science baselines for Rust→IPC parity validation.

pub use super::discovery::*;
pub use super::json_rpc::*;
pub use super::proto_nucleate::*;

/// Science capability baseline for Rust→IPC parity validation.
#[derive(Clone)]
pub struct ScienceBaseline {
    /// JSON-RPC method name (e.g. `science.spectral_analysis`).
    pub method: &'static str,
    /// JSON-RPC params to send.
    pub params: serde_json::Value,
    /// Keys in the IPC response to validate (each maps to a known Rust value).
    pub expected: Vec<(&'static str, f64)>,
    /// Absolute tolerance for numeric comparison.
    pub tolerance: f64,
}

/// Canonical science baselines for Rust→IPC parity.
#[must_use]
pub fn science_baselines() -> Vec<ScienceBaseline> {
    // Baseline 2: science.ipr (uniform wavefunction, IPR = 1/n)
    let ipr_uniform = {
        let n = 8_usize;
        let n_f64 = 8.0_f64;
        let amp = 1.0 / n_f64.sqrt();
        let wf: Vec<f64> = vec![amp; n];
        let expected_ipr = crate::anderson_localization::ipr(&wf);

        ScienceBaseline {
            method: "science.ipr",
            params: serde_json::json!({ "wavefunction": wf }),
            expected: vec![("ipr", expected_ipr)],
            tolerance: crate::tolerances::EXACT_F64,
        }
    };

    #[cfg(not(feature = "barracuda"))]
    {
        vec![ipr_uniform]
    }

    #[cfg(feature = "barracuda")]
    {
        use crate::anderson_localization::{anderson_hamiltonian_random, mean_ipr};
        use crate::eigh::eigh_householder_qr;
        use crate::rng::Rng;
        use crate::tolerances;
        use crate::weight_spectral;

        // Baseline 1: science.spectral_analysis (dim=16, disorder=2.0, seed=42)
        let spectral = {
            let n = 16;
            let w = 2.0;
            let seed = 42;
            let mut rng = Rng::new(seed);
            let h = anderson_hamiltonian_random(n, 1.0, w, &mut rng);
            let decomp = eigh_householder_qr(&h, n);
            let ipr_val = mean_ipr(&decomp.eigenvectors, n);
            let mut evals = decomp.eigenvalues;
            evals.sort_by(f64::total_cmp);
            let lsr = weight_spectral::level_spacing_ratio(&evals);
            let bw = weight_spectral::spectral_bandwidth(&evals);

            ScienceBaseline {
                method: "science.spectral_analysis",
                params: serde_json::json!({ "dim": n, "disorder": w, "seed": seed }),
                expected: vec![
                    ("mean_ipr", ipr_val),
                    ("level_spacing_ratio", lsr),
                    ("bandwidth", bw),
                ],
                tolerance: tolerances::SPECIAL_FUNCTION_F64,
            }
        };

        // Baseline 3: science.hessian_eigen (quadratic surface, dim=10)
        let hessian_quad = {
            let n = 10;
            let mut hessian = vec![0.0; n * n];
            for i in 0..n {
                #[expect(clippy::cast_precision_loss, reason = "small index → f64")]
                let v = (i + 1) as f64;
                hessian[i * n + i] = v;
            }
            let _decomp = eigh_householder_qr(&hessian, n);
            #[expect(clippy::cast_precision_loss, reason = "small index → f64")]
            let expected_trace = (1..=n).map(|i| i as f64).sum::<f64>();

            ScienceBaseline {
                method: "science.hessian_eigen",
                params: serde_json::json!({ "dim": n, "surface_type": "quadratic" }),
                expected: vec![("trace", expected_trace)],
                tolerance: tolerances::SPECIAL_FUNCTION_F64,
            }
        };

        // Baseline 4: science.disorder_sweep (lattice_size=10, seed=42)
        let disorder_sweep_baseline = {
            let n = 10;
            let seed = 42_u64;
            let w_vals = vec![1.0, 4.0, 16.0];
            let mut rng = Rng::new(seed);
            let iprs = crate::anderson_localization::disorder_sweep(n, 1.0, &w_vals, &mut rng);

            ScienceBaseline {
                method: "science.disorder_sweep",
                params: serde_json::json!({
                    "lattice_size": n,
                    "disorder_values": w_vals,
                    "seed": seed,
                    "hopping": 1.0,
                }),
                expected: iprs
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| match i {
                        0 => ("ipr_w1", v),
                        1 => ("ipr_w4", v),
                        _ => ("ipr_w16", v),
                    })
                    .collect(),
                tolerance: tolerances::SPECIAL_FUNCTION_F64,
            }
        };

        vec![spectral, ipr_uniform, hessian_quad, disorder_sweep_baseline]
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[cfg(feature = "barracuda")]
    #[test]
    fn science_baselines_non_empty() {
        let baselines = science_baselines();
        assert!(baselines.len() >= 4, "should have at least 4 baselines");
        for b in &baselines {
            assert!(!b.method.is_empty());
            assert!(!b.expected.is_empty());
            assert!(b.tolerance > 0.0);
        }
    }

    #[cfg(not(feature = "barracuda"))]
    #[test]
    fn science_baselines_cpu_minimal() {
        let baselines = science_baselines();
        assert_eq!(baselines.len(), 1);
        assert_eq!(baselines[0].method, "science.ipr");
    }

    #[cfg(feature = "barracuda")]
    #[test]
    fn science_baselines_deterministic() {
        let b1 = science_baselines();
        let b2 = science_baselines();
        for (a, b) in b1.iter().zip(b2.iter()) {
            assert_eq!(a.method, b.method);
            for ((k1, v1), (k2, v2)) in a.expected.iter().zip(b.expected.iter()) {
                assert_eq!(k1, k2);
                assert_eq!(
                    v1.to_bits(),
                    v2.to_bits(),
                    "baseline {k1} must be deterministic"
                );
            }
        }
    }

    #[test]
    fn science_baselines_ipr_uniform_is_one_over_n() {
        let baselines = science_baselines();
        let ipr_baseline = baselines
            .iter()
            .find(|b| b.method == "science.ipr")
            .expect("ipr baseline");
        let expected = 1.0 / 8.0;
        assert_eq!(ipr_baseline.expected.len(), 1);
        assert!(
            (ipr_baseline.expected[0].1 - expected).abs() < crate::tolerances::EXACT_F64,
            "uniform wf IPR = 1/n"
        );
    }
}
