// SPDX-License-Identifier: AGPL-3.0-or-later

// wetSpring provenance: diversity stats, Pearson, NMF.

use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once};

pub fn validate_wetspring_bio(h: &mut ValidationHarness) {
    println!("\n─── wetSpring provenance: bio + diversity ───\n");

    // Shannon diversity (wetSpring → BarraCUDA)
    let counts = [10.0, 20.0, 30.0, 40.0];
    let (shannon, _) = bench_once("shannon (wS→BarraCUDA)", || {
        barracuda::stats::shannon(&counts)
    });
    let expected_shannon = {
        let total: f64 = counts.iter().sum();
        -counts
            .iter()
            .filter(|&&c| c > 0.0)
            .map(|c| {
                let p = c / total;
                p * p.ln()
            })
            .sum::<f64>()
    };
    h.check_abs(
        "wS→diversity: shannon",
        shannon,
        expected_shannon,
        tolerances::CROSS_LANGUAGE,
    );

    // Bray-Curtis distance (wetSpring → BarraCUDA)
    let a = [10.0, 20.0, 30.0];
    let b = [15.0, 25.0, 35.0];
    let (bc, _) = bench_once("bray_curtis (wS→BarraCUDA)", || {
        barracuda::stats::bray_curtis(&a, &b)
    });
    let expected_bc = {
        let sum_min: f64 = a.iter().zip(&b).map(|(x, y)| x.min(*y)).sum();
        let sum_a: f64 = a.iter().sum();
        let sum_b: f64 = b.iter().sum();
        1.0 - 2.0 * sum_min / (sum_a + sum_b)
    };
    h.check_abs(
        "wS→diversity: bray_curtis",
        bc,
        expected_bc,
        tolerances::CROSS_LANGUAGE,
    );

    // Alpha diversity (wetSpring → BarraCUDA)
    let abundances = [5.0, 10.0, 15.0, 20.0, 25.0, 25.0];
    let alpha = barracuda::stats::alpha_diversity(&abundances);
    h.check_bool(
        "wS→diversity: alpha_diversity computed",
        alpha.shannon > 0.0,
    );
    h.check_bool(
        "wS→diversity: chao1 ≥ observed",
        alpha.chao1 >= alpha.observed,
    );

    // Simpson diversity (wetSpring → BarraCUDA)
    let (simpson, _) = bench_once("simpson (wS→BarraCUDA)", || {
        barracuda::stats::simpson(&abundances)
    });
    h.check_bool(
        "wS→diversity: simpson in [0,1]",
        (0.0..=1.0).contains(&simpson),
    );

    // Pearson correlation (wetSpring hydrology → BarraCUDA)
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    let (r, _) = bench_once("pearson (wS/aS→BarraCUDA)", || {
        barracuda::stats::pearson_correlation(&x, &y)
    });
    h.check_abs(
        "wS→stats: pearson(linear)",
        r.unwrap_or(0.0),
        1.0,
        tolerances::CROSS_LANGUAGE,
    );

    // NMF (wetSpring ESN → BarraCUDA)
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let nmf_result = barracuda::linalg::nmf(
        &data,
        2,
        3,
        &barracuda::linalg::NmfConfig {
            rank: 2,
            max_iter: 100,
            objective: barracuda::linalg::NmfObjective::Euclidean,
            seed: 42,
            tol: tolerances::NMF_CONVERGENCE_TOL,
        },
    );
    h.check_bool("wS→linalg: NMF converges", nmf_result.is_ok());
    if let Ok(ref r) = nmf_result {
        h.check_bool(
            "wS→linalg: NMF reconstruction finite",
            barracuda::linalg::relative_reconstruction_error(&data, r).is_finite(),
        );
    }
}
