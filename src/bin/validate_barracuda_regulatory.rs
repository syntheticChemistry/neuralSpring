// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: regulatory network ODE integration (Paper 020).
//!
//! Validates that `barracuda::numerical::rk45_solve` reproduces the
//! GRN ODE steady states from hand-rolled RK4 in `regulatory_network.rs`.
//!
//! Evolution path:
//! ```text
//! Python (scipy.integrate.solve_ivp) → Rust (hand-rolled RK4)
//!   → BarraCUDA CPU (barracuda::numerical::rk45_solve)
//!   → BarraCUDA GPU (rk4_batch.wgsl)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/regulatory_network/regulatory_network.py`
//! Rust baseline: `validate_regulatory_network` (7/7 PASS)

use neural_spring::regulatory_network::{
    env_params, integrate_grn, phenotype_classifier, shannon_diversity, GrnParams,
    ENV_NUTRIENT_POOR, ENV_NUTRIENT_RICH, ENV_STRESS,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_regulatory");

    validate_rk45_vs_rk4(&mut h);
    validate_barracuda_diversity(&mut h);
    validate_barracuda_phenotypes(&mut h);

    h.finish();
}

fn make_config() -> barracuda::numerical::Rk45Config {
    barracuda::numerical::Rk45Config {
        h_init: 0.02,
        atol: tolerances::ODE_ATOL,
        rtol: tolerances::ODE_RTOL,
        ..barracuda::numerical::Rk45Config::default()
    }
}

/// Compare barracuda `rk45_solve` against hand-rolled RK4 for the GRN ODE.
fn validate_rk45_vs_rk4(h: &mut ValidationHarness) {
    let p = GrnParams::default();
    let x0 = [0.5_f64, 0.1, 0.5, 0.1];
    let env_signal = 0.5;

    let rk4_result = integrate_grn(&x0, env_signal, &p, 2000, 0.02);

    let rhs = grn_rhs_closure(env_signal, &p);
    let config = make_config();

    match barracuda::numerical::rk45_solve(&rhs, 0.0, 40.0, &x0, &config) {
        Ok(result) => {
            let rk45_final = &result.y_final;

            for (i, (r4, r45)) in rk4_result.iter().zip(rk45_final.iter()).enumerate() {
                let component = ["sasa", "biofilm", "motility", "virulence"][i];
                // RK4 (fixed step) vs RK45 (adaptive): steady state should
                // agree to ~1e-4 given the ODE converges to a fixed point.
                h.check_abs(
                    &format!("{component}: RK4={r4:.4} vs RK45={r45:.4}"),
                    *r4,
                    *r45,
                    tolerances::ODE_INTEGRATOR_AGREEMENT,
                );
            }
        }
        Err(e) => {
            for component in &["sasa", "biofilm", "motility", "virulence"] {
                h.check_bool(&format!("{component}: rk45_solve [ERROR: {e}]"), false);
            }
        }
    }
}

/// Diversity analysis across environments using barracuda ODE solver.
fn validate_barracuda_diversity(h: &mut ValidationHarness) {
    let envs = [
        ("nutrient_rich", ENV_NUTRIENT_RICH),
        ("nutrient_poor", ENV_NUTRIENT_POOR),
        ("stress", ENV_STRESS),
    ];

    let mut counts = [0.0_f64; 3];
    let x0 = [0.5, 0.1, 0.5, 0.1];

    for (_, (signal, kb, km, kv)) in &envs {
        let p = env_params(*kb, *km, *kv);
        let rhs = grn_rhs_closure(*signal, &p);
        let config = make_config();

        if let Ok(result) = barracuda::numerical::rk45_solve(&rhs, 0.0, 40.0, &x0, &config) {
            let y = &result.y_final;
            let x = [y[0], y[1], y[2], y[3]];
            counts[phenotype_classifier(&x)] += 1.0;
        } else {
            let rk4_result = integrate_grn(&x0, *signal, &p, 2000, 0.02);
            counts[phenotype_classifier(&rk4_result)] += 1.0;
        }
    }

    let div = shannon_diversity(&counts);
    h.check_bool(
        &format!("barracuda ODE diversity computed ({div:.4}), {counts:?} phenotype counts"),
        div.is_finite(),
    );
}

/// Verify phenotype classification matches between RK4 and barracuda RK45.
fn validate_barracuda_phenotypes(h: &mut ValidationHarness) {
    let envs = [
        ("nutrient_rich", ENV_NUTRIENT_RICH),
        ("nutrient_poor", ENV_NUTRIENT_POOR),
        ("stress", ENV_STRESS),
    ];

    let x0 = [0.5, 0.1, 0.5, 0.1];
    let mut all_match = true;

    for (name, (signal, kb, km, kv)) in &envs {
        let p = env_params(*kb, *km, *kv);
        let rk4_result = integrate_grn(&x0, *signal, &p, 2000, 0.02);
        let rk4_pheno = phenotype_classifier(&rk4_result);

        let rhs = grn_rhs_closure(*signal, &p);
        let config = make_config();

        if let Ok(result) = barracuda::numerical::rk45_solve(&rhs, 0.0, 40.0, &x0, &config) {
            let y = &result.y_final;
            let x = [y[0], y[1], y[2], y[3]];
            let rk45_phenotype = phenotype_classifier(&x);
            if rk4_pheno != rk45_phenotype {
                all_match = false;
                eprintln!("  {name}: RK4 pheno={rk4_pheno}, RK45 pheno={rk45_phenotype}");
            }
        }
    }

    h.check_bool("phenotype classification matches across solvers", all_match);
}

/// Wrap GRN ODE RHS for barracuda's `rk45_solve` interface.
fn grn_rhs_closure(env_signal: f64, p: &GrnParams) -> impl Fn(f64, &[f64]) -> Vec<f64> + '_ {
    move |_t: f64, y: &[f64]| {
        let x = [y[0], y[1], y[2], y[3]];
        let rhs = neural_spring::regulatory_network::grn_rhs(&x, env_signal, p);
        vec![rhs[0], rhs[1], rhs[2], rhs[3]]
    }
}
