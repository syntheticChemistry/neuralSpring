// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(clippy::suboptimal_flops)]

//! Regulatory network and diversity capacitor (Paper 020).
//!
//! Port of `control/regulatory_network/regulatory_network.py`.
//!
//! Mhatre et al. (2020) PNAS 117:21647-21657:
//! "One gene, multiple ecological strategies: a biofilm regulator is a
//!  capacitor for sustainable diversity"
//!
//! Core thesis: `SasA` acts as a capacitor for diversity — one regulatory
//! element producing multiple phenotypes (biofilm, motility, virulence).
//!
//! ## `BarraCUDA` connection
//!
//! - Hill activation/repression: elementwise ops (`barracuda::ops::elementwise`)
//! - ODE integration (RK4): `barracuda::numerical::rk45_solve` or GPU `rk4_parallel.wgsl`
//! - Steady-state phenotype scan: batch parallel ODE over parameter grid

use crate::primitives;

/// GRN parameters. State: [sasa, biofilm, motility, virulence].
#[derive(Debug, Clone)]
pub struct GrnParams {
    pub n: f64,
    pub a_s: f64,
    pub d_s: f64,
    pub k_b: f64,
    pub k_m: f64,
    pub k_v: f64,
    pub a_b: f64,
    pub d_b: f64,
    pub a_m: f64,
    pub d_m: f64,
    pub a_v: f64,
    pub d_v: f64,
}

impl Default for GrnParams {
    fn default() -> Self {
        Self {
            n: 2.0,
            a_s: 1.0,
            d_s: 0.5,
            k_b: 0.35,
            k_m: 0.4,
            k_v: 0.7,
            a_b: 1.2,
            d_b: 0.4,
            a_m: 1.0,
            d_m: 0.5,
            a_v: 0.8,
            d_v: 0.5,
        }
    }
}

/// RHS of GRN ODE: d\[state\]/dt.
#[must_use]
pub fn grn_rhs(x: &[f64; 4], env_signal: f64, p: &GrnParams) -> [f64; 4] {
    let [sasa, bio, mot, vir] = *x;
    let dsasa = p.a_s * env_signal / (0.5 + env_signal) - p.d_s * sasa;
    let dbio = primitives::hill_activation(sasa, p.a_b, p.k_b, p.n) - p.d_b * bio;
    let dmot = primitives::hill_repression(sasa, p.a_m, p.k_m, p.n) - p.d_m * mot;
    let dvir = primitives::hill_activation(sasa, p.a_v, p.k_v, p.n) - p.d_v * vir;
    [dsasa, dbio, dmot, dvir]
}

/// Single RK4 step (delegates to [`crate::primitives::rk4_step`]).
#[must_use]
pub fn rk4_step(x: &[f64; 4], env_signal: f64, p: &GrnParams, dt: f64) -> [f64; 4] {
    primitives::rk4_step(x, dt, |y| grn_rhs(y, env_signal, p))
}

/// Integrate GRN ODE to near steady state.
#[must_use]
pub fn integrate_grn(
    x0: &[f64; 4],
    env_signal: f64,
    p: &GrnParams,
    n_steps: usize,
    dt: f64,
) -> [f64; 4] {
    let mut x = *x0;
    for _ in 0..n_steps {
        x = rk4_step(&x, env_signal, p, dt);
        x[0] = x[0].max(0.0);
        x[1] = x[1].max(0.0);
        x[2] = x[2].max(0.0);
        x[3] = x[3].max(0.0);
    }
    x
}

/// Classify dominant strategy: 0=biofilm, 1=motility, 2=virulence.
#[must_use]
pub fn phenotype_classifier(x: &[f64; 4]) -> usize {
    let bio = x[1];
    let mot = x[2];
    let vir = x[3];
    let m = bio.max(mot).max(vir);
    if m <= 0.0 {
        return 0;
    }
    if bio >= m - 1e-10 {
        return 0;
    }
    if mot >= m - 1e-10 {
        return 1;
    }
    2
}

/// Shannon diversity H = -sum(p * ln(p)) for p > 0.
///
/// Delegates to [`crate::primitives::shannon_entropy_from_counts`].
#[must_use]
pub fn shannon_diversity(counts: &[f64]) -> f64 {
    primitives::shannon_entropy_from_counts(counts)
}

/// Environment configurations (signal, `K_b`, `K_m`, `K_v`).
pub const ENV_NUTRIENT_RICH: (f64, f64, f64, f64) = (0.9, 0.3, 0.5, 0.8);
pub const ENV_NUTRIENT_POOR: (f64, f64, f64, f64) = (0.2, 0.4, 0.3, 0.9);
pub const ENV_STRESS: (f64, f64, f64, f64) = (0.6, 0.35, 0.4, 0.5);

/// Build params for an environment (`K_b`, `K_m`, `K_v` from env).
#[must_use]
pub fn env_params(k_b: f64, k_m: f64, k_v: f64) -> GrnParams {
    GrnParams {
        k_b,
        k_m,
        k_v,
        ..GrnParams::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hill_activation_monotonic() {
        let a = primitives::hill_activation(0.0, 1.0, 0.5, 2.0);
        let b = primitives::hill_activation(0.5, 1.0, 0.5, 2.0);
        let c = primitives::hill_activation(1.0, 1.0, 0.5, 2.0);
        assert!(a < b && b < c, "activation should be monotonic");
    }

    #[test]
    fn hill_repression_decreasing() {
        let a = primitives::hill_repression(0.1, 1.0, 0.5, 2.0);
        let b = primitives::hill_repression(1.0, 1.0, 0.5, 2.0);
        assert!(a > b, "repression should decrease with x");
    }

    #[test]
    fn integrate_finite_nonneg() {
        let p = GrnParams::default();
        let x0 = [0.5, 0.1, 0.5, 0.1];
        let x = integrate_grn(&x0, 0.5, &p, 2000, 0.02);
        assert!(x.iter().all(|&v| v.is_finite() && v >= -1e-10));
    }

    #[test]
    fn phenotype_classifier_consistency() {
        let x = [1.0, 2.0, 0.5, 0.3];
        assert_eq!(phenotype_classifier(&x), 0);
        let x = [1.0, 0.2, 2.0, 0.3];
        assert_eq!(phenotype_classifier(&x), 1);
        let x = [1.0, 0.2, 0.1, 2.0];
        assert_eq!(phenotype_classifier(&x), 2);
    }

    #[test]
    fn shannon_diversity_nonzero() {
        let c = [1.0, 1.0, 1.0];
        assert!(shannon_diversity(&c) > 0.0);
        let c = [3.0, 0.0, 0.0];
        assert!(shannon_diversity(&c).abs() < 1e-10);
    }
}
