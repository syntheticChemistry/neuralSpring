// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: counterdiabatic driving of evolution (Paper 011).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/counterdiabatic/counterdiabatic_evolution.py`
//! Paper: Iram, Dolson et al. (2020) Nature Physics 17:135-142.
//! Command: `python3 control/counterdiabatic/counterdiabatic_evolution.py`
//! Result: 11/11 PASS (seed=42/99, N=5, K=2,3,4)

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::counterdiabatic::{
    NkLandscape, boltzmann_distribution, compute_cd_schedule, interpolated_fitness, kl_divergence,
    run_protocol_deterministic,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("counterdiabatic");

    // NK landscapes (N=5, K=2,3,4) matching Python seeds
    let n = 5;

    for k in [2, 3, 4] {
        let l0 = NkLandscape::new(n, k, 42);
        let l1 = NkLandscape::new(n, k, 99);
        let f0 = l0.all_fitnesses();
        let f1 = l1.all_fitnesses();

        h.check_bool(
            &format!("K={k}: landscapes have 2^N={} genotypes", 1 << n),
            f0.len() == (1 << n) && f1.len() == (1 << n),
        );

        // Boltzmann distributions should sum to 1
        let eq0 = boltzmann_distribution(&f0, 1.0);
        let sum0: f64 = eq0.iter().sum();
        h.check_abs(
            &format!("K={k}: Boltzmann(F0) sums to 1"),
            sum0,
            1.0,
            tolerances::EXACT_F64,
        );

        // Interpolation at s=0 returns F0, at s=1 returns F1
        let interp0 = interpolated_fitness(&f0, &f1, 0.0);
        let max_diff: f64 = interp0
            .iter()
            .zip(f0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        h.check_abs(
            &format!("K={k}: interp(s=0) == F0"),
            max_diff,
            0.0,
            tolerances::EXACT_F64,
        );

        // CD schedule endpoints
        let t = 200;
        let cd_sched = compute_cd_schedule(&f0, &f1, t, 1.0);
        h.check_bool(
            &format!("K={k}: CD schedule has {t} points"),
            cd_sched.len() == t,
        );

        // Run deterministic protocols
        let naive_sched: Vec<f64> = (0..t).map(|i| i as f64 / (t - 1) as f64).collect();
        let naive_r = run_protocol_deterministic(&f0, &f1, &naive_sched);
        let cd_r = run_protocol_deterministic(&f0, &f1, &cd_sched);

        // Core paper claim: CD reaches target closer than naive (or comparable)
        let cd_better = cd_r.final_dist < naive_r.final_dist;
        let cd_comparable =
            (cd_r.final_dist - naive_r.final_dist).abs() < tolerances::CD_COMPARABLE_DIST;
        h.check_bool(
            &format!(
                "K={k}: CD final_dist ({:.6}) <= naive ({:.6})",
                cd_r.final_dist, naive_r.final_dist
            ),
            cd_better || cd_comparable,
        );

        // Adiabaticity: CD mean KL <= naive mean KL (or within tolerance)
        let naive_mean_kl: f64 = naive_r.mean_kl.iter().sum::<f64>() / naive_r.mean_kl.len() as f64;
        let cd_mean_kl: f64 = cd_r.mean_kl.iter().sum::<f64>() / cd_r.mean_kl.len() as f64;
        let adiabatic = cd_mean_kl <= naive_mean_kl
            || (cd_mean_kl - naive_mean_kl) < tolerances::ADIABATIC_KL_GAP;
        h.check_bool(
            &format!("K={k}: CD adiabatic (KL {cd_mean_kl:.6} vs {naive_mean_kl:.6})"),
            adiabatic,
        );
    }

    // KL(p, p) should be ~0
    let p = vec![0.25, 0.25, 0.25, 0.25];
    h.check_abs(
        "KL(uniform, uniform) ≈ 0",
        kl_divergence(&p, &p),
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    h.finish();
}
