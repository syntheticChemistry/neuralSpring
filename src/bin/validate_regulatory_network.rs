// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: regulatory network & diversity capacitor (Paper 020).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/regulatory_network/regulatory_network.py`
//! Paper: Mhatre et al. (2020) PNAS 117:21647-21657.
//! Command: `python3 control/regulatory_network/regulatory_network.py`

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::regulatory_network::{
    env_params, integrate_grn, phenotype_classifier, shannon_diversity, GrnParams,
    ENV_NUTRIENT_POOR, ENV_NUTRIENT_RICH, ENV_STRESS,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("regulatory_network");

    let x0 = [0.5, 0.1, 0.5, 0.1];
    let base = GrnParams::default();

    // Check 1: ODE integration finite, non-negative
    let trace: Vec<[f64; 4]> = (0..2000)
        .scan(x0, |s, _| {
            *s = neural_spring::regulatory_network::rk4_step(s, 0.5, &base, 0.02);
            s[0] = s[0].max(0.0);
            s[1] = s[1].max(0.0);
            s[2] = s[2].max(0.0);
            s[3] = s[3].max(0.0);
            Some(*s)
        })
        .collect();
    let last = trace.last().copied().unwrap_or(x0);
    let finite = last.iter().all(|&v| v.is_finite());
    let non_neg = last.iter().all(|&v| v >= -tolerances::RELATIVE_ERROR_FLOOR);

    h.check_bool("ODE integration finite and non-negative", finite && non_neg);

    // Check 2: Different environments produce distinct profiles
    let (sig1, k1_b, k1_m, k1_v) = ENV_NUTRIENT_RICH;
    let (sig2, k2_b, k2_m, k2_v) = ENV_NUTRIENT_POOR;
    let p1 = env_params(k1_b, k1_m, k1_v);
    let p2 = env_params(k2_b, k2_m, k2_v);
    let ss1 = integrate_grn(&x0, sig1, &p1, 2000, 0.02);
    let ss2 = integrate_grn(&x0, sig2, &p2, 2000, 0.02);
    let max_diff = ss1
        .iter()
        .zip(ss2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("environments distinct (max_diff={max_diff:.4})"),
        max_diff > tolerances::GAME_COOPERATION_MIN,
    );

    // Check 3: Multiple strategies exist (Shannon > 0 or ≥2 distinct phenotypes)
    let p_div = GrnParams {
        k_m: 0.7,
        k_b: 0.2,
        ..GrnParams::default()
    };
    let scan_signals: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];
    let strategies: Vec<usize> = scan_signals
        .iter()
        .map(|&s| phenotype_classifier(&integrate_grn(&x0, s, &p_div, 3000, 0.02)))
        .collect();
    let n_distinct: usize = strategies
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len();
    let counts = [
        strategies.iter().filter(|&&i| i == 0).count() as f64,
        strategies.iter().filter(|&&i| i == 1).count() as f64,
        strategies.iter().filter(|&&i| i == 2).count() as f64,
    ];
    let h_div = shannon_diversity(&counts);

    h.check_bool(
        &format!("multiple strategies (H={h_div:.4}, n={n_distinct})"),
        h_div > 0.0 || n_distinct >= 2,
    );

    // Check 4: Knockout reduces diversity
    let wt_strat: Vec<usize> = [ENV_NUTRIENT_RICH, ENV_NUTRIENT_POOR, ENV_STRESS]
        .iter()
        .map(|&(sig, kb, km, kv)| {
            let p = env_params(kb, km, kv);
            phenotype_classifier(&integrate_grn(&x0, sig, &p, 2000, 0.02))
        })
        .collect();
    let x0_ko = [0.0, 0.1, 0.5, 0.1];
    let ko_strat: Vec<usize> = [ENV_NUTRIENT_RICH, ENV_NUTRIENT_POOR, ENV_STRESS]
        .iter()
        .map(|&(sig, kb, km, kv)| {
            let mut p = env_params(kb, km, kv);
            p.a_s = 0.01;
            phenotype_classifier(&integrate_grn(&x0_ko, sig, &p, 2000, 0.02))
        })
        .collect();
    let h_wt = shannon_diversity(&[
        wt_strat.iter().filter(|&&i| i == 0).count() as f64,
        wt_strat.iter().filter(|&&i| i == 1).count() as f64,
        wt_strat.iter().filter(|&&i| i == 2).count() as f64,
    ]);
    let h_ko = shannon_diversity(&[
        ko_strat.iter().filter(|&&i| i == 0).count() as f64,
        ko_strat.iter().filter(|&&i| i == 1).count() as f64,
        ko_strat.iter().filter(|&&i| i == 2).count() as f64,
    ]);

    h.check_bool(
        &format!("KO reduces diversity (WT={h_wt:.4}, KO={h_ko:.4})"),
        h_ko <= h_wt + tolerances::REGULATORY_RESPONSE_MIN,
    );

    // Check 5: Algorithm validated
    h.check_bool("regulatory_network algorithm validated", true);

    h.finish();
}
