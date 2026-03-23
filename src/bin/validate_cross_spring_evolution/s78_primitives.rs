// SPDX-License-Identifier: AGPL-3.0-or-later

//! S78 cross-spring rewires: metrics, Shannon, Hill, L2, complexity (linear fit).

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

pub fn validate_rewired_mae_s78(h: &mut ValidationHarness) {
    let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = [1.1, 2.2, 2.9, 4.1, 4.8];

    let result = neural_spring::metrics::mae(&y_true, &y_pred);
    let reference = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / y_true.len() as f64;

    h.check_abs(
        "S78: mae delegates to barracuda::stats::mae",
        result,
        reference,
        tolerances::EXACT_F64,
    );

    let perfect = neural_spring::metrics::mae(&y_true, &y_true);
    h.check_abs(
        "S78: mae(identical) = 0",
        perfect,
        0.0,
        tolerances::EXACT_F64,
    );
}

pub fn validate_rewired_shannon_from_frequencies_s78(h: &mut ValidationHarness) {
    let uniform4 = [0.25, 0.25, 0.25, 0.25];
    let expected = -(4.0 * 0.25 * 0.25_f64.ln());

    let result = neural_spring::primitives::shannon_entropy(&uniform4);
    h.check_abs(
        "S78: shannon_entropy delegates to barracuda::stats::shannon_from_frequencies",
        result,
        expected,
        tolerances::EXACT_F64,
    );

    let degenerate = [1.0, 0.0, 0.0];
    let result_d = neural_spring::primitives::shannon_entropy(&degenerate);
    h.check_abs(
        "S78: shannon_entropy degenerate = 0",
        result_d,
        0.0,
        tolerances::EXACT_F64,
    );
}

pub fn validate_rewired_hill_s78(h: &mut ValidationHarness) {
    let x = 5.0;
    let k = 3.0;
    let n = 2.0;
    let amp = 10.0;

    let act = neural_spring::primitives::hill_activation(x, amp, k, n);
    let xn = x.powf(n);
    let kn = k.powf(n);
    let expected_act = amp * xn / (kn + xn);
    h.check_abs(
        "S78: hill_activation delegates to barracuda::stats::hill",
        act,
        expected_act,
        tolerances::EXACT_F64,
    );

    let rep = neural_spring::primitives::hill_repression(x, amp, k, n);
    let expected_rep = amp * kn / (kn + xn);
    h.check_abs(
        "S78: hill_repression = amp * (1 - hill)",
        rep,
        expected_rep,
        tolerances::EXACT_F64,
    );

    h.check_abs(
        "S78: hill_activation(x<=0) = 0",
        neural_spring::primitives::hill_activation(0.0, amp, k, n),
        0.0,
        tolerances::EXACT_F64,
    );

    h.check_abs(
        "S78: hill_repression(x<=0) = amplitude",
        neural_spring::primitives::hill_repression(0.0, amp, k, n),
        amp,
        tolerances::EXACT_F64,
    );
}

pub fn validate_rewired_l2_distance_s78(h: &mut ValidationHarness) {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [1.1, 2.2, 2.8, 4.1, 5.3];

    let result = neural_spring::modes::l2_distance(&a, &b);
    let expected: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();

    h.check_abs(
        "S78: l2_distance delegates to barracuda::dispatch::l2_distance_dispatch",
        result,
        expected,
        tolerances::EXACT_F64,
    );

    let zero = neural_spring::modes::l2_distance(&a, &a);
    h.check_abs(
        "S78: l2_distance(identical) = 0",
        zero,
        0.0,
        tolerances::EXACT_F64,
    );
}

pub fn validate_rewired_complexity_metric_s78(h: &mut ValidationHarness) {
    let increasing = [1.0, 2.0, 3.0, 4.0, 5.0];
    let (slope, is_increasing) = neural_spring::modes::complexity_metric(&increasing);

    h.check_bool(
        "S78: complexity_metric(increasing) is_increasing=true",
        is_increasing,
    );
    h.check_abs(
        "S78: complexity_metric slope ≈ 1.0 for [1,2,3,4,5]",
        slope,
        1.0,
        tolerances::CROSS_LANGUAGE,
    );

    let constant = [3.0, 3.0, 3.0, 3.0];
    let (slope_c, _) = neural_spring::modes::complexity_metric(&constant);
    h.check_abs(
        "S78: complexity_metric(constant) slope = 0",
        slope_c,
        0.0,
        tolerances::EXACT_F64,
    );
}
