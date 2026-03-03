// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: MODES toolbox metrics (Paper 012).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/modes/modes_toolbox.py`
//! Paper: Dolson et al. (2019) Artificial Life 25(1):50-73.
//! Command: `python3 control/modes/modes_toolbox.py`
//! Result: 9/9 PASS (seed=42, open/closed/NK systems)

use neural_spring::modes::{
    change_metric, complexity_metric, ecology_metric, novelty_metric, score_system,
};
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;

fn generate_open_system(n_steps: usize, n_features: usize, seed: u64) -> SystemData {
    let mut rng = Rng::new(seed);
    let mut lineage_counts = Vec::new();
    let mut type_features = Vec::new();
    let mut complexities = Vec::new();
    let mut abundances = Vec::new();

    let mut current: Vec<f64> = (0..n_features).map(|_| rng.normal()).collect();
    let mut n_types_total = 1_usize;

    for _ in 0..n_steps {
        for v in &mut current {
            *v += rng.normal_params(0.0, 0.3) + 0.01;
        }
        if rng.uniform() < 0.3 {
            n_types_total += 1;
        }
        lineage_counts.push(n_types_total);
        type_features.push(current.clone());
        let norm: f64 = current.iter().map(|x| x * x).sum::<f64>();
        complexities.push(norm.sqrt());

        let n_alive = n_types_total.min(20);
        let abd: Vec<f64> = (0..n_alive).map(|_| rng.uniform() + 0.1).collect();
        let s: f64 = abd.iter().sum();
        abundances.push(abd.iter().map(|x| x / s).collect());
    }
    SystemData {
        lineage_counts,
        type_features,
        complexities,
        abundances,
    }
}

fn generate_closed_system(n_steps: usize, n_features: usize, seed: u64) -> SystemData {
    let mut rng = Rng::new(seed);
    let target: Vec<f64> = (0..n_features).map(|_| rng.normal()).collect();
    let mut current: Vec<f64> = (0..n_features)
        .map(|_| rng.normal_params(0.0, 5.0))
        .collect();

    let mut lineage_counts = Vec::new();
    let mut type_features = Vec::new();
    let mut complexities = Vec::new();
    let mut abundances = Vec::new();

    for _ in 0..n_steps {
        for (c, t) in current.iter_mut().zip(target.iter()) {
            *c = 0.95f64.mul_add(*c, 0.05 * t) + rng.normal_params(0.0, 0.01);
        }
        lineage_counts.push(3);
        type_features.push(current.clone());
        let norm: f64 = current.iter().map(|x| x * x).sum::<f64>();
        complexities.push(norm.sqrt());
        abundances.push(vec![0.8, 0.15, 0.05]);
    }
    SystemData {
        lineage_counts,
        type_features,
        complexities,
        abundances,
    }
}

struct SystemData {
    lineage_counts: Vec<usize>,
    type_features: Vec<Vec<f64>>,
    complexities: Vec<f64>,
    abundances: Vec<Vec<f64>>,
}

fn main() {
    let mut h = ValidationHarness::new("modes");

    let open = generate_open_system(200, 10, 42);
    let closed = generate_closed_system(200, 10, 42);

    let open_scores = score_system(
        &open.lineage_counts,
        &open.type_features,
        &open.complexities,
        &open.abundances,
    );
    let closed_scores = score_system(
        &closed.lineage_counts,
        &closed.type_features,
        &closed.complexities,
        &closed.abundances,
    );

    // Core paper claim: open-ended > closed on all four metrics
    h.check_bool(
        &format!(
            "change_total: open ({:.4}) > closed ({:.4})",
            open_scores.change_total, closed_scores.change_total
        ),
        open_scores.change_total > closed_scores.change_total,
    );

    h.check_bool(
        &format!(
            "novelty_mean: open ({:.4}) > closed ({:.4})",
            open_scores.novelty_mean, closed_scores.novelty_mean
        ),
        open_scores.novelty_mean > closed_scores.novelty_mean,
    );

    h.check_bool(
        &format!(
            "complexity_slope: open ({:.4}) > closed ({:.4})",
            open_scores.complexity_slope, closed_scores.complexity_slope
        ),
        open_scores.complexity_slope > closed_scores.complexity_slope,
    );

    h.check_bool(
        &format!(
            "ecology_mean: open ({:.4}) > closed ({:.4})",
            open_scores.ecology_mean, closed_scores.ecology_mean
        ),
        open_scores.ecology_mean > closed_scores.ecology_mean,
    );

    // Metric sanity checks
    let chg = change_metric(&open.lineage_counts);
    h.check_bool(
        "change_metric has correct length",
        chg.len() == open.lineage_counts.len(),
    );

    let nov = novelty_metric(&open.type_features);
    h.check_bool("novelty_metric non-negative", nov.iter().all(|&v| v >= 0.0));

    let (slope, increasing) = complexity_metric(&open.complexities);
    h.check_bool(
        &format!("open complexity increasing (slope={slope:.4})"),
        increasing,
    );

    let eco = ecology_metric(&open.abundances);
    h.check_bool(
        "ecology values in [0, 1]",
        eco.iter().all(|&v| (0.0..=1.0).contains(&v)),
    );

    // All metrics discriminate
    let all_discriminate = open_scores.change_total > closed_scores.change_total
        && open_scores.novelty_mean > closed_scores.novelty_mean
        && open_scores.ecology_mean > closed_scores.ecology_mean;
    h.check_bool(
        "all metrics discriminate open from closed",
        all_discriminate,
    );

    h.finish();
}
