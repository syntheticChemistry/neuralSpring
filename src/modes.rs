// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(clippy::cast_precision_loss)]

//! MODES Toolbox: Metrics of Open-Ended Evolution.
//!
//! Port of `control/modes/modes_toolbox.py`.
//!
//! Reproduces metrics from:
//! Dolson, Vostinar, Wiser, Ofria (2019) "The MODES Toolbox: Measurements
//! of Open-Ended Dynamics in Evolving Systems" Artificial Life 25(1):50–73.
//! `doi:10.1162/artl_a_00280`
//!
//! Four metrics: Change, Novelty, Complexity, Ecology.
//!
//! ## `BarraCUDA` connection
//!
//! - Change metric: `barracuda::ops::SumReduceF64` (temporal difference)
//! - Novelty metric: `barracuda::ops::pairwise_distance` (Hamming between lineages)
//! - Complexity metric: `barracuda::ops::FusedMapReduceF64` (genome variance)
//! - Ecology metric: scalar reduction (Shannon entropy of type counts)

/// WGSL shader: pairwise L2 distance for novelty search.
///
/// One thread per upper-triangular pair, computes Euclidean distance
/// in feature space. Paper 012 (MODES novelty metric).
///
/// Absorption target: `barracuda::ops::pairwise_distance`.
/// Validated: `validate_gpu_modes` (15/15 PASS).
pub use neural_spring_forge::shaders::PAIRWISE_L2 as WGSL_PAIRWISE_L2;

/// Rate of novel type appearance over time.
///
/// `change[0] = 0`, `change[t] = lineage_counts[t] - lineage_counts[t-1]`.
#[must_use]
pub fn change_metric(lineage_counts: &[usize]) -> Vec<f64> {
    if lineage_counts.is_empty() {
        return Vec::new();
    }
    let mut change = vec![0.0; lineage_counts.len()];
    change[0] = 0.0;
    for t in 1..lineage_counts.len() {
        change[t] = lineage_counts[t] as f64 - lineage_counts[t - 1] as f64;
    }
    change
}

/// How different new types are from previously seen ones.
///
/// For each timestep, mean L2 distance from current features to all
/// previously seen features. `novelty[0] = 0`.
#[must_use]
pub fn novelty_metric(type_features: &[Vec<f64>]) -> Vec<f64> {
    let mut novelty = vec![0.0; type_features.len()];
    let mut seen: Vec<&[f64]> = Vec::new();

    for (t, features) in type_features.iter().enumerate() {
        if seen.is_empty() {
            novelty[t] = 0.0;
        } else {
            let mut sum_dist = 0.0;
            for s in &seen {
                let d = l2_distance(features, s);
                sum_dist += d;
            }
            novelty[t] = sum_dist / seen.len() as f64;
        }
        seen.push(features.as_slice());
    }
    novelty
}

/// L2 (Euclidean) distance between two feature vectors.
///
/// Delegates to `barracuda::dispatch::l2_distance_dispatch` (CPU path).
/// Used by `novelty_metric`. Exposed for GPU validation.
#[must_use]
pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    barracuda::dispatch::l2_distance_dispatch(a, b, None).unwrap_or_else(|_| {
        let n = a.len().min(b.len());
        a.iter()
            .take(n)
            .zip(b.iter().take(n))
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    })
}

/// Linear regression slope of complexity over time.
///
/// Delegates to `barracuda::stats::fit_linear` (absorbed from airSpring V009
/// in `ToadStool` S66). Returns (slope, increasing).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn complexity_metric(complexities: &[f64]) -> (f64, bool) {
    let n = complexities.len();
    if n < 2 {
        return (0.0, false);
    }
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    match barracuda::stats::fit_linear(&t, complexities) {
        Some(result) => {
            let slope = result.params[0];
            (slope, slope > 0.0)
        }
        None => (0.0, false),
    }
}

/// Shannon equitability at each timestep: `H/H_max`.
///
/// p = abd/sum(abd), H = -sum(p*ln(p)), `H_max` = ln(S).
/// Delegates to [`crate::primitives::shannon_equitability`].
#[must_use]
pub fn ecology_metric(abundances: &[Vec<f64>]) -> Vec<f64> {
    abundances
        .iter()
        .map(|abd| {
            let sum: f64 = abd.iter().sum();
            if sum <= 0.0 {
                return 0.0;
            }
            let freqs: Vec<f64> = abd.iter().map(|&x| x / sum).collect();
            crate::primitives::shannon_equitability(&freqs)
        })
        .collect()
}

/// Aggregate scores from all four MODES metrics.
#[derive(Debug, Clone)]
pub struct Scores {
    pub change_total: f64,
    pub change_mean: f64,
    pub novelty_mean: f64,
    pub novelty_final: f64,
    pub complexity_slope: f64,
    pub complexity_increasing: bool,
    pub ecology_mean: f64,
    pub ecology_final: f64,
}

/// Compute all four MODES metrics and aggregate scores.
#[must_use]
pub fn score_system(
    lineage_counts: &[usize],
    type_features: &[Vec<f64>],
    complexities: &[f64],
    abundances: &[Vec<f64>],
) -> Scores {
    let chg = change_metric(lineage_counts);
    let nov = novelty_metric(type_features);
    let (cpx_slope, cpx_inc) = complexity_metric(complexities);
    let eco = ecology_metric(abundances);

    let change_total: f64 = chg.iter().sum();
    let change_mean = if chg.is_empty() {
        0.0
    } else {
        chg.iter().sum::<f64>() / chg.len() as f64
    };

    let novelty_mean = if nov.is_empty() {
        0.0
    } else {
        nov.iter().sum::<f64>() / nov.len() as f64
    };
    let tail = nov.len().saturating_sub(20);
    let novelty_final = if tail == nov.len() {
        0.0
    } else {
        nov[tail..].iter().sum::<f64>() / (nov.len() - tail) as f64
    };

    let ecology_mean = if eco.is_empty() {
        0.0
    } else {
        eco.iter().sum::<f64>() / eco.len() as f64
    };
    let eco_tail = eco.len().saturating_sub(20);
    let ecology_final = if eco_tail == eco.len() {
        0.0
    } else {
        eco[eco_tail..].iter().sum::<f64>() / (eco.len() - eco_tail) as f64
    };

    Scores {
        change_total,
        change_mean,
        novelty_mean,
        novelty_final,
        complexity_slope: cpx_slope,
        complexity_increasing: cpx_inc,
        ecology_mean,
        ecology_final,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn change_monotonic_positive() {
        let counts = vec![1, 2, 3, 5, 8];
        let chg = change_metric(&counts);
        assert_relative_eq!(chg[0], 0.0, epsilon = 1e-10);
        assert!(chg[1..].iter().all(|&x| x >= 0.0));
        assert_relative_eq!(chg.iter().sum::<f64>(), 7.0, epsilon = 1e-10);
    }

    #[test]
    fn novelty_identical_zero() {
        let features = vec![vec![1.0, 2.0, 3.0]; 5];
        let nov = novelty_metric(&features);
        assert_relative_eq!(nov[0], 0.0, epsilon = 1e-10);
        for nov_i in nov.iter().skip(1).take(4) {
            assert_relative_eq!(*nov_i, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn complexity_increasing_positive_slope() {
        let cpx = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (slope, inc) = complexity_metric(&cpx);
        assert!(slope > 0.0);
        assert!(inc);
    }

    #[test]
    fn ecology_uniform_high() {
        let abd = vec![vec![0.25, 0.25, 0.25, 0.25]];
        let eco = ecology_metric(&abd);
        assert_relative_eq!(eco[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn ecology_skewed_low() {
        let abd = vec![vec![0.9, 0.05, 0.05]];
        let eco = ecology_metric(&abd);
        assert!(eco[0] < 0.5);
    }

    #[test]
    fn determinism() {
        let counts = vec![1, 3, 5, 7, 10];
        let features = vec![vec![1.0, 0.0], vec![1.1, 0.1], vec![1.2, 0.2]];
        let cpx = vec![1.0, 2.0, 3.0];
        let abd = vec![vec![0.5, 0.5], vec![0.3, 0.7], vec![0.25, 0.75]];

        let s1 = score_system(&counts, &features, &cpx, &abd);
        let s2 = score_system(&counts, &features, &cpx, &abd);
        assert_relative_eq!(s1.change_total, s2.change_total, epsilon = 1e-10);
        assert_relative_eq!(s1.novelty_mean, s2.novelty_mean, epsilon = 1e-10);
    }

    #[test]
    fn l2_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_relative_eq!(l2_distance(&a, &b), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn l2_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(l2_distance(&a, &a), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn ecology_zero_abundances() {
        let abd = vec![vec![0.0, 0.0, 0.0]];
        let eco = ecology_metric(&abd);
        assert_relative_eq!(eco[0], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn complexity_single_point() {
        let cpx = vec![5.0];
        let (slope, inc) = complexity_metric(&cpx);
        assert_relative_eq!(slope, 0.0, epsilon = 1e-15);
        assert!(!inc);
    }

    #[test]
    fn score_system_empty_inputs() {
        let s = score_system(&[], &[], &[], &[]);
        assert_relative_eq!(s.change_total, 0.0, epsilon = 1e-15);
        assert_relative_eq!(s.novelty_mean, 0.0, epsilon = 1e-15);
        assert_relative_eq!(s.ecology_mean, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn score_system_short_sequences() {
        let s = score_system(
            &[1, 2],
            &[vec![1.0], vec![2.0]],
            &[1.0, 2.0],
            &[vec![0.5, 0.5], vec![0.3, 0.7]],
        );
        assert!(s.change_total >= 0.0);
        assert!(s.novelty_final >= 0.0);
        assert!(s.ecology_final >= 0.0);
    }
}
