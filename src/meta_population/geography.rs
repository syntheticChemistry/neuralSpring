// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::rng::Rng;

/// Euclidean distance matrix from 2D coordinates.
#[must_use]
pub fn geographic_distance_matrix(coords: &[(f64, f64)]) -> Vec<f64> {
    let n = coords.len();
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i].0 - coords[j].0;
            let dy = coords[i].1 - coords[j].1;
            let d = (dx * dx + dy * dy).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

/// Pearson correlation between upper-triangle elements of two square matrices.
///
/// Extracts the upper triangle, then delegates to
/// `barracuda::stats::pearson_correlation` (absorbed from airSpring/groundSpring
/// hydrology metrics in `ToadStool` S64).
#[must_use]
pub fn matrix_correlation(a: &[f64], b: &[f64], n: usize) -> f64 {
    let (xs, ys): (Vec<f64>, Vec<f64>) = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (a[i * n + j], b[i * n + j])))
        .unzip();
    if xs.len() < 2 {
        return 0.0;
    }
    barracuda::stats::pearson_correlation(&xs, &ys).unwrap_or(0.0)
}

/// Mantel test: correlation between distance matrices with permutation p-value.
#[must_use]
pub fn mantel_test(
    dist_a: &[f64],
    dist_b: &[f64],
    n: usize,
    n_permutations: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let r_obs = matrix_correlation(dist_a, dist_b, n);
    let mut count_ge = 0_usize;
    let mut perm: Vec<usize> = (0..n).collect();

    for _ in 0..n_permutations {
        for i in (1..n).rev() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "permutation index modulo i+1 always fits in usize"
            )]
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let mut dist_b_perm = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                dist_b_perm[i * n + j] = dist_b[perm[i] * n + perm[j]];
            }
        }
        let r_perm = matrix_correlation(dist_a, &dist_b_perm, n);
        if r_perm >= r_obs {
            count_ge += 1;
        }
    }
    let p_value = (count_ge as f64 + 1.0) / (n_permutations as f64 + 1.0);
    (r_obs, p_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tolerances;

    #[test]
    fn geographic_distance_symmetric() {
        let coords = vec![(0.0, 0.0), (3.0, 4.0), (1.0, 1.0)];
        let dist = geographic_distance_matrix(&coords);
        for i in 0..3 {
            assert!(dist[i * 3 + i].abs() < tolerances::CROSS_LANGUAGE);
            for j in 0..3 {
                assert!((dist[i * 3 + j] - dist[j * 3 + i]).abs() < tolerances::CROSS_LANGUAGE);
            }
        }
        assert!((dist[1] - 5.0).abs() < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn matrix_correlation_perfect() {
        let a = vec![0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0];
        let r = matrix_correlation(&a, &a, 3);
        assert!(
            (r - 1.0).abs() < tolerances::CROSS_LANGUAGE,
            "self-correlation should be 1.0"
        );
    }

    #[test]
    fn mantel_test_produces_finite() {
        let mut rng = Rng::new(42);
        let n = 4;
        let a: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
        let b: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
        let (r, p) = mantel_test(&a, &b, n, 99, &mut rng);
        assert!(r.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }
}
