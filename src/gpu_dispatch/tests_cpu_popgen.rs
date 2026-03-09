// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path population genetics and pangenome tests.

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

// ── Population genetics ─────────────────────────────────────

#[test]
fn cpu_allele_frequencies() {
    let d = cpu();
    let pop = vec![2.0, 0.0, 0.0, 2.0];
    let freq = d.allele_frequencies(&pop, 2, 2);
    assert_eq!(freq.len(), 2);
    assert!((freq[0] - 0.5).abs() < tolerances::EXACT_F64);
    assert!((freq[1] - 0.5).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_nucleotide_diversity() {
    let d = cpu();
    let pop = vec![0.0, 1.0, 1.0, 0.0];
    let pi = d.nucleotide_diversity(&pop, 2, 2);
    assert!(pi >= 0.0);
}

#[test]
fn cpu_matrix_correlation() {
    let d = cpu();
    #[rustfmt::skip]
    let a = vec![
        0.0, 1.0, 2.0,
        1.0, 0.0, 3.0,
        2.0, 3.0, 0.0,
    ];
    let r = d.matrix_correlation(&a, &a, 3);
    assert!(
        (r - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "self-correlation = 1.0"
    );
}

#[test]
fn cpu_geographic_distances() {
    let d = cpu();
    let coords = vec![(0.0, 0.0), (3.0, 4.0)];
    let dist = d.geographic_distances(&coords);
    assert_eq!(dist.len(), 4);
    assert!((dist[0] - 0.0).abs() < tolerances::EXACT_F64);
    assert!((dist[1] - 5.0).abs() < tolerances::EXACT_F64);
    assert!((dist[3] - 0.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_thermal_diversity_correlation() {
    let d = cpu();
    let r = d.thermal_diversity_correlation(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
    assert!(
        (r - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "perfect linear → r≈1"
    );
}

#[test]
fn cpu_thermal_diversity_short() {
    let d = cpu();
    let r = d.thermal_diversity_correlation(&[1.0], &[10.0]);
    assert!((r - 0.0).abs() < tolerances::ZERO_DETECTION, "n<2 → 0");
}

// ── Inter-population AF variance ────────────────────────────

#[test]
fn cpu_inter_population_af_variance_basic() {
    let d = cpu();
    let population_a = vec![2.0, 0.0, 0.0, 2.0];
    let population_b = vec![0.0, 2.0, 2.0, 0.0];
    let populations: Vec<&[f64]> = vec![&population_a, &population_b];
    let var = d.inter_population_af_variance(&populations, &[2, 2], 2);
    assert!(var >= 0.0, "AF variance must be non-negative");
}

// ── FST ──────────────────────────────────────────────────────

#[test]
fn cpu_pairwise_fst_divergent() {
    let d = cpu();
    let pop_a = vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
    let pop_b = vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
    let fst = d.pairwise_fst(&pop_a, 5, &pop_b, 5, 2);
    assert!(fst.is_finite(), "FST must be finite");
}

#[test]
fn cpu_global_fst_two_pops() {
    let d = cpu();
    let pop1 = vec![2.0, 0.0, 2.0, 0.0];
    let pop2 = vec![0.0, 2.0, 0.0, 2.0];
    let fst = d.global_fst(&[pop1, pop2], &[2, 2], 2);
    assert!(fst.is_finite(), "FST must be finite");
}

// ── Pangenome selection ─────────────────────────────────────

#[test]
fn cpu_spectrum_chi_squared() {
    let d = cpu();
    let obs = vec![10.0, 20.0, 30.0];
    let frac = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let chi2 = d.spectrum_chi_squared(&obs, &frac);
    assert!(chi2 >= 0.0);
}

#[test]
fn cpu_selection_coefficient() {
    let d = cpu();
    let obs = vec![10.0, 20.0, 30.0];
    let neutral = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let s = d.selection_coefficient(&obs, &neutral);
    assert!(s.is_finite());
}
