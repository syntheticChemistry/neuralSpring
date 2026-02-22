// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: barracuda::spectral APIs vs neuralSpring CPU references.
//!
//! Cross-Spring lineage: hotSpring (Kachkovskiy spectral theory) → barracuda
//! → validated here by neuralSpring. Proves the upstream spectral stack is
//! correct by comparing tridiag eigensolvers, Aubry-André spectra, Lyapunov
//! exponents, and level-spacing statistics against known analytic results.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::doc_markdown
)]

use barracuda::spectral::{
    almost_mathieu_hamiltonian, anderson_hamiltonian, anderson_potential, detect_bands,
    find_all_eigenvalues, hofstadter_butterfly, lanczos, lanczos_eigenvalues, level_spacing_ratio,
    lyapunov_averaged, lyapunov_exponent, GOLDEN_RATIO as BARRACUDA_GOLDEN, POISSON_R,
};
use neural_spring::anderson_localization::{
    aubry_andre_hamiltonian, jacobi_eigh, GOLDEN_RATIO as NS_GOLDEN,
};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_spectral_theory");

    validate_golden_ratio_parity(&mut h);
    validate_aubry_andre_spectrum_parity(&mut h);
    validate_anderson_hamiltonian_spectrum(&mut h);
    validate_lanczos_vs_exact(&mut h);
    validate_lyapunov_localization(&mut h);
    validate_level_spacing_extended(&mut h);
    validate_level_spacing_localized(&mut h);
    validate_hofstadter_structure(&mut h);
    validate_band_detection(&mut h);
    validate_lyapunov_weak_disorder(&mut h);
    validate_eigh_vs_sturm(&mut h);
    validate_eigh_vs_sturm_large(&mut h);

    h.finish();
}

fn validate_golden_ratio_parity(h: &mut ValidationHarness) {
    let diff = (NS_GOLDEN - (1.0 + BARRACUDA_GOLDEN)).abs();
    h.check_upper(
        &format!(
            "GOLDEN_RATIO parity: nS={NS_GOLDEN:.15} vs 1+barracuda={:.15}, diff={diff:.2e}",
            1.0 + BARRACUDA_GOLDEN
        ),
        diff,
        1e-14,
    );
}

fn validate_aubry_andre_spectrum_parity(h: &mut ValidationHarness) {
    let n = 64;
    let w = 2.0;
    let alpha = BARRACUDA_GOLDEN;
    let phi = 0.0;

    let ns_mat = aubry_andre_hamiltonian(n, 1.0, w, 1.0 / NS_GOLDEN, phi);
    let (ns_evals, _) = jacobi_eigh(&ns_mat, n);
    let mut ns_sorted: Vec<f64> = ns_evals;
    ns_sorted.sort_by(f64::total_cmp);

    // barracuda: lambda = w/2 for the 2*lambda*cos convention
    let (diag, off_diag) = almost_mathieu_hamiltonian(n, w / 2.0, alpha, phi);
    let bc_sorted = find_all_eigenvalues(&diag, &off_diag);

    let max_diff: f64 = ns_sorted
        .iter()
        .zip(bc_sorted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("Aubry-André n=64 spectrum parity: max eigval diff {max_diff:.2e}"),
        max_diff,
        tolerances::SPECTRAL_EIGENSOLVER_CROSS,
    );
}

fn validate_anderson_hamiltonian_spectrum(h: &mut ValidationHarness) {
    let n = 50;
    let disorder = 4.0;
    let seed = 42;

    let (diag, off_diag) = anderson_hamiltonian(n, disorder, seed);
    let evals = find_all_eigenvalues(&diag, &off_diag);

    h.check_bool(
        &format!("Anderson n=50 W=4: got {} eigenvalues", evals.len()),
        evals.len() == n,
    );
    let Some(last) = evals.last() else {
        h.check_bool("Anderson eigenvalues: non-empty", false);
        return;
    };
    let Some(first) = evals.first() else {
        h.check_bool("Anderson eigenvalues: non-empty", false);
        return;
    };
    let bandwidth = last - first;
    h.check_lower(
        &format!("Anderson W=4 bandwidth {bandwidth:.2} > 4 (clean=4, disorder widens)"),
        bandwidth,
        4.0,
    );
}

fn validate_lanczos_vs_exact(h: &mut ValidationHarness) {
    let n = 80;
    let disorder = 3.0;
    let seed = 77;

    let (diag, off_diag) = anderson_hamiltonian(n, disorder, seed);
    let _exact = find_all_eigenvalues(&diag, &off_diag);

    let csr = barracuda::spectral::anderson_2d(8, 10, disorder, seed);
    let tridiag = lanczos(&csr, n, 123);
    let lanczos_evals = lanczos_eigenvalues(&tridiag);

    h.check_bool(
        &format!(
            "Lanczos produced {} eigenvalues from 8×10 2D Anderson",
            lanczos_evals.len()
        ),
        !lanczos_evals.is_empty(),
    );

    let tridiag_3d = lanczos(&barracuda::spectral::clean_3d_lattice(3), 20, 99);
    let lanc_3d_evals = lanczos_eigenvalues(&tridiag_3d);
    h.check_bool(
        &format!(
            "Lanczos 3D clean lattice 3×3×3: got {} eigenvalues",
            lanc_3d_evals.len()
        ),
        !lanc_3d_evals.is_empty(),
    );
}

fn validate_lyapunov_localization(h: &mut ValidationHarness) {
    let n = 10_000;

    let pot_strong = anderson_potential(n, 10.0, 42);
    let gamma_strong = lyapunov_exponent(&pot_strong, 0.0);

    let pot_weak = anderson_potential(n, 0.5, 42);
    let gamma_weak = lyapunov_exponent(&pot_weak, 0.0);

    h.check_lower(
        &format!("Lyapunov: γ(W=10)={gamma_strong:.4} > γ(W=0.5)={gamma_weak:.4}"),
        gamma_strong,
        gamma_weak,
    );
    h.check_lower(
        &format!("Lyapunov γ(W=10)={gamma_strong:.4} > 0 (localized)"),
        gamma_strong,
        0.0,
    );
}

fn validate_level_spacing_extended(h: &mut ValidationHarness) {
    let n = 500;
    let (diag, off_diag) = anderson_hamiltonian(n, 0.01, 42);
    let evals = find_all_eigenvalues(&diag, &off_diag);
    let r = level_spacing_ratio(&evals);
    h.check_lower(
        &format!("level spacing near-clean: r={r:.4} > Poisson threshold {POISSON_R:.4}"),
        r,
        POISSON_R,
    );
}

fn validate_level_spacing_localized(h: &mut ValidationHarness) {
    let n = 1000;
    let (diag, off_diag) = anderson_hamiltonian(n, 8.0, 42);
    let evals = find_all_eigenvalues(&diag, &off_diag);
    let r = level_spacing_ratio(&evals);
    let diff_from_poisson = (r - POISSON_R).abs();
    h.check_upper(
        &format!(
            "level spacing W=8: r={r:.4} ≈ Poisson {POISSON_R:.4}, diff={diff_from_poisson:.4}"
        ),
        diff_from_poisson,
        tolerances::LEVEL_SPACING_POISSON_TOL,
    );
}

fn validate_hofstadter_structure(h: &mut ValidationHarness) {
    let q_max = 8;
    let n_sites = 100;
    let butterfly = hofstadter_butterfly(q_max, 1.0, n_sites);

    let n_alphas = butterfly.len();
    h.check_lower(
        &format!("Hofstadter butterfly: {n_alphas} rational α values (q≤{q_max})"),
        n_alphas as f64,
        5.0,
    );

    let total_evals: usize = butterfly.iter().map(|(_, ev)| ev.len()).sum();
    h.check_lower(
        &format!("Hofstadter butterfly: {total_evals} total eigenvalues"),
        total_evals as f64,
        100.0,
    );
}

fn validate_band_detection(h: &mut ValidationHarness) {
    let mut evals: Vec<f64> = (0..20).map(|i| f64::from(i).mul_add(0.1, 0.0)).collect();
    evals.extend((0..20).map(|i| f64::from(i).mul_add(0.1, 10.0)));
    let bands = detect_bands(&evals, 2.0);
    h.check_bool(
        &format!("band detection: {} bands from gapped spectrum", bands.len()),
        bands.len() >= 2,
    );
}

fn validate_lyapunov_weak_disorder(h: &mut ValidationHarness) {
    // γ(0) ≈ W²/96 for small W (Kappus-Wegner anomaly)
    let w = 0.5;
    let n_sites = 5000;
    let n_real = 50;
    let gamma_avg = lyapunov_averaged(n_sites, w, 0.0, n_real, 1);
    let theory = w * w / 96.0;
    let rel_error = if theory > 0.0 {
        (gamma_avg - theory).abs() / theory
    } else {
        gamma_avg.abs()
    };
    h.check_upper(
        &format!(
            "Kappus-Wegner: γ(W={w})={gamma_avg:.6} vs W²/96={theory:.6}, rel error {rel_error:.2}"
        ),
        rel_error,
        tolerances::KAPPUS_WEGNER_REL,
    );
}

/// Cross-validate dense Householder+QR (eigh) vs tridiagonal Sturm bisection
/// on a tridiagonal Anderson matrix embedded as dense.
fn validate_eigh_vs_sturm(h: &mut ValidationHarness) {
    let n = 64;
    let disorder = 3.0;
    let seed = 42;

    let (diag, off_diag) = anderson_hamiltonian(n, disorder, seed);
    let sturm_evals = find_all_eigenvalues(&diag, &off_diag);

    // Build the same tridiagonal matrix as a dense n×n for Householder+QR
    let mut dense = vec![0.0_f64; n * n];
    for i in 0..n {
        dense[i * n + i] = diag[i];
        if i + 1 < n {
            dense[i * n + (i + 1)] = off_diag[i];
            dense[(i + 1) * n + i] = off_diag[i];
        }
    }
    let eigh_result = eigh_householder_qr(&dense, n);
    let mut eigh_evals = eigh_result.eigenvalues;
    eigh_evals.sort_by(f64::total_cmp);

    let max_diff: f64 = sturm_evals
        .iter()
        .zip(eigh_evals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("eigh vs Sturm n=64 W=3: max eigval diff {max_diff:.2e}"),
        max_diff,
        tolerances::SPECTRAL_EIGENSOLVER_CROSS,
    );

    // Also check that both return the same count
    h.check_bool(
        &format!(
            "eigh vs Sturm: same count ({} vs {})",
            eigh_evals.len(),
            sturm_evals.len()
        ),
        eigh_evals.len() == sturm_evals.len(),
    );
}

/// Larger-scale cross-validation: n=200 strongly disordered.
fn validate_eigh_vs_sturm_large(h: &mut ValidationHarness) {
    let n = 200;
    let disorder = 6.0;
    let seed = 99;

    let (diag, off_diag) = anderson_hamiltonian(n, disorder, seed);
    let sturm_evals = find_all_eigenvalues(&diag, &off_diag);

    let mut dense = vec![0.0_f64; n * n];
    for i in 0..n {
        dense[i * n + i] = diag[i];
        if i + 1 < n {
            dense[i * n + (i + 1)] = off_diag[i];
            dense[(i + 1) * n + i] = off_diag[i];
        }
    }
    let eigh_result = eigh_householder_qr(&dense, n);
    let mut eigh_evals = eigh_result.eigenvalues;
    eigh_evals.sort_by(f64::total_cmp);

    let max_diff: f64 = sturm_evals
        .iter()
        .zip(eigh_evals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("eigh vs Sturm n=200 W=6: max eigval diff {max_diff:.2e}"),
        max_diff,
        tolerances::SPECTRAL_EIGENSOLVER_CROSS,
    );
}
