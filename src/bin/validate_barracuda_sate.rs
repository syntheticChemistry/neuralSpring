// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: `SATé` alignment and NJ phylogeny (Paper 017).
//!
//! Validates `barracuda::stats::variance` and `barracuda::linalg` concepts
//! for distance matrix computation and NJ tree validity.
//!
//! Evolution path:
//! ```text
//! Python (numpy, scipy) → Rust (hand-rolled pairwise, NJ)
//!   → BarraCUDA CPU (stats::variance for distance validation)
//!   → BarraCUDA GPU (cdist, tree ops)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/sate_alignment/sate_alignment.py`
//! Rust baseline: `validate_sate_alignment`

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::needless_range_loop,
    clippy::similar_names
)]

use neural_spring::rng::Rng;
use neural_spring::sate_alignment::{
    generate_tree_guided_sequences, neighbor_joining, pairwise_distance_matrix, progressive_align,
};
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_sate");

    validate_distance_matrix_symmetry(&mut h);
    validate_distance_variance(&mut h);
    validate_nj_tree(&mut h);

    h.finish();
}

fn validate_distance_matrix_symmetry(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let seqs = generate_tree_guided_sequences(10, 80, 0.05, &mut rng);
    let d = pairwise_distance_matrix(&seqs, true);
    let n = d.len();

    let mut max_sym_err = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let err = (d[i][j] - d[j][i]).abs();
            max_sym_err = max_sym_err.max(err);
        }
    }

    h.check_abs(
        "distance matrix symmetric (D[i,j]==D[j,i])",
        max_sym_err,
        0.0,
        1e-12,
    );
}

fn validate_distance_variance(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let seqs = generate_tree_guided_sequences(15, 60, 0.05, &mut rng);
    let d = pairwise_distance_matrix(&seqs, true);

    let mut upper_tri = Vec::new();
    for i in 0..d.len() {
        for j in (i + 1)..d.len() {
            upper_tri.push(d[i][j]);
        }
    }

    let barracuda_var = barracuda::stats::correlation::variance(&upper_tri).unwrap_or(f64::NAN);

    h.check_bool(
        "distance variance finite and non-negative",
        barracuda_var.is_finite() && barracuda_var >= 0.0,
    );
    h.check_bool(
        "pairwise distances non-negative",
        upper_tri.iter().all(|&x| x >= -1e-12),
    );
}

fn validate_nj_tree(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let n_seqs = 12;
    let seqs = generate_tree_guided_sequences(n_seqs, 50, 0.05, &mut rng);
    let d = pairwise_distance_matrix(&seqs, true);

    let tree = neighbor_joining(&d);

    h.check_bool(
        &format!("NJ produces N-1 joins ({})", tree.len()),
        tree.len() == n_seqs - 1,
    );

    let total_len: f64 = tree.iter().map(|&(_, _, li, lj)| li + lj).sum();
    h.check_lower("NJ tree total length positive", total_len, 0.0);

    let aln = progressive_align(&seqs, &tree);
    h.check_bool(
        "progressive align produces N sequences",
        aln.len() == n_seqs,
    );
}
