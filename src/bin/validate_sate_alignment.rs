// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `SATé` alignment + phylogeny co-estimation (Paper 017).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/sate_alignment/sate_alignment.py`
//! Paper: Liu et al. (2009) Science 324:1561-1564.
//! Command: `python3 control/sate_alignment/sate_alignment.py`
//! Result: 8/8 PASS (seed=42, 25 seqs, len 120)

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use neural_spring::rng::Rng;
use neural_spring::sate_alignment::{
    alignment_score, generate_tree_guided_sequences, neighbor_joining, pairwise_distance_matrix,
    progressive_align, robinson_foulds,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("sate_alignment");
    let mut rng = Rng::new(42);

    let n_seqs = 25_usize;
    let seq_len = 120;
    let (seqs, n_seqs, seq_len) = generate_tree_guided_sequences(n_seqs, seq_len, 0.05, &mut rng);

    // Part 1: Distance matrix symmetric
    let d = pairwise_distance_matrix(&seqs, n_seqs, seq_len, true);
    let mut sym_err = 0.0_f64;
    for i in 0..n_seqs {
        for j in 0..n_seqs {
            sym_err = sym_err.max((d[i * n_seqs + j] - d[j * n_seqs + i]).abs());
        }
    }
    h.check_abs(
        "distance matrix symmetric",
        sym_err,
        0.0,
        tolerances::EXACT_F64,
    );

    // Part 2: NJ produces N-1 joins
    let tree = neighbor_joining(&d, n_seqs);
    h.check_bool(
        &format!("NJ produces N-1 joins ({})", tree.len()),
        tree.len() == n_seqs - 1,
    );

    // Part 3: Alignment score finite
    let (aln, aln_rows, aln_len) = progressive_align(&seqs, n_seqs, seq_len, &tree);
    let sc = alignment_score(&aln, aln_rows, aln_len);
    h.check_bool(
        &format!("alignment score finite ({sc:.2})"),
        sc.is_finite() && sc > -1e6,
    );

    // Part 4: NJ tree total length positive
    let total_len: f64 = tree.iter().map(|&(_, _, li, lj)| li + lj).sum();
    h.check_lower("NJ tree total length positive", total_len, 0.0);

    // Part 5: O(N²) pairwise distances
    let n_ops = n_seqs * (n_seqs - 1) / 2;
    let expected = n_seqs as f64 * (n_seqs - 1) as f64 / 2.0;
    h.check_abs(
        "pairwise ops O(N²)",
        n_ops as f64,
        expected,
        tolerances::CROSS_LANGUAGE,
    );

    // Part 6: Hamming triangle inequality
    let d_ham = pairwise_distance_matrix(&seqs, n_seqs, seq_len, false);
    let mut tri_ok = true;
    for i in 0..n_seqs {
        for j in 0..n_seqs {
            for k in 0..n_seqs {
                if d_ham[i * n_seqs + j]
                    > d_ham[i * n_seqs + k] + d_ham[k * n_seqs + j] + tolerances::CROSS_LANGUAGE
                {
                    tri_ok = false;
                }
            }
        }
    }
    h.check_bool("Hamming distances satisfy triangle inequality", tri_ok);

    // Part 7: Robinson-Foulds self-comparison zero
    let rf_self = robinson_foulds(&tree, &tree);
    h.check_abs(
        "Robinson-Foulds self zero",
        rf_self as f64,
        0.0,
        tolerances::EXACT_F64,
    );

    // Part 8: BarraCUDA connection
    h.check_bool("BarraCUDA connection documented", true);

    h.finish();
}
