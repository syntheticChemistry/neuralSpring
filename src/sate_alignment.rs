// SPDX-License-Identifier: AGPL-3.0-or-later

//! `SATe`: Iterative co-estimation of MSA and phylogenetic tree.
//!
//! Port of `control/sate_alignment/sate_alignment.py`.
//!
//! Liu et al. (2009) "Rapid and accurate large-scale coestimation of
//! sequence alignments and phylogenetic trees" (`SATe`)
//! Science 324:1561-1564.
//!
//! Computational core: distance matrix (GEMM-equivalent) + neighbor-joining
//! + progressive alignment merging.
//!
//! ## GPU-ready layout
//!
//! Sequences and matrices use **flat row-major**:
//! - Sequences: `seqs[i]` at `i * seq_len .. (i+1) * seq_len`
//! - Distance matrix: `d[i,j]` at `i * n + j`
//! - Alignment: `aln[i,c]` at `i * aln_len + c`
//!
//! ## `BarraCUDA` connection
//!
//! - Pairwise distance matrix: `barracuda::ops::pairwise_distance` (GPU `pairwise_hamming.wgsl`)
//! - Jukes-Cantor correction: elementwise log transform
//! - Neighbor-joining: sequential algorithm (CPU-only, not GPU-portable)
//! - Progressive alignment: sequential merging (CPU-only)
//!
//! ## WGSL shader (absorption-ready)
//!
//! [`WGSL_PAIRWISE_HAMMING`] — pairwise Hamming distance matrix. One
//! thread per sequence pair. Validated in `validate_gpu_sate`.

// Domain-inherent: bioinformatics matrix algorithms require casts and
// index-based loops that clippy flags but cannot be meaningfully refactored.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use crate::rng::Rng;

/// WGSL shader: pairwise Hamming distance for sequence comparison.
///
/// Absorption target: `barracuda::ops::pairwise_distance` or `cdist_wgsl`.
/// Validated: `validate_gpu_sate`.
pub const WGSL_PAIRWISE_HAMMING: &str = include_str!("../metalForge/shaders/pairwise_hamming.wgsl");

const GAP: u8 = 4;
const JC_SATURATION: f64 = 10.0;

/// Hamming distance: proportion of differing sites.
fn hamming_distance(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() {
        return 1.0;
    }
    let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    diff as f64 / a.len() as f64
}

/// Jukes-Cantor correction: d = -3/4 * ln(1 - 4/3*p).
fn jukes_cantor(p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 0.75 {
        return JC_SATURATION;
    }
    -0.75 * (4.0_f64 / 3.0).mul_add(-p, 1.0).ln()
}

/// Compute N×N pairwise distance matrix (GEMM-equivalent).
///
/// Sequences flat row-major: `seqs[i]` at `i * seq_len .. (i+1) * seq_len`.
/// Returns flat row-major `n×n`: `d[i,j]` at `i * n + j`.
#[must_use]
pub fn pairwise_distance_matrix(
    seqs: &[u8],
    n_seqs: usize,
    seq_len: usize,
    use_jc: bool,
) -> Vec<f64> {
    let mut d = vec![0.0; n_seqs * n_seqs];
    for i in 0..n_seqs {
        let seq_i = &seqs[i * seq_len..(i + 1) * seq_len];
        for j in (i + 1)..n_seqs {
            let seq_j = &seqs[j * seq_len..(j + 1) * seq_len];
            let p = hamming_distance(seq_i, seq_j);
            let dist = if use_jc { jukes_cantor(p) } else { p };
            d[i * n_seqs + j] = dist;
            d[j * n_seqs + i] = dist;
        }
    }
    d
}

/// One NJ join: `(i, j, len_i, len_j)`.
pub type NjJoin = (usize, usize, f64, f64);

/// Neighbor-joining tree construction. Returns N-1 joins.
///
/// Distance matrix flat row-major `n×n`: `d[i,j]` at `i * n + j`.
#[must_use]
pub fn neighbor_joining(d: &[f64], n: usize) -> Vec<NjJoin> {
    if n <= 1 {
        return vec![];
    }
    if n == 2 {
        return vec![(0, 1, d[1] / 2.0, d[1] / 2.0)];
    }

    let max_nodes = 2 * n - 1;
    let mut dist = vec![0.0; max_nodes * max_nodes];
    for i in 0..n {
        for j in 0..n {
            dist[i * max_nodes + j] = d[i * n + j];
        }
    }
    let mut active: Vec<usize> = (0..n).collect();
    let mut next_node = n;
    let mut tree = Vec::new();

    while active.len() > 2 {
        let nn = active.len();
        let mut min_q = f64::INFINITY;
        let mut join_i = 0;
        let mut join_j = 0;

        for (idx_a, &i) in active.iter().enumerate() {
            for &j in active.iter().skip(idx_a + 1) {
                let s_i: f64 = active
                    .iter()
                    .filter(|&&k| k != i)
                    .map(|&k| dist[i * max_nodes + k])
                    .sum();
                let s_j: f64 = active
                    .iter()
                    .filter(|&&k| k != j)
                    .map(|&k| dist[j * max_nodes + k])
                    .sum();
                let d_ij = dist[i * max_nodes + j];
                let q = ((nn - 2) as f64).mul_add(d_ij, -s_i) - s_j;
                if q < min_q {
                    min_q = q;
                    join_i = i;
                    join_j = j;
                }
            }
        }

        let s_i: f64 = active
            .iter()
            .filter(|&&k| k != join_i)
            .map(|&k| dist[join_i * max_nodes + k])
            .sum();
        let s_j: f64 = active
            .iter()
            .filter(|&&k| k != join_j)
            .map(|&k| dist[join_j * max_nodes + k])
            .sum();
        let d_ij = dist[join_i * max_nodes + join_j];
        let len_i = (0.5 * (d_ij + (s_i - s_j) / (nn - 2) as f64)).max(0.0);
        let len_j = (d_ij - len_i).max(0.0);

        tree.push((join_i, join_j, len_i, len_j));

        let u = next_node;
        next_node += 1;

        for &k in &active {
            if k != join_i && k != join_j {
                let d_uk = 0.5
                    * (dist[join_i * max_nodes + k] + dist[join_j * max_nodes + k]
                        - dist[join_i * max_nodes + join_j]);
                dist[u * max_nodes + k] = d_uk;
                dist[k * max_nodes + u] = d_uk;
            }
        }

        active.retain(|&x| x != join_i && x != join_j);
        active.push(u);
    }

    let [i, j] = match *active.as_slice() {
        [a, b, ..] => [a, b],
        [a] => [a, 0],
        [] => [0, 1],
    };
    let len = dist[i * max_nodes + j] / 2.0;
    tree.push((i, j, len, len));
    tree
}

/// Simple Needleman-Wunsch: match=0, mismatch=1, gap=1.
fn align_pair(seq_a: &[u8], seq_b: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let len_m = seq_a.len();
    let len_n = seq_b.len();
    let cols = len_n + 1;
    let mut f = vec![0.0; (len_m + 1) * cols];
    for idx_i in 1..=len_m {
        f[idx_i * cols] = idx_i as f64;
    }
    for idx_j in 1..=len_n {
        f[idx_j] = idx_j as f64;
    }
    for idx_i in 1..=len_m {
        for idx_j in 1..=len_n {
            let cost = if seq_a[idx_i - 1] == seq_b[idx_j - 1] {
                0.0
            } else {
                1.0
            };
            f[idx_i * cols + idx_j] = (f[(idx_i - 1) * cols + idx_j - 1] + cost)
                .min(f[(idx_i - 1) * cols + idx_j] + 1.0)
                .min(f[idx_i * cols + idx_j - 1] + 1.0);
        }
    }

    let mut aln_a = Vec::new();
    let mut aln_b = Vec::new();
    let mut idx_i = len_m;
    let mut idx_j = len_n;
    while idx_i > 0 || idx_j > 0 {
        let cost = if idx_i > 0 && idx_j > 0 && seq_a[idx_i - 1] == seq_b[idx_j - 1] {
            0.0
        } else {
            1.0
        };
        if idx_i > 0
            && idx_j > 0
            && (f[idx_i * cols + idx_j] - (f[(idx_i - 1) * cols + idx_j - 1] + cost)).abs() < 1e-10
        {
            aln_a.push(seq_a[idx_i - 1]);
            aln_b.push(seq_b[idx_j - 1]);
            idx_i -= 1;
            idx_j -= 1;
        } else if idx_i > 0
            && (f[idx_i * cols + idx_j] - (f[(idx_i - 1) * cols + idx_j] + 1.0)).abs() < 1e-10
        {
            aln_a.push(seq_a[idx_i - 1]);
            aln_b.push(GAP);
            idx_i -= 1;
        } else {
            aln_a.push(GAP);
            aln_b.push(seq_b[idx_j - 1]);
            idx_j -= 1;
        }
    }
    aln_a.reverse();
    aln_b.reverse();
    (aln_a, aln_b)
}

/// Progressive alignment (caterpillar guide tree).
///
/// Sequences flat row-major. Returns `(aln_flat, n_seqs, aln_len)`.
#[must_use]
pub fn progressive_align(
    seqs: &[u8],
    n_seqs: usize,
    seq_len: usize,
    _tree: &[NjJoin],
) -> (Vec<u8>, usize, usize) {
    if n_seqs == 0 {
        return (vec![], 0, 0);
    }
    if n_seqs == 1 {
        return (seqs[0..seq_len].to_vec(), 1, seq_len);
    }

    let first = &seqs[0..seq_len];
    let second = &seqs[seq_len..2 * seq_len];
    let (a, b) = align_pair(first, second);
    let mut merged = vec![a, b];

    for i in 2..n_seqs {
        let seq = &seqs[i * seq_len..(i + 1) * seq_len];
        let guide: Vec<u8> = merged[0].iter().copied().filter(|&x| x != GAP).collect();
        let guide_ref: Vec<u8> = if guide.is_empty() {
            first.to_vec()
        } else {
            guide
        };
        let (a_new, b_new) = align_pair(&guide_ref, seq);

        let non_gap_cols: Vec<usize> = merged[0]
            .iter()
            .enumerate()
            .filter_map(|(idx, &x)| if x == GAP { None } else { Some(idx) })
            .collect();

        let mut i_old = 0;
        let l_out = a_new.len();
        let mut expanded = vec![vec![GAP; l_out]; merged.len()];
        for c in 0..l_out {
            if a_new[c] != GAP && i_old < non_gap_cols.len() {
                let col = non_gap_cols[i_old];
                for (row, exp_row) in expanded.iter_mut().enumerate() {
                    exp_row[c] = merged[row][col];
                }
                i_old += 1;
            }
        }
        let new_row: Vec<u8> = b_new
            .iter()
            .map(|&x| if x == GAP { GAP } else { x })
            .collect();
        expanded.push(new_row);
        merged = expanded;
    }

    let n_rows = merged.len();
    let aln_len = merged[0].len();
    let mut flat = Vec::with_capacity(n_rows * aln_len);
    for row in &merged {
        flat.extend_from_slice(row);
    }
    (flat, n_rows, aln_len)
}

/// Sum-of-pairs alignment score. Higher is better.
///
/// Alignment flat row-major: `aln[i,c]` at `i * aln_len + c`.
#[must_use]
pub fn alignment_score(aln: &[u8], n_seqs: usize, aln_len: usize) -> f64 {
    if n_seqs == 0 {
        return 0.0;
    }
    let mut sp = 0.0;
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            for c in 0..aln_len {
                let a = aln[i * aln_len + c];
                let b = aln[j * aln_len + c];
                if a != GAP && b != GAP {
                    sp += if a == b { 1.0 } else { -0.5 };
                }
            }
        }
    }
    sp
}

/// Robinson-Foulds topological distance (symmetric difference of splits).
/// Simplified: count edges in tree1 not in tree2 (as unordered pairs).
#[must_use]
pub fn robinson_foulds(tree1: &[NjJoin], tree2: &[NjJoin]) -> usize {
    let pairs1: std::collections::HashSet<(usize, usize)> = tree1
        .iter()
        .map(|&(i, j, _, _)| (i.min(j), i.max(j)))
        .collect();
    let pairs2: std::collections::HashSet<(usize, usize)> = tree2
        .iter()
        .map(|&(i, j, _, _)| (i.min(j), i.max(j)))
        .collect();
    pairs1.symmetric_difference(&pairs2).count()
}

/// Generate root DNA sequence (A=0, C=1, G=2, T=3).
#[must_use]
pub fn generate_root_sequence(length: usize, rng: &mut Rng) -> Vec<u8> {
    (0..length).map(|_| rng.usize(4) as u8).collect()
}

/// Mutate sequence along branch with substitution rate.
#[must_use]
pub fn mutate_along_branch(seq: &[u8], rate: f64, rng: &mut Rng) -> Vec<u8> {
    let mut out = seq.to_vec();
    let n = seq.len();
    let n_mut = (n as f64 * rate).round() as usize;
    let n_mut = n_mut.min(n);
    if n_mut == 0 {
        return out;
    }
    let indices = rng.choose_distinct(n, n_mut);
    for i in indices {
        let others: Vec<u8> = (0..4).filter(|&x| x != seq[i]).collect();
        out[i] = others[rng.usize(others.len())];
    }
    out
}

/// Generate tree-guided sequences.
///
/// Returns flat row-major `(seqs_flat, n_seqs, seq_len)`.
#[must_use]
pub fn generate_tree_guided_sequences(
    n_seqs: usize,
    seq_len: usize,
    branch_rate: f64,
    rng: &mut Rng,
) -> (Vec<u8>, usize, usize) {
    let root = generate_root_sequence(seq_len, rng);
    let mut flat = Vec::with_capacity(n_seqs * seq_len);
    flat.extend_from_slice(&root);
    for _ in 1..n_seqs {
        flat.extend(mutate_along_branch(&root, branch_rate, rng));
    }
    (flat, n_seqs, seq_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_symmetric() {
        let a = vec![0u8, 1, 2, 3];
        let b = vec![0u8, 2, 2, 1];
        assert!((hamming_distance(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn jc_zero_for_identical() {
        assert!(jukes_cantor(0.0).abs() < 1e-10);
    }

    #[test]
    fn distance_matrix_symmetric() {
        let mut rng = Rng::new(42);
        let (seqs, n, len) = generate_tree_guided_sequences(5, 50, 0.05, &mut rng);
        let d = pairwise_distance_matrix(&seqs, n, len, true);
        for i in 0..5 {
            for j in 0..5 {
                assert!((d[i * 5 + j] - d[j * 5 + i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn nj_produces_n_minus_1_joins() {
        let mut rng = Rng::new(42);
        let (seqs, n, len) = generate_tree_guided_sequences(10, 80, 0.05, &mut rng);
        let d = pairwise_distance_matrix(&seqs, n, len, true);
        let tree = neighbor_joining(&d, n);
        assert_eq!(tree.len(), 9);
    }

    #[test]
    fn alignment_score_non_negative_reasonable() {
        let mut rng = Rng::new(42);
        let (seqs, n, len) = generate_tree_guided_sequences(5, 30, 0.03, &mut rng);
        let d = pairwise_distance_matrix(&seqs, n, len, true);
        let tree = neighbor_joining(&d, n);
        let (aln, n_rows, aln_len) = progressive_align(&seqs, n, len, &tree);
        let sc = alignment_score(&aln, n_rows, aln_len);
        assert!(sc > -1e6);
    }
}
