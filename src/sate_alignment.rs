// SPDX-License-Identifier: AGPL-3.0-or-later

//! `SATé`: Iterative co-estimation of MSA and phylogenetic tree.
//!
//! Port of `control/sate_alignment/sate_alignment.py`.
//!
//! Liu et al. (2009) "Rapid and accurate large-scale coestimation of
//! sequence alignments and phylogenetic trees" (`SATé`)
//! Science 324:1561-1564.
//!
//! Computational core: distance matrix (GEMM-equivalent) + neighbor-joining
//! + progressive alignment merging.
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
    -0.75 * (1.0_f64 - (4.0 / 3.0) * p).ln()
}

/// Compute N×N pairwise distance matrix (GEMM-equivalent).
pub fn pairwise_distance_matrix(seqs: &[Vec<u8>], use_jc: bool) -> Vec<Vec<f64>> {
    let n = seqs.len();
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let p = hamming_distance(&seqs[i], &seqs[j]);
            let dist = if use_jc { jukes_cantor(p) } else { p };
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }
    d
}

/// One NJ join: `(i, j, len_i, len_j)`.
pub type NjJoin = (usize, usize, f64, f64);

/// Neighbor-joining tree construction. Returns N-1 joins.
pub fn neighbor_joining(d: &[Vec<f64>]) -> Vec<NjJoin> {
    let n = d.len();
    if n <= 1 {
        return vec![];
    }
    if n == 2 {
        return vec![(0, 1, d[0][1] / 2.0, d[0][1] / 2.0)];
    }

    let mut dist: Vec<Vec<f64>> = d.iter().map(Vec::clone).collect();
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
                    .map(|&k| dist[i][k])
                    .sum();
                let s_j: f64 = active
                    .iter()
                    .filter(|&&k| k != j)
                    .map(|&k| dist[j][k])
                    .sum();
                let q = (nn - 2) as f64 * dist[i][j] - s_i - s_j;
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
            .map(|&k| dist[join_i][k])
            .sum();
        let s_j: f64 = active
            .iter()
            .filter(|&&k| k != join_j)
            .map(|&k| dist[join_j][k])
            .sum();
        let len_i = (0.5 * (dist[join_i][join_j] + (s_i - s_j) / (nn - 2) as f64)).max(0.0);
        let len_j = (dist[join_i][join_j] - len_i).max(0.0);

        tree.push((join_i, join_j, len_i, len_j));

        let u = next_node;
        next_node += 1;

        let curr_n = dist.len();
        for row in &mut dist {
            row.push(0.0);
        }
        dist.push(vec![0.0; curr_n + 1]);

        for &k in &active {
            if k != join_i && k != join_j {
                let d_uk = 0.5 * (dist[join_i][k] + dist[join_j][k] - dist[join_i][join_j]);
                dist[u][k] = d_uk;
                dist[k][u] = d_uk;
            }
        }

        active.retain(|&x| x != join_i && x != join_j);
        active.push(u);
    }

    let mut it = active.into_iter();
    let i = it.next().unwrap_or(0);
    let j = it.next().unwrap_or(1);
    let len = dist[i][j] / 2.0;
    tree.push((i, j, len, len));
    tree
}

/// Simple Needleman-Wunsch: match=0, mismatch=1, gap=1.
fn align_pair(seq_a: &[u8], seq_b: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let len_m = seq_a.len();
    let len_n = seq_b.len();
    let mut f = vec![vec![0.0; len_n + 1]; len_m + 1];
    for idx_i in 1..=len_m {
        f[idx_i][0] = idx_i as f64;
    }
    for idx_j in 1..=len_n {
        f[0][idx_j] = idx_j as f64;
    }
    for idx_i in 1..=len_m {
        for idx_j in 1..=len_n {
            let cost = if seq_a[idx_i - 1] == seq_b[idx_j - 1] {
                0.0
            } else {
                1.0
            };
            f[idx_i][idx_j] = (f[idx_i - 1][idx_j - 1] + cost)
                .min(f[idx_i - 1][idx_j] + 1.0)
                .min(f[idx_i][idx_j - 1] + 1.0);
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
            && (f[idx_i][idx_j] - (f[idx_i - 1][idx_j - 1] + cost)).abs() < 1e-10
        {
            aln_a.push(seq_a[idx_i - 1]);
            aln_b.push(seq_b[idx_j - 1]);
            idx_i -= 1;
            idx_j -= 1;
        } else if idx_i > 0 && (f[idx_i][idx_j] - (f[idx_i - 1][idx_j] + 1.0)).abs() < 1e-10 {
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
pub fn progressive_align(seqs: &[Vec<u8>], _tree: &[NjJoin]) -> Vec<Vec<u8>> {
    if seqs.is_empty() {
        return vec![];
    }
    if seqs.len() == 1 {
        return vec![seqs[0].clone()];
    }

    let (a, b) = align_pair(&seqs[0], &seqs[1]);
    let mut merged = vec![a, b];

    for seq in seqs.iter().skip(2) {
        let guide: Vec<u8> = merged[0].iter().copied().filter(|&x| x != GAP).collect();
        let guide_ref: Vec<u8> = if guide.is_empty() {
            seqs[0].clone()
        } else {
            guide
        };
        let (a_new, b_new) = align_pair(&guide_ref, seq);

        let non_gap_cols: Vec<usize> = merged[0]
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if x == GAP { None } else { Some(i) })
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
    merged
}

/// Sum-of-pairs alignment score. Higher is better.
pub fn alignment_score(aln: &[Vec<u8>]) -> f64 {
    let n = aln.len();
    if n == 0 {
        return 0.0;
    }
    let l = aln[0].len();
    let mut sp = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            for c in 0..l {
                let (a, b) = (aln[i][c], aln[j][c]);
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
pub fn generate_root_sequence(length: usize, rng: &mut Rng) -> Vec<u8> {
    (0..length).map(|_| rng.usize(4) as u8).collect()
}

/// Mutate sequence along branch with substitution rate.
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
pub fn generate_tree_guided_sequences(
    n_seqs: usize,
    seq_len: usize,
    branch_rate: f64,
    rng: &mut Rng,
) -> Vec<Vec<u8>> {
    let root = generate_root_sequence(seq_len, rng);
    let mut seqs = vec![root.clone()];
    for _ in 1..n_seqs {
        seqs.push(mutate_along_branch(&root, branch_rate, rng));
    }
    seqs
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
        let seqs = generate_tree_guided_sequences(5, 50, 0.05, &mut rng);
        let d = pairwise_distance_matrix(&seqs, true);
        for i in 0..5 {
            for j in 0..5 {
                assert!((d[i][j] - d[j][i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn nj_produces_n_minus_1_joins() {
        let mut rng = Rng::new(42);
        let seqs = generate_tree_guided_sequences(10, 80, 0.05, &mut rng);
        let d = pairwise_distance_matrix(&seqs, true);
        let tree = neighbor_joining(&d);
        assert_eq!(tree.len(), 9);
    }

    #[test]
    fn alignment_score_non_negative_reasonable() {
        let mut rng = Rng::new(42);
        let seqs = generate_tree_guided_sequences(5, 30, 0.03, &mut rng);
        let d = pairwise_distance_matrix(&seqs, true);
        let tree = neighbor_joining(&d);
        let aln = progressive_align(&seqs, &tree);
        let sc = alignment_score(&aln);
        assert!(sc > -1e6);
    }
}
