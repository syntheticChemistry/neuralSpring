// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sequence search pipelines — BLAST-like seed-and-extend over barraCuda GPU.
//!
//! Composes barraCuda primitives (`SmithWatermanGpu`, `KmerHistogramGpu`) into
//! multi-stage pipelines that replace external tools (BLAST, HMMER).
//!
//! ## Architecture
//!
//! ```text
//! Phase 1: SEED        Phase 2: EXTEND            Phase 3: SCORE
//! ─────────────────    ─────────────────────────   ────────────────
//! k-mer index of DB    Smith-Waterman on hits      E-value, bit score
//! exact word match     banded + affine gaps        Karlin-Altschul
//! O(n) scan            GPU batch dispatch          per-hit filter
//! ```
//!
//! See `specs/BLAST_LIKE_SEARCH_SCOPE.md` for the full design.

pub mod kmer_index;
pub mod seed_extend;

#[cfg(test)]
mod tests {
    use super::kmer_index::{KmerIndex, SeedHit};

    #[test]
    fn kmer_index_basic_lookup() {
        let seq: Vec<u32> = vec![0, 1, 2, 3]; // ACGT
        let sequences = [(0u32, seq.as_slice())];
        let idx = KmerIndex::build(3, &sequences);
        // ACG = k-mer at pos 0
        let hits = idx.lookup(&[0, 1, 2]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0], SeedHit { seq_id: 0, pos: 0 });
    }

    #[test]
    fn kmer_index_no_match() {
        let seq: Vec<u32> = vec![0, 0, 0, 0]; // AAAA
        let sequences = [(0u32, seq.as_slice())];
        let idx = KmerIndex::build(3, &sequences);
        let hits = idx.lookup(&[1, 2, 3]); // CGT not in AAAA
        assert!(hits.is_empty());
    }

    #[test]
    fn kmer_index_skips_n_bases() {
        let seq: Vec<u32> = vec![0, 4, 2, 3]; // A, N, G, T — N=4 should skip
        let sequences = [(0u32, seq.as_slice())];
        let idx = KmerIndex::build(3, &sequences);
        // ANG is invalid (contains N=4), should return empty
        let hits = idx.lookup(&[0, 4, 2]);
        assert!(hits.is_empty());
    }

    #[test]
    fn kmer_index_multiple_sequences() {
        let seq1: Vec<u32> = vec![0, 1, 2]; // ACG
        let seq2: Vec<u32> = vec![0, 1, 2]; // ACG (duplicate)
        let sequences = [(0u32, seq1.as_slice()), (1, seq2.as_slice())];
        let idx = KmerIndex::build(3, &sequences);
        let hits = idx.lookup(&[0, 1, 2]);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn kmer_index_short_sequence_skipped() {
        let seq: Vec<u32> = vec![0, 1]; // too short for k=3
        let sequences = [(0u32, seq.as_slice())];
        let idx = KmerIndex::build(3, &sequences);
        assert_eq!(idx.n_sequences(), 1);
    }
}
