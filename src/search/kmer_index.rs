// SPDX-License-Identifier: AGPL-3.0-or-later

//! K-mer index for BLAST-like seed-and-extend search.
//!
//! Builds a hash-based index from a FASTA database: for each k-mer,
//! stores (`sequence_id`, position) pairs. Used as Phase 1 (SEED) of the
//! BLAST pipeline.
//!
//! ## DNA encoding
//!
//! A=0, C=1, G=2, T=3. K-mers are packed as base-4 integers:
//! `ACGT` → 0×64 + 1×16 + 2×4 + 3 = 27.
//!
//! K-mers containing N (or any non-ACGT) are skipped.

use std::collections::HashMap;

/// A seed hit: (database sequence index, position within that sequence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedHit {
    /// Index of the database sequence containing this hit.
    pub seq_id: u32,
    /// Zero-based start position of the k-mer within that sequence.
    pub pos: u32,
}

/// Hash-based k-mer index for a sequence database.
#[derive(Debug)]
pub struct KmerIndex {
    k: usize,
    index: HashMap<u64, Vec<SeedHit>>,
    n_sequences: u32,
}

impl KmerIndex {
    /// Build a k-mer index from encoded sequences.
    ///
    /// Each entry in `sequences` is a `(seq_id, encoded_bases)` pair where
    /// bases use DNA encoding (A=0, C=1, G=2, T=3, N=4+).
    #[must_use]
    pub fn build(k: usize, sequences: &[(u32, &[u32])]) -> Self {
        let mut index: HashMap<u64, Vec<SeedHit>> = HashMap::new();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "database sequence count fits in u32"
        )]
        let n_sequences = sequences.len() as u32;

        for &(seq_id, bases) in sequences {
            if bases.len() < k {
                continue;
            }
            for pos in 0..=(bases.len() - k) {
                if let Some(kmer) = encode_kmer(&bases[pos..pos + k]) {
                    index.entry(kmer).or_default().push(SeedHit {
                        seq_id,
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "sequence positions fit in u32"
                        )]
                        pos: pos as u32,
                    });
                }
            }
        }

        Self {
            k,
            index,
            n_sequences,
        }
    }

    /// Look up all database positions matching a query k-mer.
    #[must_use]
    pub fn lookup(&self, kmer: &[u32]) -> &[SeedHit] {
        debug_assert_eq!(kmer.len(), self.k);
        encode_kmer(kmer)
            .and_then(|h| self.index.get(&h))
            .map_or(&[], Vec::as_slice)
    }

    /// Find all seed hits for a query sequence against this index.
    ///
    /// Returns `(query_pos, SeedHit)` pairs for every k-mer match.
    #[must_use]
    pub fn seed_query(&self, query: &[u32]) -> Vec<(u32, SeedHit)> {
        if query.len() < self.k {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for qpos in 0..=(query.len() - self.k) {
            let kmer = &query[qpos..qpos + self.k];
            for &hit in self.lookup(kmer) {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "query positions fit in u32"
                )]
                hits.push((qpos as u32, hit));
            }
        }
        hits
    }

    /// K-mer size.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of distinct k-mers in the index.
    #[must_use]
    pub fn n_kmers(&self) -> usize {
        self.index.len()
    }

    /// Number of sequences in the indexed database.
    #[must_use]
    pub const fn n_sequences(&self) -> u32 {
        self.n_sequences
    }
}

/// Encode a k-mer as a packed base-4 integer. Returns `None` if any
/// base is not in {0,1,2,3} (i.e. contains N or unknown).
fn encode_kmer(kmer: &[u32]) -> Option<u64> {
    let mut hash: u64 = 0;
    for &base in kmer {
        if base > 3 {
            return None;
        }
        hash = hash * 4 + u64::from(base);
    }
    Some(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_dna(s: &[u8]) -> Vec<u32> {
        s.iter()
            .map(|&b| match b {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => 4,
            })
            .collect()
    }

    #[test]
    fn encode_kmer_basic() {
        assert_eq!(encode_kmer(&[0, 1, 2, 3]), Some(27));
        assert_eq!(encode_kmer(&[0, 0, 0, 0]), Some(0));
        assert_eq!(encode_kmer(&[3, 3, 3, 3]), Some(255));
    }

    #[test]
    fn encode_kmer_with_n() {
        assert_eq!(encode_kmer(&[0, 4, 2, 3]), None);
    }

    #[test]
    fn build_and_lookup() {
        let s1 = encode_dna(b"ACGTACGT");
        let s2 = encode_dna(b"TGCAACGT");

        let idx = KmerIndex::build(4, &[(0, &s1), (1, &s2)]);
        assert_eq!(idx.k(), 4);
        assert_eq!(idx.n_sequences(), 2);
        assert!(idx.n_kmers() > 0);

        let acgt = encode_dna(b"ACGT");
        let hits = idx.lookup(&acgt);
        assert!(hits.len() >= 2, "ACGT appears in both sequences");
    }

    #[test]
    fn seed_query() {
        let db1 = encode_dna(b"ACGTACGTACGT");
        let db2 = encode_dna(b"TTTTTTTTTTTT");

        let idx = KmerIndex::build(4, &[(0, &db1), (1, &db2)]);

        let query = encode_dna(b"ACGTTTTT");
        let hits = idx.seed_query(&query);

        assert!(
            hits.iter().any(|(_, h)| h.seq_id == 0),
            "ACGT in query should hit db sequence 0"
        );

        assert!(
            hits.iter().any(|(_, h)| h.seq_id == 1),
            "TTTT in query should hit db sequence 1"
        );
    }

    #[test]
    fn empty_database() {
        let idx = KmerIndex::build(4, &[]);
        assert_eq!(idx.n_sequences(), 0);
        assert_eq!(idx.n_kmers(), 0);
        assert!(idx.seed_query(&[0, 1, 2, 3]).is_empty());
    }

    #[test]
    fn short_sequence_skipped() {
        let short = encode_dna(b"AC");
        let idx = KmerIndex::build(4, &[(0, &short)]);
        assert_eq!(idx.n_kmers(), 0);
    }

    #[test]
    fn n_bases_skipped() {
        let seq = encode_dna(b"ACNGT");
        let idx = KmerIndex::build(3, &[(0, &seq)]);
        let hits = idx.lookup(&encode_dna(b"ACN"));
        assert!(hits.is_empty(), "k-mers with N should not be indexed");
    }

    #[test]
    fn seed_positions_correct() {
        let db = encode_dna(b"ACGTACGT");
        let idx = KmerIndex::build(4, &[(0, &db)]);

        let hits = idx.lookup(&encode_dna(b"ACGT"));
        let positions: Vec<u32> = hits.iter().map(|h| h.pos).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&4));
    }
}
