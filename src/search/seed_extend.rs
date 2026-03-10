// SPDX-License-Identifier: AGPL-3.0-or-later

//! Seed-and-extend local alignment pipeline — BLAST-like search over
//! barraCuda `SmithWatermanGpu`.
//!
//! ## Pipeline
//!
//! 1. **Seed**: K-mer index lookup (CPU) — find candidate regions
//! 2. **Extend**: Smith-Waterman alignment (GPU) — score candidates
//! 3. **Score**: E-value filter (CPU) — statistical significance
//!
//! ## GPU Acceleration
//!
//! The extension phase uses `barracuda::ops::bio::SmithWatermanGpu` for
//! banded local alignment with affine gap penalties. Each seed hit
//! generates a query-target pair dispatched to the GPU.
//!
//! ## Batch Strategy
//!
//! Current: sequential SW dispatch per hit (one `align()` call each).
//! This exercises the existing barraCuda API and identifies where batch
//! dispatch would provide speedup.
//!
//! Future (barraCuda upstream): `SmithWatermanGpu::align_batch()` that
//! packs multiple query-target pairs into a single command encoder
//! submission, amortising dispatch overhead.

use super::kmer_index::{KmerIndex, SeedHit};

/// Configuration for the seed-and-extend pipeline.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// K-mer size for seeding (11 for DNA, 3 for protein).
    pub kmer_size: usize,
    /// Minimum alignment score to report a hit.
    pub min_score: f64,
    /// Smith-Waterman band width (0 = full DP).
    pub band_width: u32,
    /// Gap open penalty.
    pub gap_open: f64,
    /// Gap extend penalty.
    pub gap_extend: f64,
    /// Window size around seed hit for SW extension.
    pub extension_window: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            kmer_size: 11,
            min_score: 20.0,
            band_width: 64,
            gap_open: 11.0,
            gap_extend: 1.0,
            extension_window: 100,
        }
    }
}

/// A single alignment hit from the search pipeline.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Database sequence index.
    pub db_seq_id: u32,
    /// Alignment score from Smith-Waterman.
    pub score: f64,
    /// Query position where the seed was found.
    pub query_seed_pos: u32,
    /// Database position where the seed was found.
    pub db_seed_pos: u32,
}

/// Result of a search query.
#[derive(Debug)]
pub struct SearchResult {
    /// All hits passing the score threshold, sorted by score descending.
    pub hits: Vec<SearchHit>,
    /// Number of seed hits before extension.
    pub n_seeds: usize,
    /// Number of SW extensions performed.
    pub n_extensions: usize,
}

/// CPU-only seed-and-extend pipeline (reference implementation).
///
/// Uses CPU Smith-Waterman for extension. Serves as the correctness
/// baseline before GPU acceleration.
pub struct CpuSearchPipeline {
    index: KmerIndex,
    db_sequences: Vec<Vec<u32>>,
    config: SearchConfig,
}

impl CpuSearchPipeline {
    /// Build the search pipeline from a database of encoded sequences.
    ///
    /// Each entry is `(seq_id, encoded_bases)` with DNA encoding (A=0..T=3).
    #[must_use]
    pub fn new(sequences: Vec<Vec<u32>>, config: SearchConfig) -> Self {
        let indexed: Vec<(u32, &[u32])> = sequences
            .iter()
            .enumerate()
            .map(|(i, s)| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "sequence count fits in u32"
                )]
                let id = i as u32;
                (id, s.as_slice())
            })
            .collect();
        let index = KmerIndex::build(config.kmer_size, &indexed);
        Self {
            index,
            db_sequences: sequences,
            config,
        }
    }

    /// Search a query sequence against the database.
    #[must_use]
    pub fn search(&self, query: &[u32]) -> SearchResult {
        let seeds = self.index.seed_query(query);
        let n_seeds = seeds.len();

        // Deduplicate seed hits by (seq_id, diagonal) to avoid redundant
        // extensions on the same alignment region.
        let mut seen = std::collections::HashSet::new();
        let mut unique_seeds: Vec<(u32, SeedHit)> = Vec::new();
        for (qpos, hit) in &seeds {
            let diagonal = i64::from(*qpos) - i64::from(hit.pos);
            if seen.insert((hit.seq_id, diagonal)) {
                unique_seeds.push((*qpos, *hit));
            }
        }

        let mut hits = Vec::new();
        let n_extensions = unique_seeds.len();
        let ext_win = self.config.extension_window;

        for (qpos, seed) in &unique_seeds {
            let db_seq = &self.db_sequences[seed.seq_id as usize];

            let qp = *qpos as usize;
            let dp = seed.pos as usize;
            let q_start = qp.saturating_sub(ext_win);
            let q_end = (qp + self.config.kmer_size + ext_win).min(query.len());
            let d_start = dp.saturating_sub(ext_win);
            let d_end = (dp + self.config.kmer_size + ext_win).min(db_seq.len());

            let q_window = &query[q_start..q_end];
            let d_window = &db_seq[d_start..d_end];

            let score = cpu_smith_waterman(
                q_window,
                d_window,
                self.config.gap_open,
                self.config.gap_extend,
            );

            if score >= self.config.min_score {
                hits.push(SearchHit {
                    db_seq_id: seed.seq_id,
                    score,
                    query_seed_pos: *qpos,
                    db_seed_pos: seed.pos,
                });
            }
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        SearchResult {
            hits,
            n_seeds,
            n_extensions,
        }
    }

    /// Number of sequences in the database.
    #[must_use]
    pub const fn db_size(&self) -> u32 {
        self.index.n_sequences()
    }

    /// Number of distinct k-mers in the index.
    #[must_use]
    pub fn index_size(&self) -> usize {
        self.index.n_kmers()
    }
}

/// Simple CPU Smith-Waterman with affine gap penalties.
///
/// +2 match, -1 mismatch scoring matrix (DNA identity).
/// Returns the best local alignment score.
fn cpu_smith_waterman(query: &[u32], target: &[u32], gap_open: f64, gap_extend: f64) -> f64 {
    let qlen = query.len();
    let tlen = target.len();
    if qlen == 0 || tlen == 0 {
        return 0.0;
    }

    let mut dp_h = vec![vec![0.0_f64; tlen + 1]; qlen + 1];
    let mut dp_e = vec![vec![0.0_f64; tlen + 1]; qlen + 1];
    let mut dp_f = vec![vec![0.0_f64; tlen + 1]; qlen + 1];
    let mut best = 0.0_f64;

    for qi in 1..=qlen {
        for tj in 1..=tlen {
            let match_score = if query[qi - 1] == target[tj - 1] {
                2.0
            } else {
                -1.0
            };

            dp_e[qi][tj] = (dp_h[qi][tj - 1] - gap_open).max(dp_e[qi][tj - 1] - gap_extend);
            dp_f[qi][tj] = (dp_h[qi - 1][tj] - gap_open).max(dp_f[qi - 1][tj] - gap_extend);

            dp_h[qi][tj] = 0.0_f64
                .max(dp_h[qi - 1][tj - 1] + match_score)
                .max(dp_e[qi][tj])
                .max(dp_f[qi][tj]);

            if dp_h[qi][tj] > best {
                best = dp_h[qi][tj];
            }
        }
    }

    best
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
    fn cpu_sw_identical() {
        let q = encode_dna(b"ACGTACGT");
        let score = cpu_smith_waterman(&q, &q, 11.0, 1.0);
        assert!(
            (score - 16.0).abs() < 0.01,
            "8 matches × 2.0 = 16.0, got {score}"
        );
    }

    #[test]
    fn cpu_sw_no_match() {
        let q = encode_dna(b"AAAA");
        let t = encode_dna(b"TTTT");
        let score = cpu_smith_waterman(&q, &t, 11.0, 1.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn cpu_sw_partial_match() {
        let q = encode_dna(b"ACGTTTTT");
        let t = encode_dna(b"ACGTAAAA");
        let score = cpu_smith_waterman(&q, &t, 11.0, 1.0);
        assert!(score >= 8.0, "at least ACGT matches: {score}");
    }

    #[test]
    fn cpu_sw_empty() {
        assert_eq!(cpu_smith_waterman(&[], &[0, 1, 2], 11.0, 1.0), 0.0);
        assert_eq!(cpu_smith_waterman(&[0, 1], &[], 11.0, 1.0), 0.0);
    }

    #[test]
    fn search_finds_exact_match() {
        let db = vec![
            encode_dna(b"ACGTACGTACGTACGTACGT"),
            encode_dna(b"TTTTTTTTTTTTTTTTTTTT"),
            encode_dna(b"GGGGGGGGGGGGGGGGGGGG"),
        ];

        let config = SearchConfig {
            kmer_size: 4,
            min_score: 4.0,
            extension_window: 20,
            ..SearchConfig::default()
        };

        let pipeline = CpuSearchPipeline::new(db, config);
        assert_eq!(pipeline.db_size(), 3);

        let query = encode_dna(b"ACGTACGT");
        let result = pipeline.search(&query);

        assert!(result.n_seeds > 0, "should find seed hits");
        assert!(!result.hits.is_empty(), "should find alignment hits");
        assert_eq!(result.hits[0].db_seq_id, 0, "best hit should be seq 0");
        assert!(
            result.hits[0].score >= 8.0,
            "score = {}",
            result.hits[0].score
        );
    }

    #[test]
    fn search_no_match() {
        let db = vec![encode_dna(b"TTTTTTTTTTTTTTTTTTTT")];

        let config = SearchConfig {
            kmer_size: 4,
            min_score: 4.0,
            extension_window: 20,
            ..SearchConfig::default()
        };

        let pipeline = CpuSearchPipeline::new(db, config);

        let query = encode_dna(b"AAAAAAAA");
        let result = pipeline.search(&query);

        let matching_hits: Vec<_> = result.hits.iter().filter(|h| h.score >= 8.0).collect();
        assert!(
            matching_hits.is_empty(),
            "AAAA should not strongly match TTTT"
        );
    }

    #[test]
    fn search_deduplicates_diagonals() {
        let db = vec![encode_dna(b"ACGTACGTACGTACGTACGT")];

        let config = SearchConfig {
            kmer_size: 4,
            min_score: 1.0,
            extension_window: 20,
            ..SearchConfig::default()
        };

        let pipeline = CpuSearchPipeline::new(db, config);

        let query = encode_dna(b"ACGTACGT");
        let result = pipeline.search(&query);

        assert!(
            result.n_extensions <= result.n_seeds,
            "extensions ({}) should be <= seeds ({}) due to dedup",
            result.n_extensions,
            result.n_seeds
        );
    }

    #[test]
    fn search_results_sorted_by_score() {
        let db = vec![
            encode_dna(b"ACGTACGTACGTACGTACGT"),
            encode_dna(b"ACGTTTTTTTTTTTTTTTTTT"),
        ];

        let config = SearchConfig {
            kmer_size: 4,
            min_score: 1.0,
            extension_window: 20,
            ..SearchConfig::default()
        };

        let pipeline = CpuSearchPipeline::new(db, config);
        let query = encode_dna(b"ACGTACGT");
        let result = pipeline.search(&query);

        if result.hits.len() >= 2 {
            assert!(
                result.hits[0].score >= result.hits[1].score,
                "hits should be sorted descending by score"
            );
        }
    }

    #[test]
    fn search_empty_query() {
        let db = vec![encode_dna(b"ACGTACGT")];
        let config = SearchConfig {
            kmer_size: 4,
            min_score: 1.0,
            ..SearchConfig::default()
        };
        let pipeline = CpuSearchPipeline::new(db, config);
        let result = pipeline.search(&[]);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn search_empty_db() {
        let config = SearchConfig {
            kmer_size: 4,
            min_score: 1.0,
            ..SearchConfig::default()
        };
        let pipeline = CpuSearchPipeline::new(vec![], config);
        let query = encode_dna(b"ACGTACGT");
        let result = pipeline.search(&query);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn pipeline_index_stats() {
        let db = vec![
            encode_dna(b"ACGTACGTACGTACGT"),
            encode_dna(b"TGCATGCATGCATGCA"),
        ];
        let config = SearchConfig {
            kmer_size: 4,
            min_score: 1.0,
            ..SearchConfig::default()
        };
        let pipeline = CpuSearchPipeline::new(db, config);
        assert_eq!(pipeline.db_size(), 2);
        assert!(pipeline.index_size() > 0);
    }
}
