// SPDX-License-Identifier: AGPL-3.0-or-later

//! BLAST-like search results scenario builder.
//!
//! Runs a real seed-and-extend search against a small synthetic database
//! and visualizes: alignment score bar chart, hit distribution across
//! database sequences, seed density heatmap, and pipeline statistics gauge.

#![expect(
    clippy::cast_precision_loss,
    reason = "index-to-f64 conversions for visualization axes"
)]

use crate::search::kmer_index::KmerIndex;
use crate::search::seed_extend::{CpuSearchPipeline, SearchConfig};
use crate::visualization::types::{NeuralScenario, ScenarioEdge, ThresholdRange};

use super::{bar, distribution, edge, gauge, node, scaffold, timeseries};

/// Build the search results scenario.
///
/// Nodes:
/// - `search_pipeline`: pipeline stats + alignment score chart
/// - `kmer_index`: seed density heatmap + index stats
/// - `hit_analysis`: hit distribution + score distribution
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn search_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Sequence Search Pipeline",
        "BLAST-like seed-and-extend: k-mer seeding → Smith-Waterman extension → score filter",
    );

    let db_seqs = synthetic_database();
    let query = synthetic_query();

    let config = SearchConfig {
        kmer_size: 4,
        min_score: 4.0,
        extension_window: 30,
        ..SearchConfig::default()
    };

    let db_encoded: Vec<Vec<u32>> = db_seqs.iter().map(|s| encode_dna(s)).collect();
    let query_encoded = encode_dna(&query);

    let indexed: Vec<(u32, &[u32])> = db_encoded
        .iter()
        .enumerate()
        .map(|(i, s)| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "synthetic database has < u32::MAX sequences"
            )]
            let id = i as u32;
            (id, s.as_slice())
        })
        .collect();
    let kmer_index_ref = KmerIndex::build(4, &indexed);

    let pipeline = CpuSearchPipeline::new(db_encoded, config);
    let result = pipeline.search(&query_encoded);

    // ── Pipeline stats node ──────────────────────────────────────────────

    let hit_scores: Vec<f64> = result.hits.iter().map(|h| h.score).collect();
    let hit_db_ids: Vec<String> = result
        .hits
        .iter()
        .map(|h| format!("seq_{}", h.db_seq_id))
        .collect();

    let score_mean = if hit_scores.is_empty() {
        0.0
    } else {
        hit_scores.iter().sum::<f64>() / hit_scores.len() as f64
    };

    s.ecosystem.primals.push(node(
        "search_pipeline",
        "Seed-and-Extend Pipeline",
        "pipeline",
        0.0,
        0.0,
        &["search.seed_extend", "search.smith_waterman"],
        vec![
            bar(
                "hit-scores",
                "Alignment Scores by Hit",
                hit_db_ids,
                hit_scores.clone(),
                "SW score",
            ),
            gauge(
                "seed-count",
                "Seed Hits (k-mer matches)",
                result.n_seeds as f64,
                0.0,
                1000.0,
                "seeds",
                [10.0, 500.0],
                [0.0, 10.0],
            ),
            gauge(
                "extension-count",
                "SW Extensions (unique diagonals)",
                result.n_extensions as f64,
                0.0,
                500.0,
                "extensions",
                [5.0, 200.0],
                [0.0, 5.0],
            ),
            gauge(
                "hit-count",
                "Reported Hits (above threshold)",
                result.hits.len() as f64,
                0.0,
                100.0,
                "hits",
                [1.0, 50.0],
                [0.0, 1.0],
            ),
        ],
        vec![ThresholdRange {
            label: "Good alignment".into(),
            min: 10.0,
            max: f64::INFINITY,
            status: "normal".into(),
        }],
    ));

    // ── K-mer index node ─────────────────────────────────────────────────

    let query_kmer_hits: Vec<f64> = (0..query_encoded.len().saturating_sub(3))
        .map(|pos| {
            let kmer = &query_encoded[pos..pos + 4];
            kmer_index_ref.lookup(kmer).len() as f64
        })
        .collect();
    let positions: Vec<f64> = (0..query_kmer_hits.len()).map(|i| i as f64).collect();

    s.ecosystem.primals.push(node(
        "kmer_index",
        "K-mer Seed Index",
        "index",
        300.0,
        0.0,
        &["search.kmer_index"],
        vec![
            timeseries(
                "seed-density",
                "Seed Density Along Query",
                "Query position",
                "Hits in DB",
                "count",
                positions,
                query_kmer_hits,
            ),
            gauge(
                "index-kmers",
                "Distinct K-mers in Index",
                kmer_index_ref.n_kmers() as f64,
                0.0,
                10000.0,
                "k-mers",
                [100.0, 5000.0],
                [0.0, 100.0],
            ),
        ],
        vec![],
    ));

    // ── Hit analysis node ────────────────────────────────────────────────

    let db_names: Vec<String> = (0..db_seqs.len()).map(|i| format!("seq_{i}")).collect();
    let mut hits_per_seq = vec![0.0_f64; db_seqs.len()];
    for hit in &result.hits {
        if (hit.db_seq_id as usize) < hits_per_seq.len() {
            hits_per_seq[hit.db_seq_id as usize] += 1.0;
        }
    }

    s.ecosystem.primals.push(node(
        "hit_analysis",
        "Hit Analysis",
        "analysis",
        150.0,
        300.0,
        &["search.hit_analysis"],
        vec![
            bar(
                "hits-per-seq",
                "Hits Per Database Sequence",
                db_names,
                hits_per_seq,
                "hits",
            ),
            distribution(
                "score-distribution",
                "Alignment Score Distribution",
                "SW score",
                hit_scores,
                score_mean,
                0.0,
                10.0,
            ),
        ],
        vec![],
    ));

    let edges = vec![
        edge("kmer_index", "search_pipeline", "seed hits → extension"),
        edge(
            "search_pipeline",
            "hit_analysis",
            "filtered hits → analysis",
        ),
    ];

    (s, edges)
}

fn encode_dna(seq: &[u8]) -> Vec<u32> {
    seq.iter()
        .map(|&b| match b {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => 4,
        })
        .collect()
}

fn synthetic_database() -> Vec<Vec<u8>> {
    vec![
        b"ACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(),
        b"TGCATGCATGCATGCATGCATGCATGCATGCA".to_vec(),
        b"AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT".to_vec(),
        b"ACGTTTTTACGTTTTTACGTTTTTACGTTTTT".to_vec(),
        b"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG".to_vec(),
    ]
}

fn synthetic_query() -> Vec<u8> {
    b"ACGTACGTTTTTACGT".to_vec()
}
