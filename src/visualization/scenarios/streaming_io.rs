// SPDX-License-Identifier: AGPL-3.0-or-later

//! Streaming I/O quality scenario builder.
//!
//! Parses synthetic FASTQ/FASTA/VCF data with real streaming parsers and
//! visualizes quality distributions, sequence length histograms, and variant
//! density — exactly what a scientist would look at before running an analysis.

#![expect(
    clippy::cast_precision_loss,
    reason = "index-to-f64 conversions for visualization axes"
)]
#![expect(
    clippy::similar_names,
    reason = "fastq_data/fasta_data are intentionally parallel-named for the three formats"
)]

use crate::streaming::fasta::{FastaReader, FastaRecord};
use crate::streaming::fastq::{FastqReader, FastqRecord};
use crate::streaming::vcf::VcfReader;
use crate::visualization::types::{NeuralScenario, ScenarioEdge};

use super::{bar, distribution, edge, gauge, node, scaffold, timeseries};

/// Build the streaming I/O quality scenario.
///
/// Nodes:
/// - `fastq_quality`: read quality distribution + per-position quality trace
/// - `fasta_lengths`: sequence length histogram + GC content gauge
/// - `vcf_variants`: variant density along chromosome + type breakdown
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn streaming_io_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Streaming I/O Quality",
        "Real-time quality assessment of FASTQ, FASTA, and VCF streams — \
         the first thing a scientist sees before running a pipeline",
    );

    // ── FASTQ Quality ────────────────────────────────────────────────────

    let fastq_data = synthetic_fastq();
    let reader = FastqReader::new(std::io::Cursor::new(fastq_data));
    let records: Vec<FastqRecord> = reader.filter_map(Result::ok).collect();

    let mean_qualities: Vec<f64> = records.iter().map(FastqRecord::mean_quality).collect();

    let overall_mean = if mean_qualities.is_empty() {
        0.0
    } else {
        mean_qualities.iter().sum::<f64>() / mean_qualities.len() as f64
    };
    let overall_std = if mean_qualities.len() < 2 {
        0.0
    } else {
        let var = mean_qualities
            .iter()
            .map(|q| (q - overall_mean).powi(2))
            .sum::<f64>()
            / (mean_qualities.len() - 1) as f64;
        var.sqrt()
    };

    let max_len = records.iter().map(FastqRecord::len).max().unwrap_or(0);
    let mut per_position_mean = vec![0.0_f64; max_len];
    let mut per_position_count = vec![0_u32; max_len];
    for rec in &records {
        for (pos, &score) in rec.phred_scores().iter().enumerate() {
            per_position_mean[pos] += f64::from(score);
            per_position_count[pos] += 1;
        }
    }
    for (mean, count) in per_position_mean.iter_mut().zip(&per_position_count) {
        if *count > 0 {
            *mean /= f64::from(*count);
        }
    }
    let positions: Vec<f64> = (0..max_len).map(|i| i as f64).collect();

    s.ecosystem.primals.push(node(
        "fastq_quality",
        "FASTQ Quality Assessment",
        "parser",
        0.0,
        0.0,
        &["streaming.fastq", "quality.phred"],
        vec![
            distribution(
                "read-quality-dist",
                "Mean Quality per Read (Phred)",
                "Q-score",
                mean_qualities,
                overall_mean,
                overall_std,
                30.0,
            ),
            timeseries(
                "per-position-quality",
                "Per-Position Mean Quality",
                "Base position",
                "Mean Phred score",
                "Q",
                positions,
                per_position_mean,
            ),
            gauge(
                "read-count",
                "Total Reads Parsed",
                records.len() as f64,
                0.0,
                10000.0,
                "reads",
                [100.0, 5000.0],
                [0.0, 100.0],
            ),
        ],
        vec![],
    ));

    // ── FASTA Lengths + GC ───────────────────────────────────────────────

    let fasta_data = synthetic_fasta();
    let reader = FastaReader::new(std::io::Cursor::new(fasta_data));
    let fa_records: Vec<FastaRecord> = reader.filter_map(Result::ok).collect();

    let lengths: Vec<f64> = fa_records.iter().map(|r| r.len() as f64).collect();
    let seq_names: Vec<String> = fa_records.iter().map(|r| r.id().to_string()).collect();

    let gc_fractions: Vec<f64> = fa_records
        .iter()
        .map(|r| {
            let gc = r
                .seq()
                .iter()
                .filter(|&&b| b == b'G' || b == b'C' || b == b'g' || b == b'c')
                .count();
            if r.is_empty() {
                0.0
            } else {
                gc as f64 / r.len() as f64
            }
        })
        .collect();

    let mean_gc = if gc_fractions.is_empty() {
        0.0
    } else {
        gc_fractions.iter().sum::<f64>() / gc_fractions.len() as f64
    };

    s.ecosystem.primals.push(node(
        "fasta_lengths",
        "FASTA Sequence Statistics",
        "parser",
        300.0,
        0.0,
        &["streaming.fasta", "stats.length", "stats.gc"],
        vec![
            bar(
                "seq-lengths",
                "Sequence Length by Entry",
                seq_names,
                lengths,
                "bases",
            ),
            gauge(
                "mean-gc",
                "Mean GC Content",
                mean_gc,
                0.0,
                1.0,
                "fraction",
                [0.35, 0.65],
                [0.2, 0.35],
            ),
            gauge(
                "seq-count",
                "Total Sequences",
                fa_records.len() as f64,
                0.0,
                10000.0,
                "sequences",
                [10.0, 5000.0],
                [0.0, 10.0],
            ),
        ],
        vec![],
    ));

    // ── VCF Variant Density ──────────────────────────────────────────────

    let vcf_data = synthetic_vcf();
    let vcf_records: Vec<_> = VcfReader::new(std::io::Cursor::new(vcf_data))
        .map_or_else(|_| Vec::new(), |r| r.filter_map(Result::ok).collect());

    let variant_positions: Vec<f64> = vcf_records
        .iter()
        .map(|r: &crate::streaming::vcf::VcfRecord| r.pos() as f64)
        .collect();
    let variant_indices: Vec<f64> = (0..vcf_records.len()).map(|i| i as f64).collect();

    let mut type_counts = std::collections::HashMap::new();
    for rec in &vcf_records {
        let vtype = classify_variant(rec.ref_allele(), rec.alt_alleles());
        *type_counts.entry(vtype).or_insert(0_u32) += 1;
    }
    let type_names: Vec<String> = type_counts.keys().cloned().collect();
    let type_vals: Vec<f64> = type_names
        .iter()
        .map(|n| f64::from(*type_counts.get(n).unwrap_or(&0)))
        .collect();

    s.ecosystem.primals.push(node(
        "vcf_variants",
        "VCF Variant Analysis",
        "parser",
        150.0,
        300.0,
        &["streaming.vcf", "stats.variant_density"],
        vec![
            timeseries(
                "variant-positions",
                "Variant Positions Along Chromosome",
                "Variant index",
                "Genomic position",
                "bp",
                variant_indices,
                variant_positions,
            ),
            bar(
                "variant-types",
                "Variant Type Breakdown",
                type_names,
                type_vals,
                "count",
            ),
            gauge(
                "variant-count",
                "Total Variants",
                vcf_records.len() as f64,
                0.0,
                100_000.0,
                "variants",
                [100.0, 50_000.0],
                [0.0, 100.0],
            ),
        ],
        vec![],
    ));

    let edges = vec![
        edge("fastq_quality", "fasta_lengths", "read QC → assembly input"),
        edge(
            "fasta_lengths",
            "vcf_variants",
            "reference → variant calling",
        ),
    ];

    (s, edges)
}

fn classify_variant(ref_allele: &str, alt_alleles: &[String]) -> String {
    let alt = alt_alleles.first().map_or(".", String::as_str);
    match (ref_allele.len(), alt.len()) {
        (1, 1) => "SNV".into(),
        (r, a) if r > a => "Deletion".into(),
        (r, a) if r < a => "Insertion".into(),
        _ => "Complex".into(),
    }
}

fn synthetic_fastq() -> Vec<u8> {
    let mut data = Vec::new();
    let bases = [b'A', b'C', b'G', b'T'];
    let quals_high: &[u8] = b"IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII";
    let quals_med: &[u8] = b"555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555";

    for i in 0..20 {
        let len = 100 + (i % 5) * 10;
        let seq: Vec<u8> = (0..len).map(|j| bases[(i + j) % 4]).collect();
        let qual = if i % 3 == 0 {
            &quals_med[..len]
        } else {
            &quals_high[..len]
        };

        data.extend_from_slice(format!("@read_{i} length={len}\n").as_bytes());
        data.extend_from_slice(&seq);
        data.push(b'\n');
        data.extend_from_slice(b"+\n");
        data.extend_from_slice(qual);
        data.push(b'\n');
    }
    data
}

fn synthetic_fasta() -> Vec<u8> {
    let mut data = Vec::new();
    let seqs = [
        (
            "chr1_fragment",
            b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" as &[u8],
        ),
        (
            "chr2_fragment",
            b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA",
        ),
        (
            "orf_predicted",
            b"ATGCCCGGGAAATTTCCCGGGAAATTTAAACCCGGGTTTAAACCCGGGTTTCCC",
        ),
        (
            "ribosomal_16s",
            b"GGGGGGCCCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT",
        ),
        ("plasmid_rep", b"ACACACACACGTGTGTGTGTACACACACACGTGTGTGTGT"),
        (
            "mobile_element",
            b"AAAAACCCCCGGGGGTTTTTTAAAAACCCCCGGGGGTTTTTT",
        ),
    ];

    for (name, seq) in &seqs {
        data.extend_from_slice(format!(">{name} synthetic\n").as_bytes());
        for chunk in seq.chunks(60) {
            data.extend_from_slice(chunk);
            data.push(b'\n');
        }
    }
    data
}

fn synthetic_vcf() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"##fileformat=VCFv4.2\n");
    data.extend_from_slice(b"##source=neuralSpring-synthetic\n");
    data.extend_from_slice(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");

    let variants = [
        ("chr1", 1000, "A", "G", 30.0),
        ("chr1", 1500, "C", "T", 45.0),
        ("chr1", 2200, "G", "A", 22.0),
        ("chr1", 3100, "AT", "A", 38.0),
        ("chr1", 4500, "T", "TA", 41.0),
        ("chr1", 5800, "C", "G", 55.0),
        ("chr1", 6000, "A", "C", 28.0),
        ("chr1", 7200, "GG", "G", 33.0),
        ("chr1", 8500, "T", "TAA", 42.0),
        ("chr1", 9000, "A", "T", 50.0),
        ("chr1", 10200, "C", "A", 35.0),
        ("chr1", 11000, "G", "T", 60.0),
    ];

    for (chrom, pos, ref_a, alt, qual) in &variants {
        data.extend_from_slice(
            format!("{chrom}\t{pos}\t.\t{ref_a}\t{alt}\t{qual}\tPASS\t.\n").as_bytes(),
        );
    }
    data
}
