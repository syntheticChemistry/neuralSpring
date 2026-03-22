// SPDX-License-Identifier: AGPL-3.0-or-later

//! Streaming I/O parsers for scientific file formats.
//!
//! All parsers follow `specs/STREAMING_IO_REQUIREMENTS.md`:
//!
//! - Accept `impl BufRead` (files, gzip streams, network, test cursors)
//! - Yield records via `Iterator` with O(`record_size`) memory
//! - Safe Rust only (`#![forbid(unsafe_code)]`)
//! - Round-trip validated via integration tests
//!
//! ## Formats
//!
//! | Module | Format | Spec Requirement |
//! |--------|--------|-----------------|
//! | [`fasta`] | FASTA (`.fa`, `.fasta`, `.fna`) | R-01, R-03 |
//! | [`fastq`] | FASTQ (`.fq`, `.fastq`, `.fq.gz`) | R-01, R-03 |
//! | [`vcf`] | VCF v4.x (`.vcf`) | R-01, R-03 |

pub mod fasta;
pub mod fastq;
pub mod vcf;

/// Default initial capacity for line-read buffers in streaming parsers.
///
/// Sized to avoid frequent reallocations for typical bioinformatics records
/// (FASTQ reads ~150 bp, FASTA headers ~80 chars) while keeping memory
/// footprint small for massively parallel parsing.
pub(crate) const LINE_BUF_CAPACITY: usize = 256;

/// Default initial capacity for VCF line buffers (wider records).
pub(crate) const VCF_LINE_BUF_CAPACITY: usize = 512;

/// Strips trailing `\r` / `\n` in place (what [`std::io::BufRead::read_line`] leaves).
///
/// Avoids allocating a second [`String`] for a trimmed copy of each line.
#[inline]
pub(crate) fn trim_end_newlines_in_place(s: &mut String) {
    while matches!(s.as_bytes().last(), Some(b'\n' | b'\r')) {
        s.pop();
    }
}
