// SPDX-License-Identifier: AGPL-3.0-or-later

//! Streaming FASTA parser — O(`record_size`) memory, zero full-file buffering.
//!
//! Implements requirement R-01 from `specs/STREAMING_IO_REQUIREMENTS.md`.
//! Prerequisite for all MSA and BLAST-like pipelines (see
//! `specs/MSA_PIPELINE_SCOPE.md`, `specs/BLAST_LIKE_SEARCH_SCOPE.md`).
//!
//! ## FASTA format
//!
//! Each record consists of:
//! 1. A header line starting with `>` (identifier + optional description)
//! 2. One or more sequence lines (nucleotides or amino acids)
//!
//! Records are delimited by the next `>` header or end of file.
//!
//! ## Usage
//!
//! ```
//! use std::io::Cursor;
//! use neural_spring::streaming::fasta::{FastaReader, FastaRecord};
//!
//! let data = b">seq1 example\nACGTACGT\nTGCA\n>seq2\nAAAA\n";
//! let reader = FastaReader::new(Cursor::new(data));
//! let records: Vec<FastaRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
//! assert_eq!(records.len(), 2);
//! assert_eq!(records[0].id(), "seq1");
//! assert_eq!(records[0].seq(), b"ACGTACGTTGCA");
//! assert_eq!(records[1].seq(), b"AAAA");
//! ```

use std::io::BufRead;

/// A single FASTA record (header + concatenated sequence lines).
///
/// Sequence data is stored in an owned [`Vec`] because a record may span many
/// physical lines; there is no single borrowable slice into one read buffer.
/// Zero-copy line reads are handled inside [`FastaReader`] (no extra `String`
/// per line beyond the buffer that [`BufRead::read_line`] fills).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastaRecord {
    header: String,
    sequence: Vec<u8>,
}

impl FastaRecord {
    /// Sequence identifier (header without the leading `>` and any description).
    #[must_use]
    pub fn id(&self) -> &str {
        self.header
            .split_ascii_whitespace()
            .next()
            .unwrap_or(&self.header)
    }

    /// Full header line (without the leading `>`).
    #[must_use]
    pub fn header(&self) -> &str {
        &self.header
    }

    /// Raw sequence bytes (concatenated from all continuation lines).
    #[must_use]
    pub fn seq(&self) -> &[u8] {
        &self.sequence
    }

    /// Sequence length in residues/bases.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Encode DNA bases as integer indices: A=0, C=1, G=2, T=3, N=4.
    ///
    /// Unknown bases map to 4. Useful for GPU kernel input (e.g.
    /// `SmithWatermanGpu` expects `u32` indices).
    #[must_use]
    pub fn encode_dna(&self) -> Vec<u32> {
        self.sequence
            .iter()
            .map(|&b| match b {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => 4,
            })
            .collect()
    }

    /// Write this record in FASTA format (60 chars per line).
    ///
    /// # Errors
    ///
    /// Returns I/O errors from the underlying writer.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        writeln!(w, ">{}", self.header)?;
        for chunk in self.sequence.chunks(60) {
            w.write_all(chunk)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }
}

/// Parse error for malformed FASTA records.
#[derive(Debug)]
pub enum FastaError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// First non-blank line does not start with `>`.
    InvalidHeader(String),
}

impl std::error::Error for FastaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::InvalidHeader(_) => None,
        }
    }
}

impl std::fmt::Display for FastaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidHeader(h) => write!(f, "header does not start with '>': {h:?}"),
        }
    }
}

impl From<std::io::Error> for FastaError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Streaming FASTA reader — yields one [`FastaRecord`] per iteration.
///
/// Handles multiline sequences by concatenating continuation lines.
/// Memory footprint is O(`record_size`), never O(`file_size`).
pub struct FastaReader<R> {
    reader: R,
    line_buf: String,
    pending_header: Option<String>,
}

impl<R: BufRead> FastaReader<R> {
    /// Create a new streaming FASTA reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_buf: String::with_capacity(super::LINE_BUF_CAPACITY),
            pending_header: None,
        }
    }

    /// Reads one line, strips trailing newlines in place, then moves the buffer out
    /// (avoids allocating a second [`String`] for the trimmed line).
    fn read_line_owned(&mut self) -> Result<Option<String>, FastaError> {
        self.line_buf.clear();
        let n = self.reader.read_line(&mut self.line_buf)?;
        if n == 0 {
            return Ok(None);
        }
        super::trim_end_newlines_in_place(&mut self.line_buf);
        Ok(Some(std::mem::take(&mut self.line_buf)))
    }
}

impl<R: BufRead> Iterator for FastaReader<R> {
    type Item = Result<FastaRecord, FastaError>;

    fn next(&mut self) -> Option<Self::Item> {
        let header = if let Some(h) = self.pending_header.take() {
            h
        } else {
            // Find the first header line, skipping blanks.
            loop {
                match self.read_line_owned() {
                    Ok(Some(line)) if line.is_empty() => {}
                    Ok(Some(mut line)) => {
                        if !line.starts_with('>') {
                            return Some(Err(FastaError::InvalidHeader(line)));
                        }
                        break line.split_off(1);
                    }
                    Ok(None) => return None,
                    Err(e) => return Some(Err(e)),
                }
            }
        };

        let mut sequence = Vec::with_capacity(super::LINE_BUF_CAPACITY);

        loop {
            match self.read_line_owned() {
                Ok(Some(line)) if line.is_empty() => {}
                Ok(Some(mut line)) if line.starts_with('>') => {
                    self.pending_header = Some(line.split_off(1));
                    break;
                }
                Ok(Some(line)) => {
                    sequence.extend_from_slice(line.as_bytes());
                }
                Ok(None) => break,
                Err(e) => return Some(Err(e)),
            }
        }

        Some(Ok(FastaRecord { header, sequence }))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse_all(data: &[u8]) -> Result<Vec<FastaRecord>, FastaError> {
        FastaReader::new(Cursor::new(data)).collect()
    }

    #[test]
    fn single_record_single_line() {
        let data = b">seq1\nACGT\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id(), "seq1");
        assert_eq!(records[0].seq(), b"ACGT");
        assert_eq!(records[0].len(), 4);
    }

    #[test]
    fn single_record_multiline() {
        let data = b">seq1 description\nACGT\nTGCA\nAAAA\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id(), "seq1");
        assert_eq!(records[0].header(), "seq1 description");
        assert_eq!(records[0].seq(), b"ACGTTGCAAAAA");
        assert_eq!(records[0].len(), 12);
    }

    #[test]
    fn multiple_records() {
        let data = b">s1\nACGT\n>s2\nTGCA\n>s3\nAAAA\nCCCC\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].id(), "s1");
        assert_eq!(records[0].seq(), b"ACGT");
        assert_eq!(records[1].id(), "s2");
        assert_eq!(records[1].seq(), b"TGCA");
        assert_eq!(records[2].id(), "s3");
        assert_eq!(records[2].seq(), b"AAAACCCC");
    }

    #[test]
    fn empty_file() {
        let records = parse_all(b"").unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn blank_lines_between_records() {
        let data = b">s1\nACGT\n\n\n>s2\nTGCA\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn invalid_header() {
        let data = b"NOT_HEADER\nACGT\n";
        let err = parse_all(data).unwrap_err();
        assert!(
            matches!(err, FastaError::InvalidHeader(_)),
            "expected InvalidHeader, got {err}"
        );
    }

    #[test]
    fn encode_dna() {
        let data = b">s1\nACGTN\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records[0].encode_dna(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn encode_dna_lowercase() {
        let data = b">s1\nacgtn\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records[0].encode_dna(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn round_trip_fidelity() {
        let original = b">seq1 paired end\nACGTNNACGT\nTGCA\n>seq2\nAAAA\n";
        let records = parse_all(original).unwrap();
        assert_eq!(records.len(), 2);

        let mut written = Vec::new();
        for rec in &records {
            rec.write_to(&mut written).unwrap();
        }

        let reparsed = parse_all(&written).unwrap();
        assert_eq!(records, reparsed, "round-trip must be exact");
    }

    #[test]
    fn windows_line_endings() {
        let data = b">s1\r\nACGT\r\nTGCA\r\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq(), b"ACGTTGCA");
    }

    #[test]
    fn protein_sequence() {
        let data = b">prot1\nMKTAYIAK\nLDFGSR\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records[0].seq(), b"MKTAYIAKLDFGSR");
        assert_eq!(records[0].len(), 14);
    }

    #[test]
    fn large_sequence_streaming() {
        let seq: Vec<u8> = (0..10_000).map(|i| b"ACGT"[i % 4]).collect();
        let mut data = Vec::new();
        data.extend_from_slice(b">large\n");
        for chunk in seq.chunks(80) {
            data.extend_from_slice(chunk);
            data.push(b'\n');
        }

        let records = parse_all(&data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].len(), 10_000);
    }

    #[test]
    fn write_wraps_at_60() {
        let data = b">s1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
        let records = parse_all(data).unwrap();
        let mut out = Vec::new();
        records[0].write_to(&mut out).unwrap();
        let lines: Vec<&str> = std::str::from_utf8(&out).unwrap().lines().collect();
        assert_eq!(lines[0], ">s1");
        assert_eq!(lines[1].len(), 60);
    }

    #[test]
    fn error_display() {
        let e = FastaError::InvalidHeader("bad".into());
        assert!(e.to_string().contains("does not start with '>'"));
    }

    #[test]
    fn empty_sequence() {
        let data = b">empty_seq\n>next\nACGT\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 2);
        assert!(records[0].is_empty());
        assert_eq!(records[1].seq(), b"ACGT");
    }

    #[test]
    fn no_trailing_newline() {
        let data = b">s1\nACGT";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq(), b"ACGT");
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn dna_sequence(max_len: usize) -> impl Strategy<Value = Vec<u8>> {
            proptest::collection::vec(
                prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')],
                1..=max_len,
            )
        }

        fn header_string() -> impl Strategy<Value = String> {
            "[a-zA-Z0-9_.]{1,40}"
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn write_then_parse_roundtrip(
                hdr in header_string(),
                seq in dna_sequence(500),
            ) {
                let mut fasta_text = format!(">{hdr}\n").into_bytes();
                fasta_text.extend_from_slice(&seq);
                fasta_text.push(b'\n');

                let parsed = parse_all(&fasta_text).unwrap();
                prop_assert_eq!(parsed.len(), 1);
                prop_assert_eq!(parsed[0].header(), hdr.as_str());
                prop_assert_eq!(parsed[0].seq(), seq.as_slice());

                let mut written = Vec::new();
                parsed[0].write_to(&mut written).unwrap();

                let reparsed = parse_all(&written).unwrap();
                prop_assert_eq!(reparsed.len(), 1);
                prop_assert_eq!(reparsed[0].seq(), seq.as_slice());
            }

            #[test]
            fn sequence_length_preserved(
                seq in dna_sequence(1000),
            ) {
                let fasta = format!(">test\n{}\n", std::str::from_utf8(&seq).unwrap()).into_bytes();
                let parsed = parse_all(&fasta).unwrap();
                prop_assert_eq!(parsed[0].len(), seq.len());
                prop_assert!(!parsed[0].is_empty());
            }
        }
    }
}
