// SPDX-License-Identifier: AGPL-3.0-or-later

//! Streaming FASTQ parser — O(`record_size`) memory, zero full-file buffering.
//!
//! Implements requirement R-01 from `specs/STREAMING_IO_REQUIREMENTS.md`.
//!
//! ## FASTQ format
//!
//! Each record consists of exactly 4 lines:
//!
//! 1. Header line starting with `@` (sequence identifier + optional description)
//! 2. Sequence line (A/C/G/T/N nucleotides)
//! 3. Separator line starting with `+` (optionally repeats the identifier)
//! 4. Quality line (Phred+33 ASCII-encoded quality scores, same length as sequence)
//!
//! ## Usage
//!
//! ```
//! use std::io::Cursor;
//! use neural_spring::streaming::fastq::{FastqReader, FastqRecord};
//!
//! let data = b"@read1\nACGT\n+\nIIII\n";
//! let reader = FastqReader::new(Cursor::new(data));
//! let records: Vec<FastqRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
//! assert_eq!(records.len(), 1);
//! assert_eq!(records[0].id(), "read1");
//! assert_eq!(records[0].seq(), b"ACGT");
//! ```

use std::io::BufRead;

/// A single FASTQ record (4-line group).
///
/// Sequence and quality are stored as owned [`Vec`]s so each [`FastqRecord`]
/// outlives the reader's line buffer. Line assembly avoids an extra [`String`]
/// allocation per line via in-place trimming and [`std::mem::take`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastqRecord {
    header: String,
    sequence: Vec<u8>,
    quality: Vec<u8>,
}

impl FastqRecord {
    /// Sequence identifier (header without the leading `@` and any description).
    #[must_use]
    pub fn id(&self) -> &str {
        self.header
            .split_ascii_whitespace()
            .next()
            .unwrap_or(&self.header)
    }

    /// Full header line (without the leading `@`).
    #[must_use]
    pub fn header(&self) -> &str {
        &self.header
    }

    /// Raw nucleotide sequence bytes (A/C/G/T/N).
    #[must_use]
    pub fn seq(&self) -> &[u8] {
        &self.sequence
    }

    /// Phred+33 quality scores (same length as sequence).
    #[must_use]
    pub fn quality(&self) -> &[u8] {
        &self.quality
    }

    /// Sequence length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Whether the record has zero-length sequence.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Decode quality scores to Phred integers (Q = ASCII - 33).
    #[must_use]
    pub fn phred_scores(&self) -> Vec<u8> {
        self.quality.iter().map(|&q| q.saturating_sub(33)).collect()
    }

    /// Mean Phred quality score.
    #[must_use]
    pub fn mean_quality(&self) -> f64 {
        if self.quality.is_empty() {
            return 0.0;
        }
        let sum: u64 = self
            .quality
            .iter()
            .map(|&q| u64::from(q.saturating_sub(33)))
            .sum();
        #[expect(clippy::cast_precision_loss, reason = "quality scores are small")]
        let mean = sum as f64 / self.quality.len() as f64;
        mean
    }

    /// Write this record in FASTQ format to a writer.
    ///
    /// # Errors
    ///
    /// Returns I/O errors from the underlying writer.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(b"@")?;
        w.write_all(self.header.as_bytes())?;
        w.write_all(b"\n")?;
        w.write_all(&self.sequence)?;
        w.write_all(b"\n+\n")?;
        w.write_all(&self.quality)?;
        w.write_all(b"\n")?;
        Ok(())
    }
}

/// Parse error for malformed FASTQ records.
#[derive(Debug)]
pub enum FastqError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// Header line does not start with `@`.
    InvalidHeader(String),
    /// Missing sequence, separator, or quality line.
    TruncatedRecord(String),
    /// Quality line length does not match sequence length.
    LengthMismatch {
        /// Record header (without the leading `@`).
        header: String,
        /// Nucleotide sequence length in bases.
        seq_len: usize,
        /// Quality string length in bytes.
        qual_len: usize,
    },
    /// Separator line does not start with `+`.
    InvalidSeparator(String),
}

impl std::fmt::Display for FastqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidHeader(h) => write!(f, "header does not start with '@': {h:?}"),
            Self::TruncatedRecord(h) => write!(f, "truncated record after header: {h:?}"),
            Self::LengthMismatch {
                header,
                seq_len,
                qual_len,
            } => write!(
                f,
                "quality length ({qual_len}) != sequence length ({seq_len}) for {header:?}"
            ),
            Self::InvalidSeparator(s) => write!(f, "separator does not start with '+': {s:?}"),
        }
    }
}

impl std::error::Error for FastqError {}

impl From<std::io::Error> for FastqError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Streaming FASTQ reader — yields one [`FastqRecord`] per iteration.
///
/// Memory footprint is O(`record_size`), never O(`file_size`).
pub struct FastqReader<R> {
    reader: R,
    line_buf: String,
}

impl<R: BufRead> FastqReader<R> {
    /// Create a new streaming FASTQ reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_buf: String::with_capacity(super::LINE_BUF_CAPACITY),
        }
    }

    fn read_line_owned(&mut self) -> Result<Option<String>, FastqError> {
        self.line_buf.clear();
        let n = self.reader.read_line(&mut self.line_buf)?;
        if n == 0 {
            return Ok(None);
        }
        super::trim_end_newlines_in_place(&mut self.line_buf);
        Ok(Some(std::mem::take(&mut self.line_buf)))
    }
}

impl<R: BufRead> Iterator for FastqReader<R> {
    type Item = Result<FastqRecord, FastqError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut header_line = loop {
            match self.read_line_owned() {
                Ok(Some(line)) if line.is_empty() => {}
                Ok(Some(line)) => break line,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
        };

        if !header_line.starts_with('@') {
            return Some(Err(FastqError::InvalidHeader(header_line)));
        }
        let header = header_line.split_off(1);

        let sequence = match self.read_line_owned() {
            Ok(Some(s)) => s.into_bytes(),
            Ok(None) => return Some(Err(FastqError::TruncatedRecord(header))),
            Err(e) => return Some(Err(e)),
        };

        let separator = match self.read_line_owned() {
            Ok(Some(s)) => s,
            Ok(None) => return Some(Err(FastqError::TruncatedRecord(header))),
            Err(e) => return Some(Err(e)),
        };

        if !separator.starts_with('+') {
            return Some(Err(FastqError::InvalidSeparator(separator)));
        }

        let quality = match self.read_line_owned() {
            Ok(Some(q)) => q.into_bytes(),
            Ok(None) => return Some(Err(FastqError::TruncatedRecord(header))),
            Err(e) => return Some(Err(e)),
        };

        if sequence.len() != quality.len() {
            return Some(Err(FastqError::LengthMismatch {
                header,
                seq_len: sequence.len(),
                qual_len: quality.len(),
            }));
        }

        Some(Ok(FastqRecord {
            header,
            sequence,
            quality,
        }))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse_all(data: &[u8]) -> Result<Vec<FastqRecord>, FastqError> {
        FastqReader::new(Cursor::new(data)).collect()
    }

    #[test]
    fn single_record() {
        let data = b"@read1 some desc\nACGTACGT\n+\nIIIIIIII\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id(), "read1");
        assert_eq!(records[0].header(), "read1 some desc");
        assert_eq!(records[0].seq(), b"ACGTACGT");
        assert_eq!(records[0].quality(), b"IIIIIIII");
        assert_eq!(records[0].len(), 8);
        assert!(!records[0].is_empty());
    }

    #[test]
    fn multiple_records() {
        let data = b"@r1\nACGT\n+\nIIII\n@r2\nTGCA\n+\nJJJJ\n@r3\nAAAA\n+\n!!!!\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].id(), "r1");
        assert_eq!(records[1].id(), "r2");
        assert_eq!(records[2].id(), "r3");
        assert_eq!(records[2].seq(), b"AAAA");
    }

    #[test]
    fn phred_scores() {
        let data = b"@r1\nACGT\n+\n!\"#$\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records[0].phred_scores(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn mean_quality() {
        let data = b"@r1\nAA\n+\n#%\n";
        let records = parse_all(data).unwrap();
        let mean = records[0].mean_quality();
        assert!((mean - 3.0).abs() < 1e-10, "mean = {mean}");
    }

    #[test]
    fn empty_file() {
        let records = parse_all(b"").unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn blank_lines_between_records() {
        let data = b"@r1\nACGT\n+\nIIII\n\n\n@r2\nTGCA\n+\nJJJJ\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn invalid_header() {
        let data = b"NOT_AT_SIGN\nACGT\n+\nIIII\n";
        let err = parse_all(data).unwrap_err();
        assert!(
            matches!(err, FastqError::InvalidHeader(_)),
            "expected InvalidHeader, got {err}"
        );
    }

    #[test]
    fn truncated_record() {
        let data = b"@r1\nACGT\n";
        let err = parse_all(data).unwrap_err();
        assert!(
            matches!(err, FastqError::TruncatedRecord(_)),
            "expected TruncatedRecord, got {err}"
        );
    }

    #[test]
    fn length_mismatch() {
        let data = b"@r1\nACGT\n+\nII\n";
        let err = parse_all(data).unwrap_err();
        assert!(
            matches!(err, FastqError::LengthMismatch { .. }),
            "expected LengthMismatch, got {err}"
        );
    }

    #[test]
    fn invalid_separator() {
        let data = b"@r1\nACGT\nNOTPLUS\nIIII\n";
        let err = parse_all(data).unwrap_err();
        assert!(
            matches!(err, FastqError::InvalidSeparator(_)),
            "expected InvalidSeparator, got {err}"
        );
    }

    #[test]
    fn round_trip_fidelity() {
        let original = b"@read1 paired\nACGTNNACGT\n+\nIIII!!IIII\n\
                          @read2\nTTTTGGGG\n+\nJJJJKKKK\n";
        let records = parse_all(original).unwrap();
        assert_eq!(records.len(), 2);

        let mut written = Vec::new();
        for rec in &records {
            rec.write_to(&mut written).unwrap();
        }

        let reparsed = parse_all(&written).unwrap();
        assert_eq!(records, reparsed, "round-trip must be bit-exact");
    }

    #[test]
    fn windows_line_endings() {
        let data = b"@r1\r\nACGT\r\n+\r\nIIII\r\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq(), b"ACGT");
    }

    #[test]
    fn n_bases_accepted() {
        let data = b"@r1\nACNNGT\n+\nIIIIII\n";
        let records = parse_all(data).unwrap();
        assert_eq!(records[0].seq(), b"ACNNGT");
    }

    #[test]
    fn large_record_streaming() {
        let seq: Vec<u8> = (0..10_000).map(|i| b"ACGT"[i % 4]).collect();
        let qual: Vec<u8> = vec![b'I'; 10_000];
        let mut data = Vec::new();
        data.extend_from_slice(b"@large_read\n");
        data.extend_from_slice(&seq);
        data.push(b'\n');
        data.extend_from_slice(b"+\n");
        data.extend_from_slice(&qual);
        data.push(b'\n');

        let records = parse_all(&data).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].len(), 10_000);
    }

    #[test]
    fn error_display() {
        let e = FastqError::LengthMismatch {
            header: "r1".into(),
            seq_len: 4,
            qual_len: 2,
        };
        let s = e.to_string();
        assert!(s.contains("quality length (2)"));
        assert!(s.contains("sequence length (4)"));
    }
}
