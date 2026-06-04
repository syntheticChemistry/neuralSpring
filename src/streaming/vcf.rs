// SPDX-License-Identifier: AGPL-3.0-or-later

//! Streaming VCF v4.x parser — O(`record_size`) memory, zero full-file buffering.
//!
//! Implements requirement R-01 from `specs/STREAMING_IO_REQUIREMENTS.md`.
//! Serves population genetics papers (024, 025) which consume variant call data.
//!
//! ## VCF format
//!
//! A VCF file consists of:
//! - Meta-information lines starting with `##` (key=value pairs)
//! - A header line starting with `#CHROM` defining columns
//! - Tab-delimited data lines with 8 fixed columns + optional genotype columns
//!
//! ## Usage
//!
//! ```
//! use std::io::Cursor;
//! use neural_spring::streaming::vcf::{VcfReader, VcfRecord};
//!
//! let data = b"##fileformat=VCFv4.3\n\
//!              #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
//!              chr1\t100\trs1\tA\tG\t30\tPASS\tDP=10\n";
//! let reader = VcfReader::new(Cursor::new(data)).unwrap();
//! let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
//! assert_eq!(records.len(), 1);
//! assert_eq!(records[0].chrom(), "chr1");
//! assert_eq!(records[0].pos(), 100);
//! ```

use std::io::BufRead;

/// Parsed metadata from VCF `##` header lines.
#[derive(Debug, Clone, Default)]
pub struct VcfHeader {
    /// All `##` meta-information lines (raw strings without the `##` prefix).
    pub meta_lines: Vec<String>,
    /// Sample names from the `#CHROM` header (columns 9+).
    pub samples: Vec<String>,
}

impl VcfHeader {
    /// VCF file format version from `##fileformat=...`, if present.
    #[must_use]
    pub fn file_format(&self) -> Option<&str> {
        self.meta_lines.iter().find_map(|line| {
            line.strip_prefix("fileformat=")
                .or_else(|| line.strip_prefix("fileFormat="))
        })
    }

    /// Number of samples (genotype columns).
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// A single VCF data record (one variant site).
///
/// Fields are owned [`String`]s because each data line is read into a reusable
/// buffer that is overwritten on the next iteration; owned values are required
/// for [`VcfRecord`] to outlive the reader. Parsing avoids an extra [`String`]
/// copy per `##` meta line in [`VcfReader::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct VcfRecord {
    chrom: String,
    pos: u64,
    id: String,
    ref_allele: String,
    alt_alleles: Vec<String>,
    qual: Option<f64>,
    filter: String,
    info: String,
    genotypes: Vec<String>,
}

impl VcfRecord {
    /// Chromosome name.
    #[must_use]
    pub fn chrom(&self) -> &str {
        &self.chrom
    }

    /// 1-based position.
    #[must_use]
    pub const fn pos(&self) -> u64 {
        self.pos
    }

    /// Variant identifier (e.g. `rs12345`), or `"."` if missing.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Reference allele.
    #[must_use]
    pub fn ref_allele(&self) -> &str {
        &self.ref_allele
    }

    /// Alternate alleles.
    #[must_use]
    pub fn alt_alleles(&self) -> &[String] {
        &self.alt_alleles
    }

    /// Quality score, or `None` if `.`.
    #[must_use]
    pub const fn qual(&self) -> Option<f64> {
        self.qual
    }

    /// Filter status (e.g. `"PASS"` or `"."`).
    #[must_use]
    pub fn filter(&self) -> &str {
        &self.filter
    }

    /// INFO field (unparsed key=value pairs).
    #[must_use]
    pub fn info(&self) -> &str {
        &self.info
    }

    /// Per-sample genotype strings (from FORMAT/sample columns).
    #[must_use]
    pub fn genotypes(&self) -> &[String] {
        &self.genotypes
    }

    /// Whether this is a SNP (single-nucleotide polymorphism): ref and all alts are 1 bp.
    #[must_use]
    pub fn is_snp(&self) -> bool {
        self.ref_allele.len() == 1
            && self
                .alt_alleles
                .iter()
                .all(|a| a.len() == 1 && a != "." && a != "*")
    }

    /// Whether the variant passed all filters.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        self.filter == "PASS" || self.filter == "."
    }

    /// Write this record as a tab-delimited VCF data line.
    ///
    /// # Errors
    ///
    /// Returns I/O errors from the underlying writer.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        write!(
            w,
            "{}\t{}\t{}\t{}\t{}\t",
            self.chrom,
            self.pos,
            self.id,
            self.ref_allele,
            self.alt_alleles.join(","),
        )?;
        match self.qual {
            None => write!(w, ".")?,
            Some(q) => write!(w, "{q}")?,
        }
        write!(w, "\t{}\t{}", self.filter, self.info)?;
        for gt in &self.genotypes {
            write!(w, "\t{gt}")?;
        }
        writeln!(w)?;
        Ok(())
    }
}

/// Parse error for malformed VCF data.
#[derive(Debug)]
pub enum VcfError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// Missing `#CHROM` header line.
    MissingHeader,
    /// Data line has fewer than 8 required fields.
    TooFewFields {
        /// Truncated preview of the malformed line.
        line_preview: String,
    },
    /// Position field is not a valid integer.
    InvalidPosition {
        /// Truncated preview of the malformed line.
        line_preview: String,
        /// Parser error message for the position field.
        detail: String,
    },
    /// Quality field is not a valid float or `.`.
    InvalidQuality {
        /// Truncated preview of the malformed line.
        line_preview: String,
        /// Parser error message for the quality field.
        detail: String,
    },
}

impl std::error::Error for VcfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for VcfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::MissingHeader => write!(f, "missing #CHROM header line"),
            Self::TooFewFields { line_preview } => {
                write!(f, "fewer than 8 fields: {line_preview:?}")
            }
            Self::InvalidPosition {
                line_preview,
                detail,
            } => write!(f, "invalid POS in {line_preview:?}: {detail}"),
            Self::InvalidQuality {
                line_preview,
                detail,
            } => write!(f, "invalid QUAL in {line_preview:?}: {detail}"),
        }
    }
}

impl From<std::io::Error> for VcfError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Streaming VCF reader — parses header once, then yields [`VcfRecord`] per iteration.
#[derive(Debug)]
pub struct VcfReader<R> {
    reader: R,
    line_buf: String,
    /// Parsed header (available after construction).
    header: VcfHeader,
}

impl<R: BufRead> VcfReader<R> {
    /// Parse the VCF header and prepare for streaming data records.
    ///
    /// # Errors
    ///
    /// Returns [`VcfError::MissingHeader`] if no `#CHROM` line is found, or
    /// [`VcfError::Io`] on read failure.
    pub fn new(mut reader: R) -> Result<Self, VcfError> {
        let mut line_buf = String::with_capacity(super::VCF_LINE_BUF_CAPACITY);
        let mut meta_lines = Vec::new();
        let mut samples = Vec::new();
        let mut found_header = false;

        loop {
            line_buf.clear();
            let n = reader.read_line(&mut line_buf)?;
            if n == 0 {
                break;
            }
            super::trim_end_newlines_in_place(&mut line_buf);

            if line_buf.starts_with("##") {
                let mut owned = std::mem::take(&mut line_buf);
                let meta = owned.split_off(2);
                meta_lines.push(meta);
            } else if line_buf.starts_with("#CHROM") {
                let cols: Vec<&str> = line_buf.as_str().split('\t').collect();
                if cols.len() > 9 {
                    samples = cols[9..].iter().map(|s| (*s).to_string()).collect();
                }
                found_header = true;
                break;
            } else {
                break;
            }
        }

        if !found_header {
            return Err(VcfError::MissingHeader);
        }

        Ok(Self {
            reader,
            line_buf,
            header: VcfHeader {
                meta_lines,
                samples,
            },
        })
    }

    /// Access the parsed VCF header.
    #[must_use]
    pub const fn header(&self) -> &VcfHeader {
        &self.header
    }
}

impl<R: BufRead> Iterator for VcfReader<R> {
    type Item = Result<VcfRecord, VcfError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => return None,
                Err(e) => return Some(Err(VcfError::Io(e))),
                Ok(_) => {}
            }

            super::trim_end_newlines_in_place(&mut self.line_buf);
            let line = self.line_buf.as_str();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            return Some(parse_record(line));
        }
    }
}

fn parse_record(line: &str) -> Result<VcfRecord, VcfError> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 8 {
        return Err(VcfError::TooFewFields {
            line_preview: truncate_preview(line),
        });
    }

    let chrom = fields[0].to_string();
    let pos: u64 =
        fields[1]
            .parse()
            .map_err(|e: std::num::ParseIntError| VcfError::InvalidPosition {
                line_preview: truncate_preview(line),
                detail: e.to_string(),
            })?;
    let id = fields[2].to_string();
    let ref_allele = fields[3].to_string();
    let alt_alleles: Vec<String> = fields[4].split(',').map(String::from).collect();
    let qual = if fields[5] == "." {
        None
    } else {
        Some(
            fields[5]
                .parse::<f64>()
                .map_err(|e| VcfError::InvalidQuality {
                    line_preview: truncate_preview(line),
                    detail: e.to_string(),
                })?,
        )
    };
    let filter = fields[6].to_string();
    let info = fields[7].to_string();

    let genotypes = if fields.len() > 9 {
        fields[9..].iter().map(|s| (*s).to_string()).collect()
    } else {
        Vec::new()
    };

    Ok(VcfRecord {
        chrom,
        pos,
        id,
        ref_allele,
        alt_alleles,
        qual,
        filter,
        info,
        genotypes,
    })
}

fn truncate_preview(s: &str) -> String {
    if s.len() > 80 {
        format!("{}...", &s[..80])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
mod tests {
    use super::*;
    use std::io::Cursor;

    const MINIMAL_VCF: &[u8] = b"##fileformat=VCFv4.3\n\
        ##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
        chr1\t100\trs1\tA\tG\t30\tPASS\tDP=10\n\
        chr1\t200\trs2\tC\tT,A\t.\t.\tDP=20\n\
        chr2\t300\t.\tACG\tA\t50.5\tPASS\tDP=30\n";

    #[test]
    fn parse_header() {
        let reader = VcfReader::new(Cursor::new(MINIMAL_VCF)).unwrap();
        let header = reader.header();
        assert_eq!(header.file_format(), Some("VCFv4.3"));
        assert_eq!(header.sample_count(), 0);
        assert_eq!(header.meta_lines.len(), 2);
    }

    #[test]
    fn parse_records() {
        let reader = VcfReader::new(Cursor::new(MINIMAL_VCF)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 3);

        assert_eq!(records[0].chrom(), "chr1");
        assert_eq!(records[0].pos(), 100);
        assert_eq!(records[0].id(), "rs1");
        assert_eq!(records[0].ref_allele(), "A");
        assert_eq!(records[0].alt_alleles(), &["G"]);
        assert_eq!(records[0].qual(), Some(30.0));
        assert_eq!(records[0].filter(), "PASS");
        assert_eq!(records[0].info(), "DP=10");
        assert!(records[0].is_snp());
        assert!(records[0].is_pass());
    }

    #[test]
    fn multi_alt() {
        let reader = VcfReader::new(Cursor::new(MINIMAL_VCF)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records[1].alt_alleles(), &["T", "A"]);
    }

    #[test]
    fn missing_qual() {
        let reader = VcfReader::new(Cursor::new(MINIMAL_VCF)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records[1].qual(), None);
    }

    #[test]
    fn indel_not_snp() {
        let reader = VcfReader::new(Cursor::new(MINIMAL_VCF)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert!(!records[2].is_snp());
        assert_eq!(records[2].ref_allele(), "ACG");
    }

    #[test]
    fn with_genotypes() {
        let data = b"##fileformat=VCFv4.3\n\
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2\n\
            chr1\t100\t.\tA\tG\t30\tPASS\tDP=10\tGT:DP\t0/1:15\t1/1:20\n";
        let reader = VcfReader::new(Cursor::new(data)).unwrap();
        let header = reader.header();
        assert_eq!(header.samples, vec!["Sample1", "Sample2"]);
        assert_eq!(header.sample_count(), 2);

        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records[0].genotypes(), &["0/1:15", "1/1:20"]);
    }

    #[test]
    fn missing_header_error() {
        let data = b"chr1\t100\t.\tA\tG\t30\tPASS\tDP=10\n";
        let err = VcfReader::new(Cursor::new(data)).unwrap_err();
        assert!(
            matches!(err, VcfError::MissingHeader),
            "expected MissingHeader, got {err}"
        );
    }

    #[test]
    fn too_few_fields() {
        let data =
            b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t100\n";
        let reader = VcfReader::new(Cursor::new(data)).unwrap();
        let err = reader.collect::<Result<Vec<_>, _>>().unwrap_err();
        assert!(
            matches!(err, VcfError::TooFewFields { .. }),
            "expected TooFewFields, got {err}"
        );
    }

    #[test]
    fn invalid_position() {
        let data = b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\tNOT_NUM\t.\tA\tG\t30\tPASS\tDP=10\n";
        let reader = VcfReader::new(Cursor::new(data)).unwrap();
        let err = reader.collect::<Result<Vec<_>, _>>().unwrap_err();
        assert!(
            matches!(err, VcfError::InvalidPosition { .. }),
            "expected InvalidPosition, got {err}"
        );
    }

    #[test]
    fn round_trip_fidelity() {
        let reader = VcfReader::new(Cursor::new(MINIMAL_VCF)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();

        let mut written = Vec::new();
        written.extend_from_slice(b"##fileformat=VCFv4.3\n");
        written.extend_from_slice(
            b"##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n",
        );
        written.extend_from_slice(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
        for rec in &records {
            rec.write_to(&mut written).unwrap();
        }

        let reader2 = VcfReader::new(Cursor::new(&written)).unwrap();
        let reparsed: Vec<VcfRecord> = reader2.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records, reparsed, "round-trip must be exact");
    }

    #[test]
    fn blank_lines_skipped() {
        let data = b"##fileformat=VCFv4.3\n\
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
            \n\
            chr1\t100\t.\tA\tG\t30\tPASS\tDP=10\n\
            \n";
        let reader = VcfReader::new(Cursor::new(data)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn windows_line_endings() {
        let data = b"##fileformat=VCFv4.3\r\n\
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\r\n\
            chr1\t100\t.\tA\tG\t30\tPASS\tDP=10\r\n";
        let reader = VcfReader::new(Cursor::new(data)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].chrom(), "chr1");
    }

    #[test]
    fn error_display() {
        let e = VcfError::TooFewFields {
            line_preview: "chr1\t100".into(),
        };
        let s = e.to_string();
        assert!(s.contains("fewer than 8 fields"));
    }

    #[test]
    fn float_quality() {
        let reader = VcfReader::new(Cursor::new(MINIMAL_VCF)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert!((records[2].qual().unwrap() - 50.5).abs() < 1e-10);
    }

    #[test]
    fn empty_after_header() {
        let data = b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";
        let reader = VcfReader::new(Cursor::new(data)).unwrap();
        let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert!(records.is_empty());
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

        #[test]
        fn parsed_position_always_positive(pos in 1_u64..1_000_000_000) {
            let data = format!(
                "##fileformat=VCFv4.3\n\
                 #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
                 chr1\t{pos}\t.\tA\tG\t30\tPASS\tDP=10\n"
            );
            let reader = VcfReader::new(Cursor::new(data.as_bytes())).unwrap();
            let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
            proptest::prop_assert_eq!(records.len(), 1);
            proptest::prop_assert_eq!(records[0].pos(), pos);
        }

        #[test]
        fn chrom_name_preserved(
            idx in 1_u32..22,
        ) {
            let chrom = format!("chr{idx}");
            let data = format!(
                "##fileformat=VCFv4.3\n\
                 #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n\
                 {chrom}\t100\t.\tA\tG\t30\tPASS\tDP=10\n"
            );
            let reader = VcfReader::new(Cursor::new(data.as_bytes())).unwrap();
            let records: Vec<VcfRecord> = reader.collect::<Result<Vec<_>, _>>().unwrap();
            proptest::prop_assert_eq!(records[0].chrom(), chrom.as_str());
        }
    }
}
