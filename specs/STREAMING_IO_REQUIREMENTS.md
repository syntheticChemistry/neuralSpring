<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Streaming I/O Requirements

**Created**: March 4, 2026 (Session 122 debt audit)
**Status**: Specification — no parsers implemented yet
**Scope**: Future FASTQ, mzML, MS2 file parsers and any large-file I/O

---

## Principle

All file parsers in neuralSpring MUST be streaming (zero-copy where safe
Rust permits, chunked otherwise). Buffering an entire file into memory
violates sovereignty constraints (runs must work on resource-constrained
hardware) and does not scale to production genomics/proteomics datasets.

---

## Requirements

### R-01: No full-file buffering for scientific formats

Parsers for FASTQ, mzML, MS2, SAM/BAM, VCF, and any bioinformatics
format MUST process records incrementally via `Iterator` or `Stream`.
The working memory footprint MUST be O(record_size), not O(file_size).

### R-02: Safe Rust only

All I/O code respects `#![forbid(unsafe_code)]`. Memory-mapped I/O
(`mmap`) is acceptable only if a safe wrapper is available (e.g.,
`memmap2` with documented lifetime guarantees), or via a barracuda
primitive that encapsulates the `unsafe` boundary.

### R-03: `BufReader` streaming for line/record-oriented formats

```rust
use std::io::BufRead;

fn parse_fastq(reader: impl BufRead) -> impl Iterator<Item = FastqRecord> {
    // yield records one at a time, never buffer the entire file
}
```

Accept `impl BufRead`, not `&str` or `Vec<u8>`. This allows:
- File I/O via `BufReader<File>`
- Compressed streams via `flate2::read::GzDecoder`
- Network streams via `BufReader<TcpStream>`
- Test fixtures via `Cursor<Vec<u8>>`

### R-04: XML streaming for mzML

mzML files can be multi-GB. Use a pull parser (`quick-xml` or
`xml-rs`) in streaming mode. Never load the DOM tree.

### R-05: Validation round-trip

Every parser must have an integration test that:
1. Writes a known dataset to a temp file
2. Parses it back via the streaming API
3. Asserts bit-exact round-trip fidelity

### R-06: Compile-time baselines are exempt

`include_str!` for small JSON baselines (<100 KB) is acceptable since
these are embedded at compile time, not runtime I/O. This is the
existing pattern for validation binaries.

---

## Current State

| Component | Pattern | Status |
|-----------|---------|--------|
| `weight_loader.rs` | `std::fs::read` (full buffer) | Acceptable for safetensors (<10 MB). Evolve to chunked `BufReader` if files grow. |
| Baseline JSON | `include_str!` (compile-time) | Exempt (R-06) |
| Primal IPC | `BufReader` line-by-line | Compliant |
| FASTQ parser | Not implemented | Future — must follow R-01..R-05 |
| mzML parser | Not implemented | Future — must follow R-01..R-05 |
| MS2 parser | Not implemented | Future — must follow R-01..R-05 |

---

## Reference

- wateringHole ecoBin standard: Pure Rust, cross-compilation, no C deps
- `#![forbid(unsafe_code)]` crate policy
- barracuda `BufReader` IPC pattern (JSON-RPC 2.0 streaming)
