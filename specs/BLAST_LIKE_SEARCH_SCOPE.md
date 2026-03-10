<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# BLAST-Like Sequence Search Pipeline — Scoping Document

**Created**: March 10, 2026 (Industry Tool Gap Analysis, Tier 2)
**Status**: Scoping — not yet implemented
**Owner**: barraCuda (GPU kernel) + wetSpring (pipeline), with neuralSpring
as first consumer for AlphaFold MSA generation

---

## Motivation

BLAST (Basic Local Alignment Search Tool) is the most-used sequence
search tool in bioinformatics. It finds regions of local similarity between
a query sequence and a database. Our papers in phylogenetics (016–018),
population genetics (024–025), and protein folding (nF-01/02/03) all operate
in domains where BLAST-equivalent capability is assumed.

NCBI BLAST is public domain C code. We do not wrap it — we build a
sovereign Rust equivalent that exploits barraCuda's existing GPU
Smith-Waterman kernel for the alignment phase.

---

## Architecture: Seed-and-Extend

BLAST's performance comes from a two-phase strategy:

```text
Phase 1: SEED       Phase 2: EXTEND           Phase 3: SCORE
─────────────────   ─────────────────────────  ────────────────
k-mer index of DB   Smith-Waterman on hits     E-value, bit score
exact word match     banded + affine gaps       statistical model
O(n) scan            O(h · w · b) GPU           per-hit
```

### Phase 1 — Seed (CPU or GPU)

- Extract k-mers (k=3 for protein, k=11 for DNA) from query
- Look up each k-mer in a pre-built database index (hash table or FM-index)
- Return (query_pos, db_seq_id, db_pos) hit list
- **Existing barraCuda primitive**: `KmerHistogramGpu` (k-mer counting).
  Needs extension: k-mer *lookup* against a pre-built index

### Phase 2 — Extend (GPU)

- For each seed hit, extend alignment in both directions using
  banded Smith-Waterman with affine gap penalties
- **Existing barraCuda primitive**: `SmithWatermanGpu` (banded, f64, anti-diagonal wavefront)
  - Config: `SwConfig { gap_open, gap_extend, band_width }`
  - Already dispatches via anti-diagonal wavefront shader
  - Needs: batch dispatch (many query-db pairs per submission)

### Phase 3 — Score (CPU)

- Compute E-value from alignment score using Karlin-Altschul statistics
- Filter by E-value threshold
- Report: score, E-value, identity%, alignment coordinates, aligned sequences
- **Not in barraCuda yet**: statistical model for E-value computation

---

## What Exists Today

| Component | Location | Status |
|-----------|----------|--------|
| Smith-Waterman banded f64 GPU | `barracuda::ops::bio::SmithWatermanGpu` | Committed, tested |
| K-mer histogram GPU | `barracuda::ops::bio::KmerHistogramGpu` | Committed, tested |
| Pairwise Hamming distance GPU | `barracuda::ops::bio::PairwiseHammingGpu` | Committed, tested |
| Substitution matrix (BLOSUM62) | Not in barraCuda | Needed |
| Database index (FM-index / hash) | Not in barraCuda | Needed |
| E-value statistics | Not in barraCuda | Needed |
| Streaming FASTQ parser | `neural_spring::streaming::fastq` | Implemented (this session) |
| Streaming VCF parser | `neural_spring::streaming::vcf` | Implemented (this session) |

---

## What Needs to Be Built

### In barraCuda (GPU primitives)

1. **Substitution matrix support in SW kernel** — BLOSUM62/PAM250 scoring
   instead of identity/mismatch. The WGSL shader needs a uniform buffer
   for the 20×20 (protein) or 4×4 (DNA) scoring matrix.

2. **Batch Smith-Waterman dispatch** — The current `SmithWatermanGpu::run()`
   handles a single query-database pair. BLAST needs to dispatch hundreds
   of extensions per query. Batch variant: `run_batch(&[(query, db_seq)])`
   with a single command encoder submission.

3. **K-mer index lookup GPU** — Extend `KmerHistogramGpu` to support
   lookup against a pre-built index. Or: build a new `KmerSeedGpu`
   primitive that takes a query + index and returns hit coordinates.

### In wetSpring or neuralSpring (pipeline)

4. **Database indexing** — Build a k-mer hash index from a FASTA database
   file (streaming via `FastqReader` or a new FASTA streaming parser).
   Store as a sorted k-mer → (seq_id, offset) map, memory-mapped for
   large databases.

5. **Karlin-Altschul E-value model** — CPU-side statistical significance
   computation. Well-understood math (λ, K parameters from scoring matrix).

6. **BLAST-like pipeline orchestrator** — Ties seed → extend → score.
   Streams query sequences, batches extensions to GPU, filters and
   reports results. Follows the `ValidationHarness` pattern for
   deterministic validation against NCBI BLAST output.

---

## Validation Strategy

1. **Unit**: Known alignment pairs with pre-computed BLAST scores
2. **Integration**: Small FASTA database (100 sequences), query against NCBI
   BLAST output, assert identical top hits (within tie-breaking tolerance)
3. **Benchmark**: Time comparison vs NCBI BLAST+ (CPU) for 1000-query runs
   on UniRef50 subset

---

## Estimated Effort

| Phase | Component | Effort | Dependency |
|-------|-----------|--------|------------|
| 1 | Batch SW dispatch in barraCuda | 2–3 days | None |
| 1 | Substitution matrix in SW shader | 1 day | None |
| 2 | K-mer seed index (CPU) | 3–4 days | Streaming FASTA parser |
| 2 | K-mer seed lookup (GPU) | 2–3 days | Index format |
| 3 | E-value statistics | 1–2 days | None |
| 3 | Pipeline orchestrator | 2–3 days | All above |
| 4 | Validation vs NCBI BLAST | 1–2 days | Pipeline |
| **Total** | | **~12–18 days** | |

---

## Ownership

- **barraCuda**: Batch SW, substitution matrix shader, k-mer seed GPU
- **wetSpring** or **shared crate**: Database indexing, FASTA streaming, pipeline
- **neuralSpring**: First consumer (AlphaFold MSA → JackHMMER replacement),
  validation binaries
