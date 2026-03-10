<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# MSA Generation Pipeline — Scoping Document

**Created**: March 10, 2026 (Industry Tool Gap Analysis, Tier 2)
**Status**: Scoping — not yet implemented
**Owner**: neuralSpring (AlphaFold pipeline)
**Depends on**: `specs/coral_forge_assessment/MSA_DATABASE_PLAN.md`

---

## Motivation

AlphaFold2/3 structure prediction requires Multiple Sequence Alignments (MSAs)
as input to the Evoformer/Pairformer. The standard pipeline uses:

- **JackHMMER** (iterative HMM search against UniRef90)
- **HHblits** (profile-profile search against BFD)
- **MMseqs2** (fast prefilter for large databases)

These are C/C++ tools. Our sovereign pipeline replaces them with pure Rust
GPU-accelerated equivalents built on barraCuda primitives.

---

## AlphaFold MSA Pipeline Overview

```text
Query sequence (FASTA)
    │
    ├──▶ JackHMMER search (UniRef90)  ──▶  MSA₁ (deep, ~10k seqs)
    │
    ├──▶ HHblits search (BFD)          ──▶  MSA₂ (deep, ~100k seqs)
    │
    └──▶ Template search (PDB70)        ──▶  Templates (structural)
         │
         ▼
    MSA pairing + deduplication
         │
         ▼
    Evoformer / Pairformer input
```

---

## What Exists Today

### In neuralSpring (coralForge)

| Component | Location | Status |
|-----------|----------|--------|
| Outer product mean (MSA → pair) | `coral_forge::msa::outer_product_mean` | Implemented |
| MSA row attention | `coral_forge::msa::msa_row_attention_with_pair_bias` | Implemented |
| MSA column attention | `coral_forge::msa::msa_column_attention` | Implemented |
| Evoformer block (AF2) | `coral_forge::evoformer` | Implemented |
| Pairformer block (AF3) | `coral_forge::pairformer` | Implemented |

These modules **consume** MSAs. The gap is **generating** them from raw sequence.

### In barraCuda

| Component | Location | Relevance |
|-----------|----------|-----------|
| HMM forward/backward/Viterbi (f64) | `ops::bio::hmm` | Profile HMM scoring |
| Smith-Waterman (banded, f64) | `ops::bio::smith_waterman` | Local alignment |
| K-mer histogram | `ops::bio::kmer_histogram` | k-mer prefilter |
| Pairwise Hamming/Jaccard/L2 | `ops::bio::pairwise_*` | Distance-based filtering |
| ANI batch (f64) | `ops::bio::ani` | Sequence identity |

### In neuralSpring streaming I/O

| Component | Location | Status |
|-----------|----------|--------|
| Streaming FASTQ parser | `streaming::fastq` | Implemented (this session) |
| Streaming VCF parser | `streaming::vcf` | Implemented (this session) |
| Streaming FASTA parser | Not implemented | Needed (Tier 1.5) |

---

## What Needs to Be Built

### Phase A: JackHMMER Equivalent (Iterative HMM Search)

JackHMMER iteratively builds a profile HMM from initial hits, then re-searches.
This is the most critical MSA source for AlphaFold.

**Pipeline**:
```text
1. Build initial profile from query (single sequence → PSSM)
2. Search database with profile HMM (forward algorithm)
3. Collect hits above E-value threshold
4. Re-estimate profile from hits (multiple alignment → new PSSM)
5. Repeat steps 2–4 until convergence (typically 3–5 iterations)
6. Output: final MSA
```

**barraCuda primitives needed**:
- `HmmBatchForwardF64` already handles forward scoring
- Missing: **profile HMM construction from alignment** (PSSM estimation)
- Missing: **E-value computation** from HMM scores (same as BLAST scoping)
- Missing: **iterative pipeline orchestrator** (convergence detection)

**Estimated effort**: 5–8 days (most of the math is in barraCuda already)

### Phase B: HHblits Equivalent (Profile-Profile Search)

HHblits aligns profile HMMs against profile HMMs (not sequence against profile).
This is harder than JackHMMER but provides deeper MSAs.

**barraCuda primitives needed**:
- Missing: **profile-profile alignment kernel** (HMM vs HMM scoring matrix)
- Missing: **HHsuite database format reader** (binary format)
- The MAC (maximum accuracy) alignment algorithm is more complex than SW

**Estimated effort**: 10–15 days (new kernel development)
**Recommendation**: Defer to Phase B. JackHMMER + MMseqs2 covers most targets.

### Phase C: MMseqs2 Equivalent (Fast Prefilter)

MMseqs2 uses a k-mer prefilter to quickly eliminate non-homologous sequences
before running Smith-Waterman. The prefilter is the key to speed.

**Overlaps with BLAST scoping** (`specs/BLAST_LIKE_SEARCH_SCOPE.md`):
- K-mer seeding (Phase 1) is identical
- SW extension (Phase 2) uses the same `SmithWatermanGpu`
- The clustering step (for database creation) is new

**barraCuda primitives needed** (beyond BLAST scope):
- Missing: **ungapped prefilter** (fast diagonal scoring without full DP)
- Missing: **sequence clustering** (CD-HIT-like iterative centroid selection)

**Estimated effort**: 8–12 days (after BLAST pipeline exists, incremental)

### Phase D: Template Search

Uses HHsearch to find structural templates in PDB70. Lower priority since
AlphaFold3 reduces dependence on templates (diffusion-based).

**Estimated effort**: 3–5 days (reuses profile-profile from Phase B)

---

## Streaming FASTA Parser (Required Prerequisite)

All MSA databases are in FASTA format. We need a streaming parser similar
to the FASTQ parser but for FASTA (header line starting with `>`, multiline
sequence until next `>`).

**Spec**: Same requirements as STREAMING_IO_REQUIREMENTS.md R-01/R-03.
**Estimated effort**: 0.5 days (simpler format than FASTQ).

---

## Database Handling

Per `MSA_DATABASE_PLAN.md`, the databases total ~2.5–3 TB:

| Database | Size (compressed) | Search Tool | Priority |
|----------|-----------------:|-------------|----------|
| UniRef90 | ~100 GB | JackHMMER equiv | P0 |
| PDB templates | ~200 GB | HHsearch equiv | P0 |
| BFD | ~1.7 TB | HHblits equiv | P1 |
| RNAcentral | ~50 GB | JackHMMER equiv | P1 |
| Rfam | ~5 GB | Infernal equiv | P1 |

**Indexing** is a one-time cost per database update:
- k-mer hash index (for MMseqs2-like prefilter)
- FM-index (for exact substring queries, optional)
- Profile database (for HHblits, pre-computed from clustering)

---

## Validation Strategy

1. **Unit**: Small protein (< 100 residues) against UniRef90 subset (1000 seqs).
   Compare MSA depth and column coverage against JackHMMER/HHblits output.

2. **Integration**: AlphaFold end-to-end — generate MSA sovereign, feed to
   Evoformer, compare predicted structure RMSD against MSA from standard tools.

3. **Benchmark**: Search time per query against UniRef90 (full). Compare
   against JackHMMER wall-clock time. Target: GPU pipeline within 2× of
   JackHMMER on CPU (acceptable since GPU can batch many queries).

---

## Phased Implementation

| Phase | Scope | Effort | Enables |
|-------|-------|--------|---------|
| 0 | Streaming FASTA parser | 0.5 days | All database reading |
| A | JackHMMER equivalent | 5–8 days | AlphaFold MSA₁ (UniRef90) |
| B | HHblits equivalent | 10–15 days | AlphaFold MSA₂ (BFD) — defer |
| C | MMseqs2 prefilter | 8–12 days | Fast search at scale |
| D | Template search | 3–5 days | AlphaFold templates — defer |
| **Total** | | **~27–40 days** | Full sovereign MSA |

**Recommended start**: Phase 0 (FASTA parser) → Phase A (JackHMMER).
These two phases give us sovereign MSA generation for most AlphaFold targets
using UniRef90 alone. Phases B–D are incremental improvements.

---

## Ownership

- **neuralSpring**: Pipeline orchestrator, coralForge integration, validation
- **barraCuda**: GPU primitives (HMM, SW, k-mer), profile construction kernel
- **wetSpring**: Database acquisition scripts, format documentation
- **toadStool**: Multi-device dispatch for large database searches
