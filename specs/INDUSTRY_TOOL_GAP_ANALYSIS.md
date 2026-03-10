<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Industry Tool Gap Analysis

**Created**: March 10, 2026 (S137 Industry Standard Expansion)
**Status**: Living document — updated as capabilities are implemented
**Scope**: Maps external scientific tools referenced by neuralSpring papers
to ecoPrimals sovereign implementations and identifies rebuild priorities

---

## Executive Summary

neuralSpring's 26+ papers implement algorithms from scratch in Python/Rust
rather than wrapping external tools. However, the *domains* these papers
operate in are served by major industry tools — both proprietary
(SnapGene, Chromeleon, MATLAB) and open-source (BLAST, HMMER, AlphaFold).

This analysis profiles what the papers used, what we lack access to,
and what we can rebuild as sovereign Rust + GPU implementations.

**Key finding**: The highest-impact gaps are in *sequence search* and *MSA
generation* — both prerequisites for the AlphaFold pipeline, and both
buildable on existing barraCuda GPU primitives (Smith-Waterman, HMM, k-mer).

---

## Domain Map: Papers → Industry Tools

| Domain | Papers | Key External Tools | Status in ecoPrimals |
|--------|--------|-------------------|---------------------|
| Evolutionary Computation | 011–014 (Dolson) | Avida, Empirical, MODES-toolbox | Data source only |
| Phylogenetics/Alignment | 016–018 (Liu) | BLAST, HMMER, MAFFT, RAxML | Partial (SW GPU exists) |
| Population Genetics | 024–025 | PLINK, VCFtools, Arlequin | VCF parser implemented |
| Protein Folding | nF-01/02/03 | AlphaFold2/3, OpenFold, MMseqs2, JackHMMER | Evoformer/Pairformer done; MSA generation missing |
| Biofilm/QS/Regulatory | 019–021 | MATLAB ODE, COPASI | RK4/RK45/Hill in Rust+GPU |
| LC-MS/Analytical Chem | wetSpring Track 2 | Chromeleon, MassHunter, MZmine | wetSpring owns |
| Metagenomics | wetSpring Track 1 | DADA2, QIIME2, Kraken2 | DADA2 GPU op in barraCuda |
| Molecular Biology | (not in papers) | SnapGene, Benchling, Geneious | Not applicable |

---

## Tool Classification

### Proprietary — No Access, Must Rebuild or Skip

| Tool | Vendor | Domain | Relevance to Papers |
|------|--------|--------|-------------------|
| SnapGene | Dotmatics | Plasmid design, cloning | None (wet-lab tool) |
| Chromeleon | Thermo Fisher | HPLC/GC chromatography | Indirect (wetSpring Track 2) |
| MassHunter | Agilent | LC-MS data analysis | Indirect (wetSpring Track 2) |
| Xcalibur | Thermo Fisher | MS data acquisition | Indirect (instrument-bundled) |
| MATLAB | MathWorks | ODE, signal processing | Papers 019–021 (replaced by RK4/RK45) |
| Geneious Prime | Dotmatics | Sequence analysis | None (wet-lab tool) |
| Benchling | Benchling Inc | Lab notebook, LIMS | None (wet-lab SaaS) |
| CLC Genomics | QIAGEN | Variant analysis | None |

### Open-Source — Available for Validation and Rebuild

| Tool | License | Domain | In ecoPrimals? |
|------|---------|--------|----------------|
| BLAST | Public domain | Sequence search | Scoped (`BLAST_LIKE_SEARCH_SCOPE.md`) |
| HMMER | BSD-3 | Profile HMM search | Scoped (`MSA_PIPELINE_SCOPE.md`) |
| MAFFT | BSD | Multiple alignment | Not yet |
| MMseqs2 | GPL-3 | Fast search/MSA | Scoped (`MSA_PIPELINE_SCOPE.md`) |
| JackHMMER | BSD-3 | Iterative HMM search | Scoped (`MSA_PIPELINE_SCOPE.md`) |
| RAxML-NG | AGPL-3 | ML phylogenetics | Not yet |
| DADA2 | Artistic-2.0 | Amplicon denoising | **YES** — `barracuda::ops::bio::dada2` |
| QIIME2 | BSD-3 | Microbiome pipeline | **YES** — wetSpring validation |
| Kraken2 | MIT | Metagenomic classification | Not yet |
| MZmine | GPL-2 | LC-MS feature detection | wetSpring owns |
| asari | MIT | LC-MS EIC extraction | **YES** — wetSpring validation |
| AlphaFold2/3 | Apache-2.0 | Protein folding | **YES** — coralForge |
| OpenFold | Apache-2.0 | Protein folding | **YES** — validation target |
| COPASI | Artistic-2.0 | Biochemical ODE | Replaced by RK4/RK45 |
| LAMMPS | GPL-2 | Molecular dynamics | **Benchmark target** — hotSpring |
| Avida | GPL-2 | Digital evolution | Data source only |

### Already Rebuilt in Sovereign Rust + GPU

| Capability | barraCuda Op | neuralSpring Module |
|-----------|-------------|-------------------|
| HMM forward/backward/Viterbi (f64 GPU) | `HmmBatchForwardF64`, `hmm_backward`, `hmm_viterbi` | `hmm`, metalForge |
| Smith-Waterman banded alignment | `SmithWatermanGpu` | `sate_alignment` |
| DADA2 E-step (GPU) | `Dada2EStepGpu` | — (wetSpring) |
| Wright-Fisher population sim | `WrightFisherGpu` | validation binaries |
| Hill kinetics | `stats::hill` | `regulatory_network`, `signal_integration` |
| RK4/RK45 ODE integration (GPU) | `BatchedOdeRK45F64` | metalForge `rk45_adaptive.wgsl` |
| Pairwise distances (Hamming, Jaccard, L2) | `PairwiseHammingGpu`, etc. | validation binaries |
| NK fitness landscapes | `BatchFitnessGpu` | validation binaries |
| AlphaFold Evoformer attention | — | `coral_forge::evoformer` |
| AlphaFold3 Pairformer + diffusion | — | `coral_forge::pairformer`, `coral_forge::diffusion` |
| Spectral theory (eigh, Anderson) | `eigh_f64` | `spectral_commutativity`, `anderson_localization` |
| K-mer histogram (GPU) | `KmerHistogramGpu` | — (wetSpring) |
| UniFrac propagation (GPU) | `UniFracPropagateGpu` | — (wetSpring) |
| Taxonomy FC (GPU) | `TaxonomyFcGpu` | — (wetSpring) |

---

## Streaming I/O — Implemented This Session

Per `specs/STREAMING_IO_REQUIREMENTS.md`, the following parsers were
implemented in `src/streaming/`:

| Parser | Module | Tests | Status |
|--------|--------|-------|--------|
| FASTQ | `streaming::fastq::FastqReader` | 15 (round-trip, error cases, large records) | **Done** |
| VCF v4.x | `streaming::vcf::VcfReader` | 15 (round-trip, genotypes, error cases) | **Done** |
| FASTA | `streaming::fasta::FastaReader` | 16 (round-trip, DNA encoding, multiline) | **Done** |
| mzML | Not yet | — | wetSpring owns |
| SAM/BAM | Not yet | — | Future |
| MS2 | Not yet | — | Future |

All parsers accept `impl BufRead`, yield records via `Iterator`, have
O(`record_size`) memory footprint, and include round-trip fidelity tests.

---

## Rebuild Priorities

### Tier 1 — Directly Serves neuralSpring Papers (Implemented/In Progress)

| Item | Status | Reference |
|------|--------|-----------|
| Streaming FASTQ parser | **Done** | `src/streaming/fastq.rs` |
| Streaming VCF parser | **Done** | `src/streaming/vcf.rs` |
| Streaming mzML parser | Deferred to wetSpring | `STREAMING_IO_REQUIREMENTS.md` R-04 |
| Streaming SAM/BAM parser | Future | `STREAMING_IO_REQUIREMENTS.md` R-01 |

### Tier 2 — Extends Toward Industry Parity (Scoped)

| Item | Status | Reference |
|------|--------|-----------|
| BLAST-like sequence search | **CPU pipeline implemented** | `src/search/`, `specs/BLAST_LIKE_SEARCH_SCOPE.md` |
| MSA generation (JackHMMER equiv) | **Scoped** | `specs/MSA_PIPELINE_SCOPE.md` |
| MMseqs2-equivalent fast clustering | Scoped within MSA pipeline | `specs/MSA_PIPELINE_SCOPE.md` Phase C |

### Tier 3 — New Domains (Not in Current Papers)

| Item | Owner | Status |
|------|-------|--------|
| Chromatography peak detection | wetSpring | Not scoped |
| Metagenomic classification (Kraken2) | wetSpring | Not scoped |
| Molecular cloning toolkit | New primal | Not applicable |
| ML phylogenetics (RAxML) | Future | Not scoped |

### Not Worth Rebuilding

| Tool | Reason |
|------|--------|
| SnapGene / Geneious / Benchling | Wet-lab molecular biology — none of our papers use molecular cloning |
| MATLAB | RK4/RK45/Hill/ODE already in Rust+GPU — MATLAB's value is interactive exploration |
| CLC Genomics Workbench | Proprietary QIAGEN tool — VCF parser + variant filtering covers the computational core |

---

## Ownership Matrix

| Capability | Owner | Rationale |
|-----------|-------|-----------|
| Streaming parsers (FASTQ, VCF) | **neuralSpring** (implemented) | Serves papers 016–018, 024–025 directly |
| Streaming parsers (mzML, MS2) | **wetSpring** | LC-MS / analytical chemistry domain |
| BLAST-like search | **barraCuda** (GPU) + **wetSpring** (pipeline) | SW GPU exists; seed-extend is pipeline |
| MSA generation | **neuralSpring** (pipeline) + **barraCuda** (primitives) | AlphaFold pipeline lives in neuralSpring |
| AlphaFold end-to-end | **neuralSpring** | coralForge is here |
| LC-MS pipeline | **wetSpring** | Track 2 |
| Metagenomics pipeline | **wetSpring** | Track 1 |
| Population genetics tools | **wetSpring** + **neuralSpring** | Papers 024–025 in neuralSpring |
| Molecular dynamics | **hotSpring** | Yukawa, LAMMPS benchmarks |

---

## Related Documents

- `specs/STREAMING_IO_REQUIREMENTS.md` — streaming I/O spec (R-01 through R-06)
- `specs/BLAST_LIKE_SEARCH_SCOPE.md` — BLAST-like pipeline scoping (Tier 2)
- `specs/MSA_PIPELINE_SCOPE.md` — MSA generation pipeline scoping (Tier 2)
- `specs/coral_forge_assessment/MSA_DATABASE_PLAN.md` — database acquisition plan
- `specs/BENCHMARK_ANALYSIS.md` — Python vs barraCuda benchmark gaps
