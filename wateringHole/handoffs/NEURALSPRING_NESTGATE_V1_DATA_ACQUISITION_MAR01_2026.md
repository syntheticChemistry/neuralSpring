# neuralSpring → NestGate Handoff V1 — Sovereign Data Acquisition for Science Extensions

**Date**: March 1, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: NestGate team
**License**: AGPL-3.0-or-later
**Covers**: Session 99 — Data acquisition needs for baseCamp science extensions (nS-01 weight spectral, coralForge PDB validation, LTEE structural evolution, metagenome QS gene scan)

---

## Executive Summary

- neuralSpring has **3,490+ validated checks** across 25 papers, 5 WDM surrogates, coralForge (AlphaFold2/3), and 5 baseCamp sub-theses
- Everything validated so far uses **synthetic or seeded data** — the extension phase needs **real-world data** from NCBI, RCSB PDB, and HuggingFace
- neuralSpring's primal binary already **forwards `data.*` JSON-RPC calls to NestGate** via Unix socket (`src/bin/neuralspring_primal/main.rs` lines 155–176)
- **Gap**: NestGate's semantic router does NOT implement `data.*` methods — the `NCBILiveProvider` exists in `nestgate-core` but is not wired to JSON-RPC
- **Impact**: Wiring `data.*` handlers unlocks sovereign data acquisition for all 10 gen3 baseCamp papers

---

## Part 1: What neuralSpring Already Has

### Primal Binary Forwarding

neuralSpring's biomeOS primal binary handles `data.*` methods by forwarding to NestGate:

```rust
// src/bin/neuralspring_primal/main.rs (lines 167-174)
method if method.starts_with("data.") => {
    match forward_to_primal("nestgate", method, params).await {
        Ok(resp) => JsonRpcResponse::success(id, resp),
        Err(e) => JsonRpcResponse::error(
            id, -32000, format!("NestGate forward failed: {e}")
        ),
    }
}
```

### Socket Discovery

neuralSpring discovers NestGate via:
1. `$XDG_RUNTIME_DIR/biomeos/nestgate-{family_id}.sock`
2. `$XDG_RUNTIME_DIR/biomeos/nestgate.sock`
3. Fallback: scan socket dir for `nestgate*.sock`

### Methods neuralSpring Forwards

| Method | Purpose | Params Expected |
|--------|---------|-----------------|
| `data.ncbi_search` | Search NCBI databases | `{ query, database, max_results }` |
| `data.ncbi_fetch` | Fetch sequences by ID | `{ genome_id, format, database }` |
| `data.pdb_search` | Search RCSB PDB | `{ query, max_results }` |
| `data.pdb_fetch` | Fetch PDB coordinates | `{ pdb_id, format }` |

---

## Part 2: What NestGate Needs to Wire

### The Gap

NestGate's `NCBILiveProvider` (`nestgate-core/src/data_sources/providers/live_providers/ncbi_live_provider.rs`) already implements:

```rust
// Existing API in NCBILiveProvider
pub fn new(api_key: Option<String>, email: Option<String>) -> Result<Self>
async fn search_genomes(&self, query: &str) -> Result<Vec<GenomeResult>>
async fn get_genome_sequence(&self, genome_id: &str) -> Result<GenomeSequence>
```

And the internal E-utilities calls:
- `esearch(database, query, max_results)` → `Vec<String>` (IDs)
- `esummary(database, ids)` → summaries
- `efetch(database, id, format)` → sequence data

**What's missing**: The semantic router (`rpc/semantic_router/mod.rs`) only handles `storage.*`, `discovery.*`, `health.*`, `metadata.*`, `crypto.*`. No `data.*` handlers exist.

### Recommended Implementation

Add `data.*` handlers to the semantic router that delegate to `NCBILiveProvider`:

| JSON-RPC Method | NestGate Provider Call | Returns |
|-----------------|----------------------|---------|
| `data.ncbi_search` | `NCBILiveProvider::search_genomes(query)` | `Vec<GenomeResult>` (id, title, organism) |
| `data.ncbi_fetch` | `NCBILiveProvider::get_genome_sequence(id)` | `GenomeSequence` (id, sequence, metadata) |
| `data.pdb_search` | New: RCSB REST API search | `Vec<PdbResult>` (pdb_id, title, resolution) |
| `data.pdb_fetch` | New: RCSB REST API fetch | PDB coordinates (FASTA or mmCIF format) |

PDB integration can use RCSB's REST API (`https://data.rcsb.org/rest/v1/core/entry/{pdb_id}`) following the same pattern as `NCBILiveProvider`.

### Additional Providers Already in NestGate

NestGate already has provider stubs that neuralSpring will use:

| Provider | File | neuralSpring Use |
|----------|------|-----------------|
| `NCBILiveProvider` | `ncbi_live_provider.rs` | LTEE genomes, metagenomes, taxonomy |
| `HuggingFaceLiveProvider` | `huggingface_live_provider.rs` | Pretrained model weights (GPT-2, ViT) |
| `EnsemblLiveProvider` | `ensembl_live_provider.rs` | Protein sequences for coralForge |

---

## Part 3: Data Volume Projections

### Tier 1 — Immediate (<10 GB, single Eastgate)

| Dataset | Size | Use | NestGate Method |
|---------|------|-----|-----------------|
| PDB structures (20 small proteins) | ~2 MB | coralForge validation (Paper 10) | `data.pdb_fetch` |
| LTEE assembled genomes (10 strains) | ~500 MB | Paper 02 extension | `data.ncbi_fetch` |
| HuggingFace model weights (5 models) | ~1 GB | nS-01 Paper A | Direct download or `data.hf_fetch` |
| E. coli K-12 reference proteome | ~50 MB | coralForge LTEE baseline | `data.ncbi_fetch` |

### Tier 2 — Medium (~100 GB, benefits from Strandgate storage)

| Dataset | Size | Use | NestGate Need |
|---------|------|-----|--------------|
| UniRef90 (MSA database) | ~100 GB compressed | coralForge real structure prediction | Bulk FTP + content-addressed storage |
| PDB70 (profile database) | ~15 GB | coralForge template search | Bulk FTP |

### Tier 3 — Large (~1 TB, requires LAN + Westgate cold storage)

| Dataset | Size | Use | NestGate Need |
|---------|------|-----|--------------|
| SRA metagenomes (170 communities) | ~500 GB–1 TB | Paper 05 QS gene scan | SRA Toolkit integration |
| Full PDB mirror | ~100 GB | Comprehensive coralForge validation | Bulk download + BLAKE3 indexing |
| Earth Microbiome Project | ~2 TB | Paper 06 no-till extension | Bulk FTP |

---

## Part 4: Content-Addressed Storage Needs

For Tier 2/3 datasets, neuralSpring benefits from NestGate's content-addressed storage:

1. **BLAKE3 indexing**: Reference databases (UniRef90, PDB70) should be content-addressed so multiple springs can share them without duplication. If wetSpring and neuralSpring both need UniRef90, store once, reference by hash.

2. **Streaming download → processing**: For large datasets, NestGate should support streaming downloads where the client processes chunks as they arrive rather than waiting for the full download. This is critical for SRA FASTQ files (multi-GB each).

3. **Database versioning**: UniRef90 updates monthly. NestGate should track which version is stored and support atomic updates. Provenance metadata: download date, source URL, BLAKE3 hash, format version.

---

## Part 5: Lessons from neuralSpring for NestGate

### Tolerance-Based Validation Pattern

neuralSpring's `ValidationHarness` pattern (check result against expected within named tolerance) could inform NestGate's data integrity checks:

```
Download → BLAKE3 hash → Compare against known reference → Validate or re-download
```

### Provenance Tracking

neuralSpring tracks provenance for every computed result (Python baseline → Rust → GPU). NestGate should track data provenance: source database, access date, query parameters, result hash. This supports reproducible science.

### Cross-Spring Data Sharing

Multiple springs need the same reference databases:
- **wetSpring**: SILVA (16S), RefSeq, UniRef for QS gene analysis
- **neuralSpring**: UniRef90 (MSA for coralForge), PDB (structure validation)
- **hotSpring**: Lattice parameters, nuclear data tables

NestGate as the shared data layer avoids duplicate 100GB+ downloads across springs.

---

## Part 6: Priority Actions for NestGate

| Priority | Action | Impact |
|----------|--------|--------|
| **P1** | Wire `data.ncbi_search` and `data.ncbi_fetch` to `NCBILiveProvider` in semantic router | Unlocks Papers 02, 05, 06 |
| **P2** | Add RCSB PDB provider (`data.pdb_search`, `data.pdb_fetch`) | Unlocks Paper 10 (coralForge) |
| **P3** | Wire `HuggingFaceLiveProvider` for `data.hf_fetch` | Unlocks nS-01 Paper A (model weights) |
| **P4** | Content-addressed database storage (BLAKE3) | Enables Tier 2/3 shared databases |
| **P5** | SRA Toolkit integration for bulk FASTQ | Enables Paper 05 metagenome scan |
| **P6** | Streaming download → processing pipeline | Enables efficient large dataset acquisition |

---

## Handoff Lineage

| Version | Session | Focus |
|---------|---------|-------|
| **V1** | **S99** | **Data acquisition: NCBI, PDB, HuggingFace; content-addressed storage; cross-spring data sharing** |

---

*neuralSpring → NestGate V1 handoff — March 1, 2026. Session 99. 3,490+ validated checks need real-world data. NCBILiveProvider exists but not wired to JSON-RPC. Priority: wire data.* handlers, add PDB provider, enable content-addressed storage for shared reference databases.*
