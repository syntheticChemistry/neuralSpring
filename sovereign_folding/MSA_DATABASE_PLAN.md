# MSA Database Plan — Sovereign Folding

**Last Updated**: February 12, 2026
**Purpose**: Plan acquisition, storage, and indexing of sequence databases
required for sovereign protein/RNA/DNA structure prediction.

---

## Overview

Structure prediction requires Multiple Sequence Alignments (MSAs) built from
large sequence databases. These databases are the "fuel" for the Evoformer —
without them, the model has no evolutionary signal to exploit.

**Total estimated storage**: ~2.5–3 TB compressed, ~6–8 TB uncompressed

---

## Protein Databases

### UniRef90 (Required — Tier 1)

| Field | Value |
|-------|-------|
| **Source** | UniProt (uniprot.org/uniref) |
| **Format** | FASTA |
| **Size** | ~100 GB compressed, ~300 GB uncompressed |
| **Update cycle** | Monthly |
| **URL** | `ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz` |
| **Purpose** | Primary MSA database — clustered at 90% identity |
| **Search tool** | JackHMMER / MMseqs2 |
| **Priority** | **P0 — download first** |

### BFD (Required — Tier 1)

| Field | Value |
|-------|-------|
| **Source** | Big Fantastic Database (Steinegger lab) |
| **Format** | HHsuite database format |
| **Size** | ~1.7 TB compressed |
| **URL** | `https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz` |
| **Purpose** | Metaclust metagenome database — deep MSAs for hard targets |
| **Search tool** | HHblits |
| **Priority** | **P1 — after UniRef90** |

### MGnify (Optional — Tier 2)

| Field | Value |
|-------|-------|
| **Source** | EBI Metagenomics (ebi.ac.uk/metagenomics) |
| **Format** | FASTA |
| **Size** | ~120 GB compressed |
| **URL** | `https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/` |
| **Purpose** | Additional metagenomic sequences for MSA depth |
| **Priority** | **P2 — after BFD** |

### PDB templates (Required — Tier 1)

| Field | Value |
|-------|-------|
| **Source** | RCSB PDB + PDB70/PDB100 |
| **Format** | mmCIF + HHsuite database |
| **Size** | ~200 GB processed |
| **URL** | `https://files.wwpdb.org/pub/pdb/data/structures/all/mmCIF/` |
| **Purpose** | Template structures for homology modeling |
| **Search tool** | HHsearch |
| **Priority** | **P0 — download first** |

---

## RNA Databases

### Rfam (Required — Tier 1)

| Field | Value |
|-------|-------|
| **Source** | Rfam (rfam.org) |
| **Format** | Stockholm alignments + covariance models |
| **Size** | ~5 GB compressed |
| **URL** | `ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/` |
| **Purpose** | RNA family alignments, secondary structure annotations |
| **Priority** | **P1** |

### RNAcentral (Required — Tier 1)

| Field | Value |
|-------|-------|
| **Source** | RNAcentral (rnacentral.org) |
| **Format** | FASTA |
| **Size** | ~50 GB compressed |
| **URL** | `ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/` |
| **Purpose** | Non-coding RNA sequences for RNA MSAs |
| **Priority** | **P1** |

### NT database (Optional — Tier 2)

| Field | Value |
|-------|-------|
| **Source** | NCBI Nucleotide |
| **Format** | FASTA |
| **Size** | ~200 GB compressed |
| **URL** | `ftp://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/nt.gz` |
| **Purpose** | Full nucleotide database for DNA/RNA search |
| **Priority** | **P3 — massive, defer until needed** |

---

## DNA Databases

### Genome assemblies (Future — Tier 3)

| Field | Value |
|-------|-------|
| **Source** | NCBI RefSeq / Ensembl |
| **Purpose** | Reference genomes for DNA structure context |
| **Priority** | **P3 — defer to Phase D** |

### Regulatory element databases (Future — Tier 3)

| Field | Value |
|-------|-------|
| **Source** | ENCODE, Roadmap Epigenomics |
| **Purpose** | Chromatin and regulatory element annotations |
| **Priority** | **P3 — defer to Phase D** |

---

## Storage Strategy

### Current Hardware

| Machine | Storage | Available | Role |
|---------|---------|-----------|------|
| Eastgate | NVMe 2TB + HDD | ~1 TB free | Primary download target |
| Strandgate | Dual EPYC, large storage | TBD | NUCLEUS storage node |
| NUCs | 256 GB–1 TB SSD | Limited | Indexed subset only |

### Phased Download

```
Phase 1 (immediate): UniRef90 (~100 GB) + PDB templates (~200 GB)
                      Total: ~300 GB. Enables small protein prediction.

Phase 2 (next):       Rfam (~5 GB) + RNAcentral (~50 GB)
                      Total: ~355 GB. Enables RNA structure prediction.

Phase 3 (later):      BFD (~1.7 TB). Requires Strandgate or external drive.
                      Total: ~2.1 TB. Enables deep MSAs for hard targets.

Phase 4 (future):     MGnify, NT, genome assemblies.
                      Total: ~3+ TB. Requires NUCLEUS Nest storage.
```

### Indexing

All databases need indexing for fast search:

| Database | Search Tool | Index Size | Index Time |
|----------|------------|-----------|-----------|
| UniRef90 | MMseqs2 | ~150 GB | ~2-4 hours |
| BFD | HHblits | Included in download | — |
| PDB | HHsearch | ~50 GB | ~1 hour |
| Rfam | Infernal (cmscan) | ~1 GB | <1 hour |
| RNAcentral | MMseqs2 | ~80 GB | ~2 hours |

### Sovereign Search Pipeline

The long-term goal is sovereign sequence search on consumer hardware:

1. **MMseqs2** (Pure C, no external deps) for protein/RNA search
2. **HHblits/HHsearch** (C++) for sensitive profile-profile search
3. Eventually: BarraCUDA GPU-accelerated k-mer search (new primitive)

---

## Download Script

See `download_databases.sh` for the automated download script.
The script supports resuming interrupted downloads and verifying checksums.

---

## Dependency on NUCLEUS

Phase 3+ databases (BFD at 1.7 TB) exceed single-machine storage on Eastgate.
This is where NUCLEUS Nest Atomic (Tower + NestGate) becomes critical:
content-addressed storage distributed across the basement mesh.

BFD shards can be distributed across NUCs, with NestGate handling
content-addressed retrieval. The search index lives on Eastgate (fast NVMe),
while raw sequence data is distributed.
