# NestGate Provider Plan: PDB + UniRef for MSA Phase 1

**Date**: February 28, 2026
**Purpose**: Design NestGate data providers for sovereign structure prediction
**Depends**: NestGate v0.5.2+ (NCBI provider pattern), MSA_DATABASE_PLAN.md
**Target**: ~300 GB Phase 1 data (UniRef90 + PDB templates)

---

## Architecture

New providers follow the existing `DataCapability` trait pattern from
`nestgate-core/src/data_sources/data_capabilities.rs`:

```rust
pub trait DataCapability: Send + Sync {
    fn capability_type(&self) -> &str;
    fn can_handle(&self, request: &DataRequest) -> impl Future<Output = Result<bool>> + Send;
    fn execute_request(&self, request: &DataRequest) -> impl Future<Output = Result<DataResponse>> + Send;
    fn get_metadata(&self) -> HashMap<String, String>;
}
```

All three new providers follow the NCBI provider pattern: wrap
`UniversalHttpProvider` for API access, add domain-specific methods,
register with `UniversalDataAdapter`.

---

## Provider 1: RCSB PDB Provider

### Purpose

Download and serve PDB/mmCIF structure files for template-based modeling.
AlphaFold2/3 uses structural templates from the PDB to bias the Evoformer/
Pairformer pair representation.

### API

RCSB PDB provides a REST API and bulk download:

| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `https://data.rcsb.org/rest/v1/core/entry/{pdb_id}` | Entry metadata | 10 req/s |
| `https://files.rcsb.org/download/{pdb_id}.cif` | mmCIF structure | Unlimited |
| `https://search.rcsb.org/rcsbsearch/v2/query` | Structure search | 10 req/s |
| `https://files.wwpdb.org/pub/pdb/data/structures/all/mmCIF/` | Bulk FTP mirror | N/A |

### Struct

```rust
pub struct PdbLiveProvider {
    http_provider: UniversalHttpProvider,
    cache_dir: PathBuf,       // /data/nestgate/pdb/ on Westgate ZFS
    mmcif_index: DashMap<String, PdbEntry>,
}

pub struct PdbEntry {
    pub pdb_id: String,
    pub resolution: f64,
    pub method: String,        // X-ray, cryo-EM, NMR
    pub release_date: String,
    pub chains: Vec<ChainInfo>,
    pub local_path: Option<PathBuf>,
}
```

### Methods

| Method | Parameters | Returns | Notes |
|--------|-----------|---------|-------|
| `search_structures` | query, resolution_max, method | Vec<PdbEntry> | RCSB Search API |
| `fetch_mmcif` | pdb_id | mmCIF bytes | Download + cache |
| `fetch_template` | pdb_id, chain_id | Template atoms + sequence | Parsed for folding |
| `bulk_download` | resolution_max | Progress stream | Incremental FTP sync |
| `index_local` | — | index stats | Rebuild search index from local files |

### DataCapability Registration

```rust
impl DataCapability for PdbLiveProvider {
    fn capability_type(&self) -> &str { "structure_data" }
    // ...
}
```

### JSON-RPC Methods

| Method | Maps To |
|--------|---------|
| `data.pdb_search` | `search_structures` |
| `data.pdb_fetch` | `fetch_mmcif` |
| `data.pdb_template` | `fetch_template` |
| `data.pdb_bulk_sync` | `bulk_download` |

### Storage

| Tier | Location | Size | Notes |
|------|----------|------|-------|
| Hot cache | Eastgate NVMe | ~20 GB | Frequently-used structures |
| Cold store | Westgate ZFS `/data/nestgate/pdb/` | ~200 GB | Full PDB mirror |
| Index | Eastgate SSD | ~500 MB | DashMap-backed search index |

### Bulk Download Strategy

1. Initial: `rsync` from `rsync.rcsb.org::ftp_data/structures/all/mmCIF/` (~200 GB)
2. Incremental: Weekly delta sync (RCSB publishes weekly updates on Wednesdays)
3. Validation: File count + checksum against RCSB holdings list
4. Storage: Compressed mmCIF (`.cif.gz`), decompress on access

---

## Provider 2: UniProt/UniRef Provider

### Purpose

Serve UniRef90 clustered sequences for MSA construction. JackHMMER/MMseqs2
searches against UniRef90 to build the multiple sequence alignments that
drive the Evoformer/Pairformer.

### API

UniProt provides REST API and bulk FTP:

| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `https://rest.uniprot.org/uniref/search` | Search clusters | 1 req/s |
| `https://rest.uniprot.org/uniref/{id}.fasta` | Single cluster FASTA | 1 req/s |
| `ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/` | Bulk download | N/A |
| `https://rest.uniprot.org/uniprotkb/search` | Protein search | 1 req/s |

### Struct

```rust
pub struct UniProtLiveProvider {
    http_provider: UniversalHttpProvider,
    cache_dir: PathBuf,         // /data/nestgate/uniprot/
    uniref90_db: Option<PathBuf>,  // Path to local UniRef90 database
    mmseqs_binary: Option<PathBuf>, // Path to MMseqs2 binary
}
```

### Methods

| Method | Parameters | Returns | Notes |
|--------|-----------|---------|-------|
| `search_uniref` | query, identity_threshold | Vec<UniRefCluster> | REST API search |
| `fetch_cluster_fasta` | cluster_id | FASTA string | Single cluster |
| `build_msa` | query_sequence, database, e_value | MSA result | JackHMMER/MMseqs2 |
| `bulk_download_uniref90` | — | Progress stream | FTP download |
| `index_mmseqs` | — | index stats | Build MMseqs2 index |

### DataCapability Registration

```rust
impl DataCapability for UniProtLiveProvider {
    fn capability_type(&self) -> &str { "sequence_data" }
    // ...
}
```

### JSON-RPC Methods

| Method | Maps To |
|--------|---------|
| `data.uniprot_search` | `search_uniref` |
| `data.uniprot_fetch` | `fetch_cluster_fasta` |
| `data.msa_build` | `build_msa` |
| `data.uniref90_sync` | `bulk_download_uniref90` |

### MSA Search Integration

The critical path is `build_msa`, which shells out to JackHMMER or MMseqs2:

```
query.fasta → JackHMMER/MMseqs2 → UniRef90 DB → aligned.a3m
```

MMseqs2 is preferred for speed (~100x faster than JackHMMER):

| Tool | Speed (per query) | RAM | Accuracy |
|------|-------------------|-----|----------|
| JackHMMER | ~hours | 8 GB | Gold standard |
| MMseqs2 | ~minutes | 32 GB | ~98% sensitivity vs JackHMMER |

**Recommendation**: MMseqs2 on Strandgate (256 GB RAM, dual EPYC) for production.
JackHMMER on Eastgate for single-query validation.

### Storage

| Tier | Location | Size | Notes |
|------|----------|------|-------|
| UniRef90 DB | Westgate ZFS `/data/nestgate/uniref90/` | ~100 GB compressed | MMseqs2 indexed |
| MSA cache | Eastgate NVMe | ~10 GB | Recently-built alignments |
| Index | Westgate ZFS | ~30 GB | MMseqs2 precomputed index |

---

## Provider 3: CCD Provider (Chemical Component Dictionary)

### Purpose

Serve ligand and small-molecule definitions for AlphaFold3 multi-molecule
tokenization. The CCD maps 3-letter codes to atom names, bond topology,
and ideal coordinates.

### API

| Endpoint | Purpose |
|----------|---------|
| `https://files.wwpdb.org/pub/pdb/data/monomers/components.cif` | Full CCD (~50 MB) |
| `https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}` | Single component |

### Struct

```rust
pub struct CcdProvider {
    http_provider: UniversalHttpProvider,
    components: HashMap<String, ChemicalComponent>,
    local_path: Option<PathBuf>,
}

pub struct ChemicalComponent {
    pub id: String,           // e.g. "ATP", "HEM"
    pub name: String,
    pub formula: String,
    pub atoms: Vec<CcdAtom>,
    pub bonds: Vec<CcdBond>,
    pub ideal_coordinates: Vec<[f64; 3]>,
}
```

### Methods

| Method | Parameters | Returns |
|--------|-----------|---------|
| `fetch_component` | comp_id | ChemicalComponent |
| `fetch_full_dictionary` | — | component count |
| `tokenize_ligand` | comp_id | atom names + coordinates |

### Storage

Entire CCD is ~50 MB. Cache on Eastgate NVMe. No ZFS needed.

---

## Registration with UniversalDataAdapter

```rust
let adapter = UniversalDataAdapterBuilder::new()
    .with_provider(Arc::new(NCBILiveProvider::new(/* ... */)?))      // existing
    .with_provider(Arc::new(EnsemblLiveProvider::new(/* ... */)?))    // existing
    .with_provider(Arc::new(HuggingFaceLiveProvider::new(/* ... */)?)) // existing
    .with_provider(Arc::new(PdbLiveProvider::new(cache_dir, zfs_path)?))        // NEW
    .with_provider(Arc::new(UniProtLiveProvider::new(cache_dir, mmseqs_path)?)) // NEW
    .with_provider(Arc::new(CcdProvider::new(cache_dir)?))                      // NEW
    .build();
```

---

## Phased Download Plan

### Phase 1: Protein-Only (~300 GB)

| Step | Data | Size | Target | Time (100 Mbps) |
|------|------|------|--------|-----------------|
| 1 | CCD dictionary | 50 MB | Eastgate NVMe | seconds |
| 2 | PDB mmCIF (all) | 200 GB | Westgate ZFS | ~5 hours |
| 3 | UniRef90 | 100 GB | Westgate ZFS | ~2.5 hours |
| 4 | MMseqs2 index build | — | Strandgate | ~4 hours |

**Total**: ~300 GB, ~12 hours end-to-end

### Phase 2: + RNA/DNA (~355 GB)

| Step | Data | Size | Target |
|------|------|------|--------|
| 5 | Rfam | 5 GB | Westgate ZFS |
| 6 | RNAcentral | 50 GB | Westgate ZFS |

### Phase 3: Full Databases (~2.5 TB)

| Step | Data | Size | Target |
|------|------|------|--------|
| 7 | BFD | 1.7 TB | Westgate ZFS |
| 8 | MGnify | 120 GB | Westgate ZFS |

---

## Validation Plan

| Test | Checks | Description |
|------|--------|-------------|
| PDB fetch + parse | 5 | Fetch known PDB IDs, validate mmCIF parsing |
| PDB template extraction | 3 | Extract chain atoms + sequence from mmCIF |
| UniRef90 search | 3 | Query REST API, validate cluster results |
| MSA build (small) | 2 | Build MSA for 50-residue test protein |
| CCD fetch + tokenize | 3 | Fetch ATP, HEM, parse atoms + bonds |
| Round-trip cache | 3 | Fetch → cache → retrieve → compare |
| JSON-RPC interface | 4 | All data.pdb_* and data.uniprot_* methods |

**Target**: 23/23 provider validation checks

---

## Hardware Assignment

| Gate | Role | Relevant Hardware |
|------|------|-------------------|
| **Eastgate** | Development + hot cache | RTX 4070, 32 GB RAM, 2 TB NVMe |
| **Westgate** | Cold storage + bulk data | 76 TB ZFS NAS (10 GbE pending) |
| **Strandgate** | MSA search (MMseqs2) | Dual EPYC, 256 GB RAM, RTX 3090 |
| **Northgate** | Large inference | RTX 5090, 192 GB RAM |

---

*All providers follow NestGate's existing DataCapability pattern. The NCBI
provider serves as the template. CCD can be implemented immediately (50 MB,
no bulk infra). PDB and UniRef90 providers need Westgate ZFS and 10G LAN
cables for production use, but can be validated with single-file fetches
on Eastgate.*
