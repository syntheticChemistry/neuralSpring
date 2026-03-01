# neuralSpring baseCamp: Extension Plan — From Validated Science to Discovery

**Date**: March 1, 2026 (Session 98+)
**Author**: Kevin Mok (BS Microbiology, MSU 2018; MS Data Science, MSU 2025)
**Status**: PLAN — data/compute assessment, primal integration roadmap, extension priorities

---

## Where We Stand

neuralSpring has validated 3,490+ checks across 25 papers, 5 WDM surrogates,
coralForge (nF-01/02/03), 5 baseCamp sub-theses, and 3 publication experiments.
The full pipeline is proven: Python → Rust CPU → BarraCUDA CPU → GPU Tensor →
Pure GPU → metalForge cross-substrate. 199/199 validators pass. 83.6× faster
than Python. Zero debt.

**What's missing**: everything has been validated against published science,
but almost none of it has been applied to new data or new systems. The
pipeline is proven portable. The extension phase applies it to discovery.

---

## Extension Axes

### Axis 1: Real Data Into Validated Pipelines

| Extension | Sub-thesis | Data Source | Data Size | Acquisition | Compute |
|-----------|-----------|-------------|-----------|-------------|---------|
| **Weight spectral analysis on pretrained models** | nS-01 (Paper A) | torchvision / HuggingFace model weights | ~50MB–5GB per model (weights only) | Free download, convert to flat f32/f64 tensors | `eigh_f64`: seconds per layer on Eastgate (i9-12900). GPU `BatchIprGpu`: milliseconds. **Light.** |
| **coralForge on real PDB proteins** | Paper 10 | RCSB PDB (experimental structures), UniRef90 (MSA) | PDB: ~100GB total, 1–50KB per protein. UniRef90: ~100GB compressed | NestGate `EFetch` for PDB sequences. Bulk UniRef90 download via FTP. | MSA generation: CPU-heavy (JackHMMer equiv, minutes–hours per protein on EPYC). Structure prediction forward pass: GPU-heavy (matmul chains). **Medium per protein, heavy at scale.** |
| **No-till soil microbiome real data** | gen3 Paper 06 | Earth Microbiome Project (EMP), Brandt farm time series | EMP: ~2TB 16S amplicon. Brandt: custom acquisition | EMP: public FTP. Brandt: field collection | 16S → diversity → Anderson r(t): CPU-light (wetSpring pipeline). **Light.** |
| **LTEE frozen fossil sequences** | gen3 Paper 02 | Lenski lab E. coli genome data (NCBI BioProject) | ~100GB raw reads, ~500MB assembled genomes | NestGate `ESearch`/`EFetch` from NCBI SRA/GenBank | Alignment: CPU-medium. Variant calling: CPU-medium. coralForge structural prediction: GPU-heavy per mutant. **Medium.** |
| **NCBI metagenomes for QS gene scan** | gen3 Paper 05 | SRA metagenome datasets (170 communities) | ~500GB–1TB raw reads | NestGate bulk SRA fetch | Assembly: CPU-heavy (Strandgate EPYC ideal). QS gene detection: CPU-light. **Heavy at acquisition, light at analysis.** |

### Axis 2: Novel Compositions (No New Math)

| Extension | Components Combined | Compute | Impact |
|-----------|-------------------|---------|--------|
| **Anderson spectral analysis of attention weights** | nS-01 (`eigh_f64`, `BatchIprGpu`) + coralForge (Pairformer attention) | Light — eigendecomposition of small matrices | If attention weight spectra predict structure quality, connects biophysical AI interpretability to structural biology |
| **WDM surrogate ensemble QS** | nS-05 (game theory, Anderson QS) + nW-01..05 (WDM surrogates) | Light — reuse existing validators | Treat surrogate ensemble as "quorum" — when do predictions agree? Disagreement = phase boundary signal |
| **LSTM anomaly detection on MD trajectories** | nW-03 (LSTM S(q,ω)) + hotSpring transport coefficients | Light — feed existing MD output through existing LSTM | Phase transition signatures in transport time series |
| **HMM introgression applied to neural network layers** | nS-04 (introgression.rs) + pretrained model layer weights | Light — reuse HMM infrastructure | Detect "knowledge transfer" between neural network layers via introgression statistics |

### Axis 3: System Scaling

| Extension | What Changes | Hardware | Timeline |
|-----------|-------------|----------|----------|
| **DF64 Anderson at L=14–20** | Large-lattice eigenproblems at extended precision | GPU-heavy (Northgate RTX 5090: 32GB VRAM for large matrices) | After LAN |
| **Streaming coralForge pipeline** | Single CommandEncoder for full structure prediction | Eastgate RTX 4070 (proven substrate) | After `Tensor::gelu()` in ToadStool |
| **Multi-gate coralForge** | MSA on Strandgate EPYC (64c), GPU inference on Northgate 5090 | LAN required (10GbE) | After LAN + biomeOS deployment |
| **NPU triage for structure prediction** | AKD1000 classifies which sequences merit full prediction | Eastgate (1 NPU) or Strandgate (2 NPU) | After NPU-GPU co-processing pattern validated |

---

## Data Hunger Assessment

### Tier 0: Zero External Data (ready now)

These extensions use only synthetic data or data we generate ourselves:

- **nS-01 controlled experiments** (Exp-050, 051): Train small MLPs/LSTMs
  on MNIST/synthetic, extract weight matrices at checkpoints. Already
  defined in PAPER_OUTLINES.md. **Data: 0 bytes external.**
- **nS-05 multi-agent coordination** (Exp-053): Simulated agents with
  Anderson QS dynamics. **Data: 0 bytes external.**
- **WDM surrogate ensemble**: Compose existing nW-01..05 validators.
  **Data: 0 bytes external.**
- **Attention weight spectral analysis**: Run coralForge on synthetic
  sequences, extract attention matrices, run `eigh_f64`. **Data: 0 bytes.**

### Tier 1: Small External Data (~1–10 GB)

- **Pretrained model weights** (nS-01 Paper A): Download 5–10 torchvision
  models (ResNet-18/50/152, ViT-B/L, GPT-2 small). Convert .pth to raw
  weight tensors. ~5GB total. **No NestGate needed — direct HTTP download.**
- **PDB reference structures** (Paper 10): 10–20 small proteins from
  RCSB PDB. ~100KB each. **NestGate EFetch or direct RCSB REST API.**
- **LTEE assembled genomes**: E. coli K-12 + citrate+ mutant genomes from
  NCBI GenBank. ~500MB. **NestGate EFetch.**
- **EMP pilot**: 50–100 soil 16S samples from Earth Microbiome Project.
  ~10GB. **Public FTP.**

### Tier 2: Medium External Data (~10–100 GB)

- **UniRef90 for MSA**: Required for real coralForge structure prediction.
  ~100GB compressed. **Bulk FTP download to Westgate (76TB cold storage).**
  NestGate content-addressed storage ideal.
- **LTEE raw reads**: SRA BioProject PRJNA294072. ~100GB.
  **NestGate SRA fetch or direct aspera download.**

### Tier 3: Large External Data (~100 GB–1 TB+)

- **Full EMP dataset**: ~2TB 16S amplicon data across thousands of samples.
  **Westgate cold storage or Strandgate (20TB+).**
- **SRA metagenomes for QS gene scan**: 170 communities × 5–10GB each.
  **Strandgate storage. NestGate bulk SRA pipeline.**
- **Full PDB mirror**: ~100GB (all 220K structures). Useful for comprehensive
  coralForge validation. **Westgate cold storage.**

---

## Compute Hunger Assessment

### By Hardware Class

| Task | CPU Demand | GPU Demand | Memory | Best Gate(s) |
|------|-----------|-----------|--------|-------------|
| Weight spectral (`eigh_f64` for 1024×1024) | Light (~100ms) | Light (`BatchIprGpu` ~1ms) | <100MB | Eastgate |
| coralForge forward pass (100 residues) | — | Medium (~seconds, matmul chains) | ~1GB VRAM | Eastgate (4070) or Northgate (5090) |
| MSA generation (JackHMMer, 100 residues) | **Heavy** (~10 min/protein on 8 cores) | — | ~8GB RAM | **Strandgate** (64 EPYC cores) |
| 16S diversity → Anderson r(t) | Light (~seconds) | — | <1GB | Any |
| DF64 Anderson L=16 eigensolve | — | **Heavy** (~minutes, large matrices) | ~8GB VRAM | **Northgate** (5090, 32GB) |
| LSTM/ESN anomaly detection | Light | Light (if GPU-promoted) | <1GB | Eastgate |
| WDM surrogate ensemble | Light | Light | <1GB | Eastgate |
| Metagenome assembly (per sample) | **Heavy** (~1 hour on 16 cores) | — | ~32GB RAM | **Strandgate** (256GB ECC) |

### Effective Compute Budget (single Eastgate)

On Eastgate alone (i9-12900, RTX 4070, 32GB):

| What You Can Do | Time | Feasibility |
|-----------------|------|-------------|
| nS-01 Paper A: 10 pretrained models × 50 layers × eigendecomp | ~1 hour total | **Immediate** |
| coralForge: 20 small proteins, forward pass only | ~30 min total | **Immediate** (needs pipeline assembly) |
| No-till Anderson on 50 EMP samples | ~5 min total | **Immediate** (needs wetSpring 16S) |
| LTEE genome variant calling (10 genomes) | ~2 hours | **Immediate** |
| DF64 Anderson L=14 | ~20 min | **Immediate** |
| DF64 Anderson L=20 | ~hours, may exceed 12GB VRAM | **Northgate needed** |
| MSA for 100 proteins | ~16 hours (sequential) | **Strandgate much better** |

### With LAN Towers (10GbE)

| Pipeline | Gate Roles | Speedup |
|----------|-----------|---------|
| coralForge multi-protein | Strandgate: MSA (64 EPYC cores) → Northgate: GPU inference (5090) → Westgate: result storage (76TB) | ~10× vs Eastgate serial |
| Metagenome QS scan | Strandgate: assembly → Eastgate: Anderson analysis → Westgate: archive | ~8× for CPU-heavy phases |
| DF64 large lattice | Northgate: 5090 (32GB VRAM) → biomeGate: Titan V (12GB, alternative precision) | Unlocks L=20+ |
| Batch weight spectral | Northgate: large model weights → multiple GPUs | ~4× for large models |

---

## Primal Integration Roadmap

### Phase 0: Local Atomic (Eastgate only, now)

neuralSpring already has metalForge substrate discovery and NUCLEUS-pattern
validation (47/47 mixed-hardware, 41/41 metalForge NUCLEUS). The next step
is connecting to actual primals rather than validating the patterns.

**biomeOS NUCLEUS locally on Eastgate:**

```
Tower (BearDog + Songbird) — crypto, discovery, TLS
  └── Node (Tower + ToadStool) — compute, GPU dispatch
       └── Nest (Tower + NestGate) — storage, NCBI data acquisition
```

- `biomeos nucleus start --mode full --node-id eastgate`
- NestGate fetches PDB/UniRef/NCBI sequences
- ToadStool provides GPU compute
- All orchestrated locally, no LAN needed

**What this unlocks:**
- NestGate `EFetch` for PDB structures → coralForge validation against real data
- NestGate `ESearch` for LTEE BioProject sequences → Paper 02 extension
- Content-addressed storage (BLAKE3) for downloaded reference databases
- Squirrel AI routing for experiment orchestration

### Phase 1: LAN Atomic (multi-gate, after 10GbE cabling)

```
Eastgate (Tower+Node)     — primary dev, GPU inference, NPU triage
Strandgate (Node+Nest)    — heavy CPU (64 EPYC cores), bulk storage (20TB+), 2 NPU
Northgate (Node)          — flagship GPU (5090, 32GB VRAM), heavy inference
Westgate (Nest)           — cold storage (76TB ZFS), archival
biomeGate (Node)          — Titan V (alternative GPU), 3090, NPU
```

**biomeOS Plasmodium** connects gates:
- `biomeos plasmodium status` — see all gate health
- Neural API routes `capability.call("compute.gpu.matmul", ...)` to best gate
- Workloads auto-route: MSA → Strandgate, GPU inference → Northgate, storage → Westgate

**What this unlocks:**
- Multi-gate coralForge: MSA generation parallelized across EPYC cores,
  structure prediction on 5090, results archived on ZFS
- Batch weight spectral analysis: large models on Northgate's 32GB VRAM
- DF64 Anderson L=20+ on Northgate 5090
- Metagenome assembly on Strandgate (256GB ECC ideal)
- Cross-gate validation: same computation on multiple GPUs confirms portability

### Phase 2: Science Extensions (after Phase 0/1 infrastructure)

| Priority | Extension | Phase Needed | First Data | Primal Dependencies |
|----------|-----------|-------------|------------|---------------------|
| **P1** | nS-01 Paper A (weight spectral on real models) | Phase 0 | torchvision .pth files | None (HTTP download) |
| **P2** | coralForge real PDB validation (10 proteins) | Phase 0 | PDB REST API / NestGate | NestGate (EFetch) |
| **P3** | Attention weight spectral analysis | Phase 0 | Synthetic (coralForge output) | None |
| **P4** | LTEE genome sequences | Phase 0 | NestGate NCBI SRA | NestGate (ESearch/EFetch) |
| **P5** | No-till EMP pilot (50 samples) | Phase 0 | EMP FTP download | wetSpring 16S pipeline |
| **P6** | coralForge at scale (100 proteins + MSA) | Phase 1 | UniRef90 (100GB) | NestGate (bulk FTP), Strandgate |
| **P7** | DF64 Anderson L=16–20 | Phase 1 | Synthetic (large lattices) | Northgate 5090 |
| **P8** | Metagenome QS scan (170 samples) | Phase 1 | SRA metagenomes (500GB+) | NestGate (SRA bulk), Strandgate |
| **P9** | Multi-gate streaming pipeline | Phase 1 | Any workload | biomeOS Plasmodium |
| **P10** | LTEE structural evolution (coralForge + LTEE) | Phase 1 | LTEE sequences + UniRef90 | NestGate + Strandgate + Northgate |

---

## What NestGate Provides Today

NestGate's `NCBILiveProvider` already supports:

```rust
// Search NCBI databases
let results = ncbi.search("nucleotide", "E. coli K-12[organism]", 100).await?;

// Fetch sequences
let sequence = ncbi.fetch("nucleotide", "U00096.3", "fasta").await?;

// Summary metadata
let summary = ncbi.summary("protein", "P0A8V2").await?;
```

**For neuralSpring extensions:**
- P2 (PDB validation): `EFetch` protein sequences from UniProt/PDB
- P4 (LTEE genomes): `ESearch` BioProject PRJNA294072, `EFetch` assembled genomes
- P6 (MSA databases): bulk download coordination (UniRef90 FTP)
- P8 (metagenomes): `ESearch` SRA studies, `EFetch` read sets

**Not yet available** (NestGate evolution targets):
- SRA Toolkit integration for bulk FASTQ download
- Content-addressed database mirroring (UniRef90, PDB70)
- Streaming download → processing pipeline

---

## What biomeOS NUCLEUS Provides Today

biomeOS NUCLEUS is operational in `biomeos-nucleus` crate:

| Atomic | Binary | Capabilities |
|--------|--------|--------------|
| Tower | `tower` | BearDog crypto + Songbird discovery + TLS |
| Node | `nucleus --mode node` | Tower + ToadStool compute dispatch |
| Nest | `nucleus --mode nest` | Tower + NestGate storage |
| Full | `nucleus --mode full` | All above + Squirrel AI + Neural API |

**For neuralSpring extensions:**
- Phase 0: `nucleus --mode full --node-id eastgate` — local orchestration
- Phase 1: `biomeos plasmodium` — multi-gate LAN coordination
- Neural API: `capability.call("compute.gpu.eigendecomp", ...)` → routes to best gate

**Integration path:** neuralSpring's metalForge substrate model aligns with
NUCLEUS atomics. The bridge is:
1. metalForge `discover()` → NUCLEUS `Tower` hardware inventory
2. metalForge `dispatch()` → NUCLEUS `Node` compute routing
3. metalForge `provenance()` → NUCLEUS `Nest` result storage

---

## Summary: Resource Requirements

| Extension Category | Data | Compute | Timeline | Phase |
|-------------------|------|---------|----------|-------|
| Weight spectral on real models | ~5GB (model weights) | **Light** (hours on Eastgate) | **Now** | 0 |
| coralForge 10 small proteins | ~10MB (PDB) | **Light-Medium** (30 min Eastgate) | **Now** (pipeline assembly needed) | 0 |
| Synthetic compositions (Axis 2) | 0 | **Light** | **Now** | 0 |
| LTEE genomes (10 strains) | ~500MB (NCBI) | **Medium** (2 hours Eastgate) | **Now** (NestGate fetch) | 0 |
| coralForge at scale (100 proteins) | ~100GB (UniRef90) | **Heavy** (days on Eastgate, hours on LAN) | After LAN | 1 |
| DF64 Anderson L=20 | 0 | **Heavy** (Northgate 5090 needed) | After LAN | 1 |
| Metagenome QS scan | ~500GB–1TB (SRA) | **Heavy** (Strandgate EPYC) | After LAN | 1 |
| Multi-gate streaming | varies | Distributed | After biomeOS Plasmodium | 1 |

**Bottom line**: P1–P5 are doable on Eastgate alone with <10GB of external
data. P6–P10 benefit enormously from LAN towers. Nothing here requires
cloud — the aggregate 176GB GPU VRAM + 1.2TB RAM + 105TB storage is more
than sufficient for every planned extension.

---

*neuralSpring baseCamp extension plan — March 1, 2026. S98+ planning.
Data: 0–1TB depending on scope. Compute: Eastgate sufficient for P1–P5,
LAN towers for P6–P10. Primal integration: NestGate for data, biomeOS
NUCLEUS for orchestration, ToadStool for GPU compute. Zero cloud
dependency. All infrastructure sovereign.*
