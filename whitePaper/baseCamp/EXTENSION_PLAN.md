# neuralSpring baseCamp: Extension Plan — From Validated Science to Discovery

**Date**: Jun 6, 2026 (Session S225)
**Author**: Kevin Mok (BS Microbiology, MSU 2018; MS Data Science, MSU 2025)
**Status**: ACTIVE — Axis 2 compositions complete. GPU dispatch evolution complete. **Python→Rust→Primal→Live Composition four-tier validation stack** — proto-nucleate aligned to upstream `downstream_manifest.toml`. playGround: compute triangle + Squirrel MCP + HuggingFace Model Lab. **754 workspace tests** (IPC-first), **269 binaries**, **521+ `.rs` files**. Wave 17+20 signal API (`primal.announce` + `nest.store` + `nest.commit` + `node.compute`). 45 capabilities, 10 validation scenarios. 6/6 GPU dispatch. V171 handoff. barraCuda v0.4.0.
**S151–S152 update**: Deep audit + deep debt execution — ecoBin compliance, 15+ tolerance literals centralized, capability-based primal discovery, coralReef bridge capability-first, shared validation infrastructure (`validate_tensor_binary`, `gen_test_f64`), V103 handoff.
**S148–S150 update**: playGround evolution — Squirrel MCP adapter (14 tool definitions), HuggingFace Model Lab (GPT-2 inference on barraCuda), compute triangle (ToadStool/coralReef IPC clients, hot/cold dispatch benchmarks: 7–45× pipeline reuse, 8–22× PyTorch/CUDA gap). 63 playGround unit + 13 integration tests. Live ToadStool verified. V101 handoff.
**S147 update**: Deep debt — zero inline magic numbers, zero duplicate math, capability-based discovery. V100 handoff.
**S145–S146 update**: barraCuda v0.3.5 (`0649cd0`), 5 workload rewires, NUCLEUS GPU dispatch, 4 GPU experiments (Exp 103–106), industry GPU parity benchmarks. 1115 lib tests, 73 forge tests, 260 binaries, 47 modules, 0 clippy. 25 absorbed workloads. toadStool S146+, coralReef Phase 10.
**S143–S144 update**: 5 novel composition experiments (Exp 097–101), petalTongue composition visualization, NUCLEUS pipeline executor.
Axis 2 "Novel Compositions (No New Math)" complete for all locally composable modules.

---

## Where We Stand

neuralSpring has validated 4,200+ checks across **27 papers** (full queue complete),
5 WDM surrogates, coralForge (nF-01/02/03), 6 baseCamp sub-theses, and 3
publication experiments. The full pipeline is proven: Python → Rust CPU →
BarraCUDA CPU → GPU Tensor → Pure GPU → metalForge cross-substrate → NUCLEUS
composition validation. 1,225 lib tests, 264 binaries, zero clippy, zero debt.
83.6× faster than Python. NUCLEUS composition validators prove primal
composition patterns alongside science validation targets.

**Session 104b state**: Complete cross-spring rewire to modern ToadStool f97fc2ae.
15 core functions delegate to upstream BarraCUDA (chi², KL divergence, spectral
bandwidth/condition, 7 Dispatcher domain ops, graph/belief/hessian/boltzmann).
Cross-spring provenance fully mapped: hotSpring precision → wetSpring bio →
neuralSpring domain → ToadStool fused ops → consumed by all Springs. All
validation binaries source shaders from single-source-of-truth forge crate.
`FusedChiSquaredGpu` and `FusedKlDivergenceGpu` are round-trip examples:
neuralSpring domain shaders absorbed by ToadStool, evolved to f64 fused ops,
now consumed back by neuralSpring at higher precision than the originals.

**What's missing**: everything has been validated against published science,
but almost none of it has been applied to new data or new systems. The
pipeline is proven portable. The extension phase applies it to discovery.

**What's ready for primal integration**: NestGate has `NCBILiveProvider` wired
(`data.ncbi_search`, `data.ncbi_fetch`). biomeOS NUCLEUS has Tower/Node/Nest
atomics. neuralSpring's metalForge substrate model aligns with NUCLEUS patterns
(47/47 mixed-hardware, 41/41 NUCLEUS validators). The infrastructure primals
are mature enough to start incorporating into the science pipeline.

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
| **Anaerobic digester microbiome 16S** | gen3 Paper 16 | NCBI BioProjects (ADREC, municipal WWTP, thermophilic/mesophilic) | ~5–50GB per BioProject | NestGate `ESearch`/`EFetch` | 16S → diversity → Anderson W → ESN yield prediction. **Light.** Paper 027 ESN validated. |
| **Real digester operational data** | gen3 Paper 16 | Published supplementary tables (Wang 2020, other Liao group papers) | ~10MB–100MB (tabular) | Manual + NestGate | Direct ESN training on real T/pH/OLR/HRT/yield. **Trivial.** |
| **FNR/ArcAB/Rex QS regulon mapping** | gen3 Paper 16 | NCBI Protein (anaerobic QS gene families) | ~100MB | NestGate `ESearch` | Map oxygen-regulated QS genes in digester taxa. **Light.** |
| **OhioT1DM / OpenAPS real CGM data** | Paper 026 ext | OhioT1DM (public), OpenAPS Data Commons | ~1–5GB | Direct download | LSTM glucose prediction on real patient data. **Light.** Paper 026 LSTM validated. |

### Axis 2: Novel Compositions (No New Math)

| Extension | Components Combined | Compute | Impact |
|-----------|-------------------|---------|--------|
| **Anderson spectral analysis of attention weights** | nS-01 (`eigh_f64`, `BatchIprGpu`) + coralForge (Pairformer attention) | Light — eigendecomposition of small matrices | If attention weight spectra predict structure quality, connects biophysical AI interpretability to structural biology |
| **WDM surrogate ensemble QS** | nS-05 (game theory, Anderson QS) + nW-01..05 (WDM surrogates) | Light — reuse existing validators | Treat surrogate ensemble as "quorum" — when do predictions agree? Disagreement = phase boundary signal |
| **LSTM anomaly detection on MD trajectories** | nW-03 (LSTM S(q,ω)) + hotSpring transport coefficients | Light — feed existing MD output through existing LSTM | Phase transition signatures in transport time series |
| **HMM introgression applied to neural network layers** | nS-04 (introgression.rs) + pretrained model layer weights | Light — reuse HMM infrastructure | Detect "knowledge transfer" between neural network layers via introgression statistics |
| **ESN digester yield + Anderson QS disorder coupling** | Paper 027 (digestion_prediction.rs) + Anderson QS (anderson_localization.rs) + wetSpring 16S | Light — compose existing modules | ESN predicts methane yield from operational params; Anderson W predicts QS regime from community structure. Couple: does W predict ESN residuals? |
| **ESN/LSTM ensemble: digester + glucose + weather** | Paper 027 ESN + Paper 026 LSTM + Study 003/004 LSTM | Light — run existing predictors, compare reservoir dynamics | Same reservoir computing architecture across three domains (bioprocess, biomedical, meteorology). Isomorphic thesis proof. |
| **Digester → gut cross-domain transfer** | Paper 027 (ESN digester) + healthSpring (Anderson gut lattice) | Light — same ESN architecture, different domain | Anaerobic digester = engineered gut. Transfer learning: does ESN trained on digester params predict gut fermentation metrics? |

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

### Tier 2: Medium External Data (~10–100 GB) — PINNED until NUCLEUS

- **UniRef90 for MSA**: Required for real coralForge structure prediction.
  ~100GB compressed. **PINNED** — bulk FTP download to Westgate (76TB ZFS).
  Unpin: NUCLEUS Nest online + Westgate storage.
- **LTEE raw reads**: SRA BioProject PRJNA294072. ~100GB.
  **PINNED** — NestGate SRA pipeline needed. Unpin: NestGate SRA Toolkit.

### Tier 3: Large External Data (~100 GB–1 TB+) — PINNED until NUCLEUS + LAN

- **Full EMP dataset**: ~2TB. **PINNED** — Westgate cold storage.
- **SRA metagenomes for QS gene scan**: ~500GB–1TB. **PINNED** — Strandgate.
- **Full PDB mirror**: ~100GB. **PINNED** — Westgate cold storage.

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

### Effective Compute Budget (single Eastgate — ACTIVE extensions only)

On Eastgate alone (i9-12900, RTX 4070, 32GB):

| Active Extension | Time | Data | Feasibility |
|------------------|------|------|-------------|
| P1: 10 pretrained models × 50 layers × eigendecomp | ~1 hour | ~5GB | **GO** |
| P2: coralForge 20 small proteins, forward pass only | ~30 min | ~10MB | **GO** (pipeline assembly needed) |
| P3: Attention weight spectral on coralForge output | ~10 min | 0 | **GO** |
| P4: LTEE genome variant calling (10 genomes) | ~2 hours | ~500MB | **GO** |
| P5: No-till Anderson on 50 EMP samples | ~5 min | ~10GB | **GO** (needs wetSpring 16S) |
| P7: DF64 Anderson L=14 | ~20 min | 0 | **GO** |
| P13: Digester 16S microbiome (5 BioProjects) | ~30 min | ~5–50GB | **GO** (NestGate + wetSpring 16S) |
| P14: Real digester operational data (tabular) | ~5 min | ~10MB | **GO** (ESN retrain) |
| P15: FNR/ArcAB/Rex QS regulon mapping | ~15 min | ~100MB | **GO** (NestGate Protein search) |
| P16: OhioT1DM real CGM data | ~1 hour | ~1–5GB | **GO** (LSTM retrain) |
| Axis 2 compositions (7 novel combinations) | ~1 hour total | 0 | **GO** |
| **Total active compute** | **~7 hours** | **~26–76GB** | **All on Eastgate** |

### With LAN Towers (unpins P6–P12)

| Pinned Extension | Gate Roles | Speedup vs Eastgate |
|------------------|-----------|---------------------|
| P6: coralForge multi-protein | Strandgate: MSA (64 EPYC) → Northgate: GPU (5090) → Westgate: ZFS | ~10× |
| P7b: DF64 Anderson L=20+ | Northgate: 5090 (32GB VRAM) | Unlocks (impossible on Eastgate) |
| P8: Metagenome QS scan | Strandgate: assembly → Eastgate: Anderson → Westgate: archive | ~8× |
| P9: Multi-gate streaming | All gates via Plasmodium | Distributed |
| P10: LTEE structural evo | Strandgate MSA + Northgate GPU + Westgate archive | ~10× |

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

**ACTIVE — data < 100GB, manageable on Eastgate local until NUCLEUS online:**

| Priority | Extension | Data | Compute | Status |
|----------|-----------|------|---------|--------|
| **P1** | nS-01 Paper A (weight spectral on real models) | ~5GB (.pth) | Light (~1hr Eastgate) | **ACTIVE** — local |
| **P2** | coralForge real PDB validation (10 proteins) | ~10MB (PDB REST) | Medium (~30min) | **ACTIVE** — local |
| **P3** | Attention weight spectral analysis | 0 (synthetic) | Light | **ACTIVE** — local |
| **P4** | LTEE genome sequences (10 assembled genomes) | ~500MB (NCBI) | Medium (~2hr) | **ACTIVE** — local |
| **P5** | No-till EMP pilot (50 samples) | ~10GB (EMP FTP) | Light (~5min) | **ACTIVE** — local |
| **P7** | DF64 Anderson L=14 (fits 12GB VRAM) | 0 (synthetic) | Medium (~20min) | **ACTIVE** — local |
| **P13** | Digester 16S microbiome (NCBI BioProjects) | ~5–50GB | Light (wetSpring 16S) | **ACTIVE** — NestGate EFetch |
| **P14** | Real digester operational data (Wang 2020 supp) | ~10MB | Trivial (ESN retrain) | **ACTIVE** — manual |
| **P15** | FNR/ArcAB/Rex QS regulon mapping (NCBI Protein) | ~100MB | Light | **ACTIVE** — NestGate |
| **P16** | OhioT1DM / OpenAPS real CGM data | ~1–5GB | Light (LSTM retrain) | **ACTIVE** — download |
| **P17** | ESN digester × Anderson QS coupling | 0 (compose) | Light | **DONE** — Exp 096 (S143) |
| **P18** | ESN/LSTM ensemble isomorphic thesis | 0 (compose) | Light | **DONE** — Exp 097 (S143) |

**PINNED — data ≥ 100GB or requires LAN hardware, deferred until NUCLEUS/LAN:**

| Priority | Extension | Data | Pin Reason | Unpin When |
|----------|-----------|------|------------|------------|
| **P6** | coralForge at scale (100+ proteins + MSA) | ~100GB (UniRef90) | UniRef90 bulk download + Strandgate EPYC for MSA | NUCLEUS Nest online + Westgate ZFS |
| **P7b** | DF64 Anderson L=20+ (exceeds 12GB VRAM) | 0 | Northgate 5090 32GB VRAM needed | LAN + Northgate online |
| **P8** | Metagenome QS scan (170 samples) | ~500GB–1TB (SRA) | Bulk SRA download + Strandgate assembly | NUCLEUS + NestGate SRA pipeline |
| **P9** | Multi-gate streaming pipeline | varies | Requires biomeOS Plasmodium | LAN + Plasmodium deployed |
| **P10** | LTEE structural evolution (coralForge + LTEE) | UniRef90 + LTEE reads (~200GB) | UniRef90 dependency (P6) | P6 unpin + Strandgate |
| **P11** | Full EMP metagenome reanalysis | ~2TB (EMP 16S) | Bulk download + Strandgate | NUCLEUS + Westgate ZFS |
| **P12** | Full PDB mirror for coralForge benchmark | ~100GB (220K structures) | Bulk download | NUCLEUS Nest + Westgate ZFS |

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

## Summary: Resource Requirements (100GB pin boundary)

### ACTIVE — proceed on Eastgate local

| Extension | Data | Compute | Status |
|-----------|------|---------|--------|
| P1: Weight spectral on real models | ~5GB | Light (~1hr) | **GO** |
| P2: coralForge 10 small proteins | ~10MB | Medium (~30min) | **GO** |
| P3: Attention weight spectral | 0 | Light | **GO** |
| P4: LTEE genomes (10 strains) | ~500MB | Medium (~2hr) | **GO** |
| P5: No-till EMP pilot (50 samples) | ~10GB | Light (~5min) | **GO** |
| P7: DF64 Anderson L=14 | 0 | Medium (~20min) | **GO** |
| P13: Digester 16S microbiome | ~5–50GB | Light | **GO** (NestGate) |
| P14: Real digester operational | ~10MB | Trivial | **GO** |
| P15: FNR/ArcAB/Rex regulons | ~100MB | Light | **GO** (NestGate) |
| P16: OhioT1DM / OpenAPS CGM | ~1–5GB | Light (~1hr) | **GO** |
| P17: ESN × Anderson QS coupling | 0 | Light | **DONE** (Exp 096) |
| P18: ESN/LSTM isomorphic ensemble | 0 | Light | **DONE** (Exp 097) |
| Axis 2: WDM ensemble QS | 0 | Light | **DONE** (Exp 098) |
| Axis 2: HMM introgression NN | 0 | Light | **DONE** (Exp 099) |
| Axis 2: Attention Anderson spectral | 0 | Light | **DONE** (Exp 100) |
| Axis 2: LSTM anomaly on MD trajectories | 0 | Light | **BLOCKED** (needs hotSpring MD data) |
| Axis 2: Digester→gut cross-domain transfer | 0 | Light | **BLOCKED** (needs healthSpring) |

**Total active data: ~26–76GB.** Fits on Eastgate local with room to spare.

### PINNED — data ≥ 100GB or LAN hardware required

| Extension | Data | Pin Reason | Unpin Trigger |
|-----------|------|------------|---------------|
| P6: coralForge at scale + MSA | ~100GB | UniRef90 bulk | NUCLEUS Nest + Westgate |
| P7b: DF64 Anderson L=20+ | 0 (compute) | 32GB VRAM needed | Northgate 5090 on LAN |
| P8: Metagenome QS scan | ~500GB–1TB | SRA bulk download | NUCLEUS + Strandgate |
| P9: Multi-gate streaming | varies | Plasmodium needed | LAN + biomeOS |
| P10: LTEE structural evo | ~200GB | UniRef90 dep (P6) | P6 unpin |
| P11: Full EMP reanalysis | ~2TB | Bulk download | NUCLEUS + Westgate |
| P12: Full PDB mirror | ~100GB | Bulk download | NUCLEUS Nest + Westgate |

**Total pinned data: ~3TB.** Lives on Westgate ZFS (76TB) once LAN online.

**Bottom line**: 13 active extensions on <76GB local data. 7 pinned extensions
on ~3TB deferred until NUCLEUS online + 10GbE cabled. Nothing requires cloud.
The 100GB boundary cleanly separates what Eastgate handles solo from what
needs the LAN aggregate (176GB GPU VRAM, 1.2TB RAM, 105TB storage).
Paper 027 completion (S142) unlocked P13–P18: digester microbiome, real
operational data, QS regulon mapping, real CGM, and two novel compositions.

---

---

## Primal Incorporation Strategy (Session 104b+)

Before building extensions, incorporate infrastructure primals into the
science pipeline. This is the integration layer between validated math
and real discovery.

### Step 0: Wire NestGate data acquisition locally (Eastgate, no LAN)

NestGate `NCBILiveProvider` is already wired. Wire it into neuralSpring's
experiment pipeline:

```
neuralSpring experiment binary
    → JSON-RPC: data.ncbi_search("nucleotide", "E. coli K-12", 10)
    → NestGate NCBILiveProvider
    → NCBI E-utilities (https://eutils.ncbi.nlm.nih.gov)
    → sequences → eigh_f64 / coralForge / HMM
```

**Concrete wiring**: neuralSpring experiment binary opens NestGate Unix
socket (`$XDG_RUNTIME_DIR/biomeos/nestgate-{family}.sock`), sends
JSON-RPC 2.0, receives FASTA/GenBank. No HTTP in neuralSpring — data
acquisition is NestGate's responsibility (capability separation).

### Step 1: Start biomeOS NUCLEUS locally (Eastgate, Tower mode)

```bash
biomeos nucleus start --mode full --node-id eastgate
```

This starts:
- BearDog: crypto (key management, primal identity)
- Songbird: discovery (primal registry, capability advertisement)
- ToadStool: compute dispatch (GPU streaming)
- NestGate: storage + data acquisition (NCBI, PDB)
- Squirrel: AI routing (experiment orchestration)

neuralSpring registers as a primal via `neuralspring`:
```bash
neuralspring --family-id eastgate
```

11 science capabilities become available to the NUCLEUS:
`science.ipr`, `science.spectral_analysis`, `science.anderson_localization`,
`science.hessian_eigen`, `science.agent_coordination`, etc.

### Step 2: First real-data experiment via NUCLEUS

**Target**: nS-01 Paper A (Weight Spectral Analysis on Pretrained Models)

Pipeline:
1. neuralSpring experiment binary requests model weights
2. → biomeOS routes to Squirrel → NestGate → HuggingFace `data.hf_fetch`
3. Weights downloaded → content-addressed storage (BLAKE3)
4. neuralSpring `eigh_f64` → eigendecomposition per layer
5. Results → NestGate Nest storage → Westgate ZFS archive (when LAN ready)

**Data**: ~5GB (10 model checkpoints). **Compute**: ~1 hour on Eastgate. **Phase 0.**

### Step 3: Wire PDB data for coralForge (Phase 0)

NestGate NESTGATE_PROVIDER_PLAN.md defines the PDB provider:
- `data.pdb_search(query)` → RCSB PDB
- `data.pdb_fetch(pdb_id)` → mmCIF/PDB coordinate files
- Storage: Westgate ZFS `/data/nestgate/pdb/`

coralForge pipeline: PDB sequence → MSA (via UniRef90) → structure prediction.
Phase 0 can do forward-pass-only (no MSA) on small proteins.

### Step 4: LAN deployment (after 10GbE cabling)

The gen3/about/HARDWARE.md shows 10G backbone: switch acquired, NICs
installed, cables pending. Once cabled:

```
biomeos plasmodium start

# Gates join automatically via Songbird mDNS discovery
Eastgate:   Tower+Node (dev + NPU + RTX 4070)
Strandgate: Node+Nest  (64 EPYC cores + 256GB ECC + 2 NPU)
Northgate:  Node       (RTX 5090 32GB VRAM — flagship GPU)
Westgate:   Nest       (76TB ZFS cold storage)
biomeGate:  Node       (Titan V + 3090 + NPU)
```

Workload auto-routing via biomeOS capability system:
- `compute.msa` → Strandgate (CPU-heavy, 64 EPYC cores)
- `compute.gpu.eigendecomp` → Northgate (32GB VRAM)
- `storage.archive` → Westgate (76TB ZFS)
- `data.ncbi_fetch` → any Nest with NestGate
- `compute.gpu.df64` → biomeGate (Titan V native f64)

### Hardware Budget (from gen3/about/HARDWARE.md)

| Resource | Aggregate | Best Gate |
|----------|-----------|-----------|
| CPU cores | ~130+ | Strandgate (64 EPYC) |
| GPU VRAM | ~176GB | Northgate (32GB 5090) |
| System RAM | ~1.2TB | Strandgate (256GB ECC) |
| Storage | ~105TB | Westgate (76TB ZFS) |
| NPU | 4× AKD1000 | Eastgate, Strandgate, biomeGate |

**Bottom line**: The LAN aggregate exceeds many institutional HPC clusters
for our workload profile. The constraint is not hardware — it's cabling
and biomeOS Plasmodium deployment.

---

*neuralSpring baseCamp extension plan — March 10, 2026. S142 update.
27/27 papers complete. Paper queue closed. Extension phase begins.
Data: 0–3TB depending on scope. Compute: Eastgate sufficient for P1–P7 +
P13–P18 (~7 hours, <76GB). LAN towers for P6–P12 (~3TB). Primal integration:
NestGate for data, biomeOS NUCLEUS for orchestration, ToadStool for GPU
compute. Zero cloud dependency. All infrastructure sovereign. Next steps:
(1) wire NestGate data acquisition into first real-data experiment (nS-01),
(2) start biomeOS NUCLEUS locally on Eastgate, (3) begin P13–P18 digester
and biomedical extensions enabled by Paper 027.*
