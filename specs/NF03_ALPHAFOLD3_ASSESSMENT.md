# nF-03: AlphaFold3 Primitive Assessment

**Date**: February 28, 2026 (updated S93)
**Paper**: Abramson et al. "Accurate structure prediction for all molecules"
Nature 630:493-500 (2024)
**Status**: Phase A+B+C DONE — diffusion + Pairformer + confidence heads validated
**Depends**: nF-01 (OpenFold, Phase B.4 DONE), nF-02 (AlphaFold2, Phase B DONE)

## Phase A+B+C Results (Session 93)

| Component | Python | Rust | Max Diff | Unit Tests |
|-----------|--------|------|----------|------------|
| Diffusion primitives | 29/29 | 26/26 | 1.24e-14 | 8 |
| Pairformer block | 14/14 | 13/13 | 6.66e-16 | 3 |
| Confidence heads | 19/19 | 16/16 | 1.42e-14 | 7 |
| **Total** | **62/62** | **55/55** | **Machine ε** | **18** |

Validates: noise schedules (cosine/linear), forward diffusion, DDPM/DDIM reverse,
SE(3)-equivariant operations, sinusoidal timestep embedding, pair conditioning,
full Pairformer block (TriMul + TriAttn + FFN), pLDDT head, PAE head, pDE head,
ranking score (weighted combination).

---

## Architecture Differences: AlphaFold2 vs AlphaFold3

AlphaFold3 replaces the Structure Module with a **diffusion model** and adds
support for RNA, DNA, ligands, and covalent modifications. The Evoformer is
replaced by a **Pairformer** — similar but operating on pair representations
only (no MSA track after initial processing).

```
AlphaFold2:  MSA + Pair → [Evoformer × 48] → [IPA × 8] → Coordinates
AlphaFold3:  MSA + Pair → [MSA Module] → [Pairformer × 48] → [Diffusion × T] → Coordinates
```

### What Changes

| Component | AlphaFold2 | AlphaFold3 | Delta |
|-----------|-----------|------------|-------|
| MSA processing | Evoformer (MSA+pair tracks) | Separate MSA module → pair only | Simpler |
| Pair updates | Triangle mul/attn in Evoformer | **Pairformer** (same ops, pair-only) | Reuse 90% |
| Structure prediction | IPA + backbone update × 8 | **Diffusion denoising** × T steps | NEW MATH |
| Coordinate output | Single prediction | **Sample multiple** → confidence | NEW MATH |
| Molecule types | Protein only | Protein + RNA + DNA + ligand + ion | Tokenization |
| Confidence | pLDDT, PAE | pLDDT, PAE + **pDE** (distance error) | NEW metric |
| Loss function | FAPE | **Diffusion loss** + auxiliary losses | NEW MATH |

---

## New Math Required

### Priority 1: Diffusion Model Core

These are genuinely new primitives that do not exist in neuralSpring or BarraCUDA.

#### 1.1 Gaussian Noise Schedule

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

where `alpha_bar_t = prod(1 - beta_s, s=1..t)` and `beta` follows a cosine
or linear schedule.

**New ops**: noise schedule generation, `alpha_bar` cumulative product
**BarraCUDA status**: cumulative product not available; simple to implement
**Effort**: Low — pure arithmetic, no GPU shader needed

#### 1.2 Denoising Network (Score Prediction)

The diffusion model predicts the noise `epsilon` (or equivalently the score
`nabla log p(x_t)`) given noisy coordinates `x_t` and conditioning from the
Pairformer. This is structurally similar to IPA but operates on noisy
coordinates and is conditioned on the diffusion timestep.

**New ops**: Timestep embedding (sinusoidal or learned), coordinate denoising
**Reuse**: IPA attention mechanism (~70%), backbone frame operations (100%)
**BarraCUDA status**: Sinusoidal embedding is trivial; denoising network
reuses existing attention + frame primitives
**Effort**: Medium — new composition of existing primitives + timestep conditioning

#### 1.3 DDPM / DDIM Sampling

Reverse diffusion process — iteratively denoise from `x_T ~ N(0, I)` to `x_0`:

```
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta(x_t, t))
         + sigma_t * z
```

**New ops**: Reverse step computation, variance scheduling
**BarraCUDA status**: Not available
**Effort**: Low — pure f64 arithmetic per step, chain of GPU dispatches

#### 1.4 SE(3) Equivariant Diffusion

AlphaFold3's diffusion operates on atom coordinates in a SE(3)-equivariant
manner. The model must be invariant to global rotations and translations.

**Reuse**: `quat_to_rotation`, `apply_frame`, `compose_frames` from
`coral_forge/structure/frame.rs` — all validated in nF-01/nF-02
**New ops**: Frame-aligned point cloud diffusion, center-of-mass subtraction
**Effort**: Low — composition of existing frame ops

### Priority 2: Pairformer (Adapted Evoformer)

The Pairformer reuses almost all Evoformer primitives but drops the MSA track.

| Primitive | AlphaFold2 Module | Reuse? | Notes |
|-----------|------------------|--------|-------|
| Triangle mul outgoing | `coral_forge/triangle.rs` | 100% | Same algorithm |
| Triangle mul incoming | `coral_forge/triangle.rs` | 100% | Same algorithm |
| Triangle attention | `coral_forge/triangle.rs` | 100% | Same algorithm |
| Row-wise softmax | `coral_forge/attention.rs` | 100% | Same algorithm |
| Layer normalization | `coral_forge/activation.rs` | 100% | Same algorithm |
| GELU activation | `coral_forge/activation.rs` | 100% | Same algorithm |
| Pair transition (FFN) | NEW | ~80% | Linear → GELU → Linear (existing ops) |

**Effort**: Low — Pairformer is a subset of the Evoformer we already have.

### Priority 3: Multi-Molecule Tokenization

AlphaFold3 handles proteins, RNA, DNA, ligands, and ions using a unified
tokenization scheme:

| Molecule | Token | Atoms/Token | Notes |
|----------|-------|-------------|-------|
| Protein | Residue | ~14 (backbone + sidechain) | Same as AF2 |
| RNA | Nucleotide | ~23 | NEW |
| DNA | Nucleotide | ~22 | NEW |
| Ligand | Atom | 1 | NEW — CCD dictionary |
| Ion | Atom | 1 | NEW |
| Covalent mod | Atom | varies | NEW |

**New ops**: Atom-level tokenizer, CCD (Chemical Component Dictionary) parser,
residue/nucleotide/atom coordinate mapping
**BarraCUDA status**: Not available
**Effort**: Medium — mostly data structure work, not math

### Priority 4: Confidence Heads

| Metric | What It Predicts | Existing? |
|--------|------------------|-----------|
| pLDDT | Per-residue accuracy | Needs implementation (Linear → sigmoid) |
| PAE | Pairwise alignment error | Needs implementation (pair head → softmax) |
| pDE | Predicted distance error | NEW to AF3 |
| Ranking score | Which sample is best | NEW — weighted combination |

**Effort**: Low — all are linear layers + softmax/sigmoid on existing representations

---

## What We Already Have (Reuse Inventory)

| Validated Primitive | nF-01/02 Module | AF3 Use | Reuse % |
|--------------------|----------------|---------|---------|
| `gelu` / `gelu_vec` | `coral_forge/activation.rs` | Pairformer, denoising | 100% |
| `layer_norm` | `coral_forge/activation.rs` | Pairformer, denoising | 100% |
| `softmax_rows` | `coral_forge/activation.rs` | Attention, confidence | 100% |
| `sdpa_scores` / `sdpa_full` | `coral_forge/attention.rs` | Pairformer attention | 100% |
| `triangle_mul_outgoing` | `coral_forge/triangle.rs` | Pairformer | 100% |
| `triangle_mul_incoming` | `coral_forge/triangle.rs` | Pairformer | 100% |
| `triangle_attention_scores` | `coral_forge/triangle.rs` | Pairformer | 100% |
| `outer_product_mean` | `coral_forge/msa.rs` | MSA module (pre-Pairformer) | 100% |
| `ipa_scores` | `coral_forge/structure/ipa.rs` | Denoising network (~70%) | 70% |
| `backbone_update` | `coral_forge/structure/backbone.rs` | Frame updates in diffusion | 100% |
| `torsion_angles` | `coral_forge/structure/backbone.rs` | Side-chain prediction | 100% |
| `quat_to_rotation` | `coral_forge/structure/frame.rs` | SE(3) operations | 100% |
| `apply_frame` / `compose_frames` | `coral_forge/structure/frame.rs` | Equivariant diffusion | 100% |
| 15 WGSL shaders (df64) | `metalForge/shaders/` | GPU acceleration | 100% |

**Reuse estimate**: ~75% of AF3 compute uses primitives we already have.
The ~25% that is new is primarily the diffusion process (noise schedule,
denoising steps, sampling) and multi-molecule tokenization.

---

## Compute Budget (Eastgate: RTX 4070, 12 GB VRAM, 32 GB RAM)

| Task | VRAM | RAM | Time | Feasible? |
|------|------|-----|------|-----------|
| Pairformer (128 res) | ~400 MB | 2 GB | seconds | Yes |
| Pairformer (384 res) | ~1.5 GB | 4 GB | seconds | Yes |
| Diffusion (128 res, T=200) | ~800 MB | 2 GB | minutes | Yes |
| Diffusion (384 res, T=200) | ~3 GB | 4 GB | minutes | Yes |
| Diffusion (1024 res, T=200) | ~10 GB | 8 GB | 10+ min | Tight (gradient checkpoint) |
| Full pipeline (128 res) | ~1.2 GB | 4 GB | minutes | Yes |
| Full pipeline (384 res) | ~4.5 GB | 8 GB | minutes | Yes |
| **Training** (any size) | 24+ GB | 64+ GB | weeks | **No — needs Northgate/Strandgate** |

---

## Data Requirements

| Database | Size | Priority | NestGate Provider |
|----------|------|----------|-------------------|
| UniRef90 | 100 GB compressed | P0 | NEW (UniProt FTP) |
| PDB templates | 200 GB | P0 | NEW (RCSB PDB) |
| CCD dictionary | ~50 MB | P0 | NEW (wwPDB CCD) |
| BFD | 1.7 TB | P1 | NEW (MMseqs2) |
| Rfam | 5 GB | P1 (RNA only) | NEW (EBI) |
| RNAcentral | 50 GB | P1 (RNA only) | NEW (EBI) |

**Phase 1 (proteins only)**: ~300 GB — fits on Eastgate NVMe
**Phase 2 (+ RNA/DNA)**: ~355 GB — fits on Eastgate NVMe
**Phase 3 (full databases)**: ~2.5 TB — needs Westgate ZFS NAS

---

## Buildout Order

### Phase A: Diffusion Primitives — DONE (Session 92)

1. ✅ Noise schedule (cosine/linear beta schedule, cumulative alpha_bar)
2. ✅ Forward diffusion (add noise to clean coordinates)
3. ✅ Reverse step (DDPM/DDIM denoising step)
4. ✅ SE(3)-equivariant noise (center-of-mass removal, translation invariance)
5. ✅ pLDDT confidence head (Linear → sigmoid → [0,1])
6. ✅ PAE confidence head (pair → softmax → expected distance)
7. ✅ Python control: `control/coral_forge/alphafold3_diffusion.py` (29/29)
8. ✅ Rust validation: `validate_alphafold3_diffusion.rs` (26/26)

### Phase B: Pairformer — DONE (Session 92)

1. ✅ Pair-only pipeline (triangle mul/attn + FFN, no MSA track)
2. ✅ Pair transition FFN (Linear → GELU → Linear)
3. ✅ Sinusoidal timestep embedding + pair conditioning
4. ✅ Multi-block iteration (3 blocks with decreasing timestep)
5. ✅ Python control: `control/coral_forge/alphafold3_pairformer.py` (14/14)
6. ✅ Rust validation: `validate_alphafold3_pairformer.rs` (13/13)

### Phase C: Confidence Heads — DONE (Session 93)

1. ✅ pLDDT head (linear → sigmoid)
2. ✅ PAE head (pair → bin softmax)
3. ✅ pDE head (pair → distance error bins)
4. ✅ Ranking score (weighted combination)
5. ✅ Python control: `control/coral_forge/alphafold3_confidence.py` (19/19)
6. ✅ Rust validation: `validate_alphafold3_confidence.rs` (16/16)

### Phase D: Multi-Molecule Tokenization (needs CCD only — 50 MB)

1. CCD dictionary parser
2. Atom-level tokenizer for ligands/ions
3. RNA/DNA nucleotide handling
4. Rust validation: `validate_alphafold3_tokenizer.rs`

### Phase E: Full Pipeline Integration (needs MSA databases)

1. MSA module → Pairformer → Diffusion → Confidence
2. NestGate: PDB + UniRef90 providers
3. MMseqs2/JackHMMER integration via NestGate
4. End-to-end validation on known structures

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Diffusion quality on consumer GPU | Medium | High | df64 core streaming proven for AF2; same approach |
| VRAM limits at 1024+ residues | High | Medium | Gradient checkpointing, tiling |
| MSA database storage | Low | Low | Phased download, Westgate ZFS has 76 TB |
| Training infeasible on Eastgate | Certain | Medium | Northgate (RTX 5090) + Strandgate (EPYC + RTX 3090) |
| RNA/DNA quality without Rfam | Medium | Low | Protein-first, RNA/DNA Phase 2 |

---

*Phase A (diffusion primitives) can begin immediately with zero data dependencies.
All synthetic, deterministic seed 42. Estimated: 2-3 sessions for Py + Rs + GPU.*
