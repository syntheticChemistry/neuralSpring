# neuralSpring White Paper

## The Isomorphic Learning Engine

**Status**: Working draft
**Date**: February 19, 2026
**License**: AGPL-3.0-or-later

---

### Document Index

| Document | Audience | Description |
|----------|----------|-------------|
| [STUDY.md](STUDY.md) | Technical | Main study: experiments, results, BarraCUDA evolution |
| [BARRACUDA_EVOLUTION.md](BARRACUDA_EVOLUTION.md) | ToadStool team | Shader evolution narrative: Python → CPU → GPU |
| `specs/BENCHMARK_ANALYSIS.md` | Engineering | Full 3-way benchmark with analysis |
| `specs/TOADSTOOL_HANDOFF.md` | Engineering | 11 BarraCUDA shortcomings + local fixes |
| `wateringHole/handoffs/` | Cross-project | Formal handoffs (date-stamped) |

---

### What This Study Is

neuralSpring validates machine learning primitives on consumer hardware using
BarraCUDA's WGSL shader library — the same library hotSpring uses for nuclear
physics. The central claim: **all neural architectures decompose into six
fundamental primitives**, and a single engine optimizing those primitives in
WGSL serves every domain.

### Three Questions

1. **Can neural surrogates replace equation chains?**
   Yes. MLP surrogate for FAO-56 ET₀ achieves R²>0.999 with 2000 training
   samples. Same 6-layer pipeline replaces the full Penman-Monteith chain.

2. **Can compiled WGSL shaders beat Python/NumPy for ML inference?**
   Yes, at scale. GPU (RTX 4070) is **104× faster** than single-thread Python
   at 103M FLOPs. CPU (llvmpipe) is **3.9× faster** at the same scale.
   Both execute the same WGSL source — ToadStool compiles to x86 or Vulkan.

3. **Does the hotSpring progression (Python > CPU > GPU) hold for ML?**
   Yes, at crossover scales. The 3-way benchmark achieves
   **GPU < CPU < Python** at MLP large (3.1M FLOPs) and Transformer medium
   (103M FLOPs). GPU dominates CPU by 4–80× at every scale.

---

### Key Results Summary

**Phase 0/0+**: 75/75 Python PASS (48 synthetic + 27 scholarly reproductions)
**Phase 1**: 285/285 Rust validation PASS (43 native + 242 BarraCUDA)

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Synthetic baselines — 5 experiments, 48 checks | **Complete** |
| 0+ | Scholarly reproductions — 5 studies, 27 checks | **Complete** |
| 1a | Rust validation layer — 43 native checks | **Complete** |
| 1b | BarraCUDA validation — 242 checks (10 domains) | **Complete** |
| 1c | Fused pipeline — 46–78× speedup | **Complete** |
| 1d | 3-way benchmark + double-buffered shaders | **Complete** |

#### 3-Way Benchmark Highlights (Phase 1d)

| Scale | Py(1t) | CPU | GPU | GPU/CPU |
|-------|--------|-----|-----|---------|
| MLP large (3.1M FLOPs) | 3.0 ms | **2.7 ms** | **178 µs** | 15× |
| TF medium (103M FLOPs) | 59 ms | **15.1 ms** | **566 µs** | 27× |
| TF xlarge (6.6B FLOPs) | 232 ms | 1.42 s | **17.8 ms** | **80×** |

Correctness: max diff 1.49e-8 (MLP), 1.10e-6 (Transformer) — same WGSL,
same math, both backends.

---

### The Isomorphism Theorem

All neural architectures decompose into compositions of six fundamental primitives:

1. **GEMM** (matrix multiply) — 60–90% of all FLOPs
2. **Attention** (scaled dot-product) — learned routing
3. **Normalization** (LN/BN/RMS) — scale stabilization
4. **Nonlinearity** (ReLU/GELU/SiLU) — feature carving
5. **Reduction** (sum/mean/max) — aggregation
6. **Gating** (sigmoid × value) — information filtering

A single engine optimizing these 6 ops in WGSL serves every domain:
language (llama.cpp), protein (OpenFold), vision (ViT), physics (hotSpring),
time series (weather), and quantized deployment.

---

### BarraCUDA Shader Evolution

The same pattern hotSpring demonstrated for nuclear physics — Python control,
then Rust/WGSL evolution — applies to ML inference:

| Stage | What Happened | Result |
|-------|---------------|--------|
| Python control | NumPy/PyTorch baselines for all 10 experiments | 75/75 PASS |
| BarraCUDA validation | 242 checks across 10 modules (CPU + GPU) | 242/242 PASS |
| Fused pipeline | Single-encoder dispatch, eliminate per-op overhead | **46–78× over per-op** |
| BLAS-evolved CPU shader | 32×32 tiles, vec4, 8×4 micro-kernel, k-unroll | CPU beats Py at 3M+ FLOPs |
| Double-buffered GPU shader | Load/compute overlap, 2×2 micro-kernel | **10–12% faster at scale** |
| 4-tier router | DeviceCapabilities-driven matmul selection | Best kernel per dispatch |

See [BARRACUDA_EVOLUTION.md](BARRACUDA_EVOLUTION.md) for the full technical narrative.

---

### Cross-Spring Connection

| Spring | Provides | neuralSpring Uses |
|--------|----------|-------------------|
| airSpring | FAO-56 ET₀ model | Surrogate target, real weather data |
| groundSpring | Noise labels, uncertainty | Training robustness, domain gap quantification |
| hotSpring | Physics surrogates (RBF), BarraCUDA patterns | Shader evolution methodology, benchmark patterns |
| wetSpring | Taxonomy pipelines | Future: learned classifiers, HMM for metagenomics |

---

### Research Questions Answered

1. **Can neural surrogates replace equation chains?** Yes — MLP for FAO-56 at R²>0.999
2. **Is self-attention correct from scratch?** Yes — NumPy matches PyTorch to <1e-10
3. **Can LSTM learn weather patterns?** Yes — R²≈0.93, NSE=0.849 on real ERA5
4. **Does transfer learning work across climates?** Yes — 200 NM samples recover domain gap
5. **Are architectures isomorphic?** Yes — 6 primitives, all in BarraCUDA
6. **Can PINNs solve PDEs?** Yes — Burgers' equation to 5.1% L2 error
7. **Can operators be learned?** Yes — DeepONet to 1.2% L2 error
8. **Does quantization preserve accuracy?** Yes — INT8: 0.017% loss, INT4: 0.79%
9. **Can WGSL beat Python for ML?** Yes — CPU 3.9× faster at 103M, GPU 104× faster
10. **Does the hotSpring progression hold?** Yes — GPU < CPU < Python at crossover

---

### Reproduction

```bash
# Phase 0/0+ Python baselines (75/75)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh

# Phase 1 Rust validation (285/285)
cargo test
make validate

# 3-way benchmark (Python vs CPU vs GPU)
cargo run --release --bin bench_scaling
```

---

### Next Phase: Faculty-Driven Paper Candidates

Three professors from the master's program (Dolson, Liu, Bazavov) and one from
undergrad (Waters) provide the next wave of reproduction targets — moving from
"validate ML primitives" to "apply ML to real science."

**Priority targets**:
1. Iram, Dolson et al. (2020) — counterdiabatic driving (Nature Physics)
2. Dolson et al. (2019) — MODES open-ended evolution metrics
3. Liu et al. (2014) — PhyloNet-HMM genomic inference
4. Bruger & Waters (2018) — quorum sensing game theory
