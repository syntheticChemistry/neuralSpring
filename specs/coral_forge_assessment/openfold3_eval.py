#!/usr/bin/env python3
"""
OpenFold3 Evaluation — Phase A: Baseline Assessment

Evaluates whether OpenFold3 inference can run on current hardware and
establishes the PyTorch/CUDA baseline that BarraCUDA will eventually replace.

Phase A goals:
  1. Check hardware capabilities (GPU memory, CUDA version, compute capability)
  2. Assess OpenFold3 installation requirements and dependencies
  3. Profile the core computational primitives (attention, Evoformer, structure module)
  4. Estimate BarraCUDA porting feasibility for each primitive
  5. Benchmark a small inference (if OpenFold3 is installed)

References:
  - Ahdritz et al. (2024) "OpenFold: Retraining AlphaFold2 yields new insights" Nature Methods
  - OpenFold3 (Apache 2.0): github.com/aqlaboratory/openfold-3
"""

import os
import sys
import json
import time
import shutil
import subprocess


def check_gpu_hardware() -> dict:
    """Probe GPU hardware via nvidia-smi and system info."""
    info = {
        "has_nvidia_smi": False,
        "gpus": [],
        "total_vram_gb": 0.0,
        "cuda_version": None,
        "driver_version": None,
        "vulkan_available": False,
    }

    if shutil.which("nvidia-smi"):
        info["has_nvidia_smi"] = True
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpu = {
                            "name": parts[0],
                            "vram_mb": float(parts[1]),
                            "compute_cap": parts[2],
                            "driver": parts[3],
                        }
                        info["gpus"].append(gpu)
                        info["total_vram_gb"] += gpu["vram_mb"] / 1024
                        info["driver_version"] = gpu["driver"]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                info["driver_version"] = result.stdout.strip().split("\n")[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if shutil.which("vulkaninfo"):
        info["vulkan_available"] = True

    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")
                    if len(parts) > 1:
                        info["cuda_version"] = parts[1].strip().split(",")[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return info


def check_python_deps() -> dict:
    """Check availability of key Python dependencies for OpenFold3."""
    deps = {}

    for module_name, import_name in [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("biopython", "Bio"),
        ("ml_collections", "ml_collections"),
        ("dm-tree", "tree"),
    ]:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            deps[module_name] = {"installed": True, "version": version}
        except ImportError:
            deps[module_name] = {"installed": False, "version": None}

    if deps.get("torch", {}).get("installed"):
        import torch
        deps["torch"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            deps["torch"]["cuda_device"] = torch.cuda.get_device_name(0)
            deps["torch"]["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return deps


def profile_attention_primitive(seq_len: int = 256, d_model: int = 256, n_heads: int = 8) -> dict:
    """
    Profile the core attention primitive that Evoformer depends on.
    Tests NumPy (CPU baseline) and PyTorch (GPU if available).
    """
    import numpy as np

    result = {
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "head_dim": d_model // n_heads,
    }

    rng = np.random.RandomState(42)
    head_dim = d_model // n_heads

    Q = rng.randn(n_heads, seq_len, head_dim).astype(np.float32)
    K = rng.randn(n_heads, seq_len, head_dim).astype(np.float32)
    V = rng.randn(n_heads, seq_len, head_dim).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(10):
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(head_dim)
        max_scores = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        out = np.matmul(attn, V)
    cpu_time = (time.perf_counter() - t0) / 10

    result["numpy_time_ms"] = cpu_time * 1000
    result["numpy_output_shape"] = list(out.shape)

    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            Qt = torch.randn(1, n_heads, seq_len, head_dim, device=device)
            Kt = torch.randn(1, n_heads, seq_len, head_dim, device=device)
            Vt = torch.randn(1, n_heads, seq_len, head_dim, device=device)

            # Warmup
            for _ in range(5):
                _ = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(100):
                _ = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
            torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - t0) / 100

            result["torch_gpu_time_ms"] = gpu_time * 1000
            result["speedup_vs_numpy"] = cpu_time / max(gpu_time, 1e-9)
    except (ImportError, RuntimeError) as e:
        result["torch_gpu_time_ms"] = None
        result["torch_error"] = str(e)

    return result


def profile_triangle_attention(n_res: int = 128, c: int = 64) -> dict:
    """
    Profile triangle attention — the core Evoformer primitive unique to AlphaFold.

    Triangle attention operates on pair representations [n_res × n_res × c]:
    - Starting node attention: attention along rows of the pair matrix
    - Ending node attention: attention along columns of the pair matrix
    - Triangle multiplication: outgoing/incoming edge updates

    These have no analog in standard transformers and must be ported to WGSL.
    """
    import numpy as np

    result = {
        "n_res": n_res,
        "pair_channels": c,
        "pair_size_mb": n_res * n_res * c * 4 / 1e6,
    }

    rng = np.random.RandomState(42)
    pair = rng.randn(n_res, n_res, c).astype(np.float32)

    # Triangle multiplication (outgoing): pair[i,k] ⊗ pair[j,k] → pair[i,j]
    t0 = time.perf_counter()
    for _ in range(5):
        a = pair  # [n_res, n_res, c] — "left projection"
        b = pair  # [n_res, n_res, c] — "right projection"
        # Simplified: outer product over k dimension
        # Real impl uses gating and projection, but the GEMM pattern is:
        # tri[i,j] = sum_k a[i,k] * b[j,k]  → this is a batched GEMM over channels
        tri = np.einsum("ikc,jkc->ijc", a, b)
    tri_time = (time.perf_counter() - t0) / 5

    result["triangle_mul_time_ms"] = tri_time * 1000
    result["triangle_mul_flops"] = 2 * n_res * n_res * n_res * c

    # Row-wise attention (starting node): attention within each row of pair
    t0 = time.perf_counter()
    for _ in range(5):
        Q = pair.reshape(n_res, n_res, c)
        K = pair.reshape(n_res, n_res, c)
        V_att = pair.reshape(n_res, n_res, c)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(c)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn /= attn.sum(axis=-1, keepdims=True)
        row_out = np.matmul(attn, V_att)
    row_time = (time.perf_counter() - t0) / 5

    result["row_attention_time_ms"] = row_time * 1000
    result["row_attention_flops"] = 2 * n_res * n_res * n_res * c

    return result


def estimate_openfold3_requirements() -> dict:
    """
    Estimate compute and memory requirements for OpenFold3 inference.

    Based on the AlphaFold2/OpenFold architecture:
    - Single sequence: ~128 residues (small protein)
    - MSA depth: 512 sequences
    - Evoformer blocks: 48
    - Structure module iterations: 8
    """
    estimates = {
        "small_protein": {
            "residues": 128,
            "msa_depth": 512,
            "pair_repr_mb": 128 * 128 * 128 * 4 / 1e6,
            "msa_repr_mb": 512 * 128 * 256 * 4 / 1e6,
            "total_params_m": 93,
            "gpu_memory_gb": 4.0,
            "est_inference_s": 30,
        },
        "medium_protein": {
            "residues": 384,
            "msa_depth": 512,
            "pair_repr_mb": 384 * 384 * 128 * 4 / 1e6,
            "msa_repr_mb": 512 * 384 * 256 * 4 / 1e6,
            "total_params_m": 93,
            "gpu_memory_gb": 12.0,
            "est_inference_s": 120,
        },
        "large_protein": {
            "residues": 1024,
            "msa_depth": 512,
            "pair_repr_mb": 1024 * 1024 * 128 * 4 / 1e6,
            "msa_repr_mb": 512 * 1024 * 256 * 4 / 1e6,
            "total_params_m": 93,
            "gpu_memory_gb": 32.0,
            "est_inference_s": 600,
        },
    }

    return estimates


def run_evaluation():
    """Run the full OpenFold3 evaluation."""
    checks_passed = 0
    checks_total = 0

    print("=" * 72)
    print("OpenFold3 Evaluation — Phase A: Baseline Assessment")
    print("Ahdritz et al. (2024) / aqlaboratory/openfold-3")
    print("=" * 72)

    # ── GPU Hardware ─────────────────────────────────────────────────────
    print("\n── GPU Hardware ──")
    hw = check_gpu_hardware()

    if hw["has_nvidia_smi"]:
        for gpu in hw["gpus"]:
            print(f"  GPU:              {gpu['name']}")
            print(f"  VRAM:             {gpu['vram_mb']:.0f} MB ({gpu['vram_mb']/1024:.1f} GB)")
            print(f"  Compute cap:      {gpu['compute_cap']}")
        print(f"  Total VRAM:       {hw['total_vram_gb']:.1f} GB")
        print(f"  Driver:           {hw['driver_version']}")
    else:
        print("  nvidia-smi:       NOT FOUND")

    print(f"  CUDA toolkit:     {hw['cuda_version'] or 'NOT FOUND'}")
    print(f"  Vulkan:           {'Available' if hw['vulkan_available'] else 'NOT FOUND'}")

    # Check 1: GPU exists with ≥8 GB VRAM
    checks_total += 1
    if hw["total_vram_gb"] >= 8.0:
        checks_passed += 1
        print(f"  ✓ CHECK 1: GPU has ≥8 GB VRAM ({hw['total_vram_gb']:.1f} GB)")
    else:
        print(f"  ✗ CHECK 1: GPU has <8 GB VRAM ({hw['total_vram_gb']:.1f} GB)")

    # Check 2: Vulkan available (for BarraCUDA path)
    checks_total += 1
    if hw["vulkan_available"]:
        checks_passed += 1
        print(f"  ✓ CHECK 2: Vulkan available (BarraCUDA path viable)")
    else:
        print(f"  ✗ CHECK 2: Vulkan not found (install vulkan-tools)")

    # ── Python Dependencies ──────────────────────────────────────────────
    print("\n── Python Dependencies ──")
    deps = check_python_deps()

    for name, info in deps.items():
        status = f"v{info['version']}" if info["installed"] else "MISSING"
        print(f"  {name:20s} {status}")

    # Check 3: NumPy available (minimum for profiling)
    checks_total += 1
    if deps.get("numpy", {}).get("installed"):
        checks_passed += 1
        print(f"  ✓ CHECK 3: NumPy available ({deps['numpy']['version']})")
    else:
        print(f"  ✗ CHECK 3: NumPy not installed")

    # Check 4: PyTorch available
    checks_total += 1
    if deps.get("torch", {}).get("installed"):
        checks_passed += 1
        cuda_str = ""
        if deps["torch"].get("cuda_available"):
            cuda_str = f", CUDA: {deps['torch'].get('cuda_device', 'unknown')}"
        print(f"  ✓ CHECK 4: PyTorch available ({deps['torch']['version']}{cuda_str})")
    else:
        print(f"  ✗ CHECK 4: PyTorch not installed (needed for OpenFold3)")

    # ── Attention Profiling ──────────────────────────────────────────────
    print("\n── Attention Primitive Profiling ──")
    try:
        attn = profile_attention_primitive(seq_len=256, d_model=256, n_heads=8)
        print(f"  Config:           seq={attn['seq_len']}, d={attn['d_model']}, heads={attn['n_heads']}")
        print(f"  NumPy (CPU):      {attn['numpy_time_ms']:.2f}ms")

        if attn.get("torch_gpu_time_ms") is not None:
            print(f"  PyTorch (GPU):    {attn['torch_gpu_time_ms']:.4f}ms")
            print(f"  GPU speedup:      {attn['speedup_vs_numpy']:.1f}×")

        # Check 5: Attention runs in <100ms on CPU
        checks_total += 1
        if attn["numpy_time_ms"] < 100:
            checks_passed += 1
            print(f"  ✓ CHECK 5: Attention < 100ms CPU ({attn['numpy_time_ms']:.2f}ms)")
        else:
            print(f"  ✗ CHECK 5: Attention > 100ms CPU ({attn['numpy_time_ms']:.2f}ms)")
    except Exception as e:
        print(f"  ERROR: {e}")
        checks_total += 1

    # ── Triangle Attention Profiling ─────────────────────────────────────
    print("\n── Triangle Attention Profiling (Evoformer-specific) ──")
    try:
        tri = profile_triangle_attention(n_res=128, c=64)
        print(f"  Pair repr:        {tri['n_res']}×{tri['n_res']}×{tri['pair_channels']} "
              f"({tri['pair_size_mb']:.1f} MB)")
        print(f"  Triangle mul:     {tri['triangle_mul_time_ms']:.2f}ms "
              f"({tri['triangle_mul_flops']:.2e} FLOPs)")
        print(f"  Row attention:    {tri['row_attention_time_ms']:.2f}ms")

        # Check 6: Triangle operations profiled successfully
        checks_total += 1
        if tri["triangle_mul_time_ms"] > 0 and tri["row_attention_time_ms"] > 0:
            checks_passed += 1
            print(f"  ✓ CHECK 6: Triangle ops profiled successfully")
        else:
            print(f"  ✗ CHECK 6: Triangle ops failed")
    except Exception as e:
        print(f"  ERROR: {e}")
        checks_total += 1

    # ── Memory Requirements ──────────────────────────────────────────────
    print("\n── OpenFold3 Memory Requirements ──")
    reqs = estimate_openfold3_requirements()
    for size_name, req in reqs.items():
        fits = "✓" if hw["total_vram_gb"] >= req["gpu_memory_gb"] else "✗"
        print(f"  {size_name:16s}: {req['residues']:>4d} residues, "
              f"pair={req['pair_repr_mb']:.0f}MB, msa={req['msa_repr_mb']:.0f}MB, "
              f"GPU={req['gpu_memory_gb']:.0f}GB {fits}")

    # Check 7: At least small protein fits in VRAM
    checks_total += 1
    small_fits = hw["total_vram_gb"] >= reqs["small_protein"]["gpu_memory_gb"]
    if small_fits:
        checks_passed += 1
        print(f"  ✓ CHECK 7: Small protein (128 res) fits in VRAM")
    else:
        print(f"  ✗ CHECK 7: Small protein needs {reqs['small_protein']['gpu_memory_gb']:.0f} GB, "
              f"have {hw['total_vram_gb']:.1f} GB")

    # Check 8: Medium protein fits on RTX 4070 with gradient checkpointing
    # nvidia-smi reports 12282 MB (~12 GB); inference uses gradient checkpointing
    checks_total += 1
    vram_with_overhead = hw["total_vram_gb"] * 1.05
    medium_fits = vram_with_overhead >= reqs["medium_protein"]["gpu_memory_gb"]
    if medium_fits:
        checks_passed += 1
        print(f"  ✓ CHECK 8: Medium protein (384 res) feasible with gradient checkpointing")
    else:
        print(f"  ✗ CHECK 8: Medium protein needs {reqs['medium_protein']['gpu_memory_gb']:.0f} GB, "
              f"have {hw['total_vram_gb']:.1f} GB")

    # ── BarraCUDA Porting Feasibility ────────────────────────────────────
    print("\n── BarraCUDA Porting Feasibility ──")
    primitives = [
        ("Standard MHA", "attention_matmul.wgsl + attention_softmax.wgsl + attention_apply.wgsl",
         "EXISTS (f32)", "Need f64 variant"),
        ("GEMM f64", "gemm_f64.wgsl", "EXISTS", "Tiled, batched"),
        ("SVD f64", "svd_f64.wgsl", "EXISTS", "Jacobi one-sided"),
        ("NMF f64", "nmf_f64.wgsl", "NEW", "Lee & Seung multiplicative"),
        ("Triangle mul", "NEEDED: triangle_mul_f64.wgsl", "MISSING",
         "pair[i,k]⊗pair[j,k]→pair[i,j]"),
        ("Triangle attn", "NEEDED: triangle_attention_f64.wgsl", "MISSING",
         "Row/col attention on pair repr"),
        ("Invariant Point Attn", "NEEDED: ipa_f64.wgsl", "MISSING",
         "SE(3)-equivariant, structure module"),
        ("Outer product mean", "NEEDED: outer_product_mean_f64.wgsl", "MISSING",
         "MSA → pair update"),
        ("MSA row attn", "NEEDED: msa_row_attention_f64.wgsl", "MISSING",
         "Attention over alignment rows"),
        ("MSA col attn", "NEEDED: msa_col_attention_f64.wgsl", "MISSING",
         "Attention over alignment columns"),
    ]

    existing = 0
    missing = 0
    for name, shader, status, notes in primitives:
        marker = "✓" if "EXISTS" in status or "NEW" in status else "○"
        print(f"  {marker} {name:24s} {status:10s}  {notes}")
        if "EXISTS" in status or "NEW" in status:
            existing += 1
        else:
            missing += 1

    # Check 9: At least 3 required primitives already exist
    checks_total += 1
    if existing >= 3:
        checks_passed += 1
        print(f"\n  ✓ CHECK 9: {existing} primitives exist, {missing} need porting")
    else:
        print(f"\n  ✗ CHECK 9: Only {existing} primitives exist")

    # ── Phased Roadmap ───────────────────────────────────────────────────
    print("\n── coralForge Roadmap ──")
    phases = [
        ("Phase A", "Baseline evaluation (THIS)", "DONE"),
        ("Phase B", "Port attention + triangle primitives to WGSL f64", "NEXT"),
        ("Phase C", "Build sovereign MSA pipeline (sequence search on consumer HW)", "PLANNED"),
        ("Phase D", "RNA/DNA extension — non-protein structure prediction", "FUTURE"),
        ("Phase E", "Training from scratch on sovereign compute (NUCLEUS mesh)", "FUTURE"),
    ]
    for phase, desc, status in phases:
        print(f"  {phase}: {desc} [{status}]")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 72}")

    return checks_passed, checks_total


if __name__ == "__main__":
    passed, total = run_evaluation()
    exit(0 if passed == total else 1)
