#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PyTorch/CUDA baseline benchmark for playGround inference comparison.

Benchmarks individual operations and full forward passes through GPT-2
using PyTorch as a thin wrapper around cuBLAS/cuDNN, then emits JSON
results for comparison with barraCuda/WGSL timings.

Usage:
    python3 pytorch_baseline.py [--model MODEL_ID] [--device DEVICE]
                                [--warmup N] [--iters N] [--seq-len N]
                                [--json] [--ops-only] [--forward-only]
"""

import argparse
import json
import sys
import time

import torch


def require_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            print("SKIP — no CUDA device", file=sys.stderr)
            sys.exit(0)
        torch.cuda.synchronize()
    return torch.device(device_name)


def time_op(func, device, warmup=50, iters=200):
    """Time a function with proper synchronization. Returns median µs."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        func()
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        func()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter_ns() - t0)

    timings.sort()
    return timings[len(timings) // 2] / 1000.0


def bench_ops(device, hidden=768, seq_len=128, num_heads=12, warmup=50, iters=200):
    """Benchmark individual operations that map to barraCuda TensorSession ops."""
    results = {}
    head_dim = hidden // num_heads

    # --- MatMul ---
    a = torch.randn(seq_len, hidden, device=device)
    b = torch.randn(hidden, hidden, device=device)
    results["matmul"] = {
        "shape": f"[{seq_len},{hidden}] x [{hidden},{hidden}]",
        "median_us": time_op(lambda: a @ b, device, warmup, iters),
    }

    # --- Layer Norm ---
    ln = torch.nn.LayerNorm(hidden).to(device)
    x = torch.randn(seq_len, hidden, device=device)
    results["layer_norm"] = {
        "shape": f"[{seq_len},{hidden}]",
        "median_us": time_op(lambda: ln(x), device, warmup, iters),
    }

    # --- GELU ---
    results["gelu"] = {
        "shape": f"[{seq_len},{hidden}]",
        "median_us": time_op(lambda: torch.nn.functional.gelu(x), device, warmup, iters),
    }

    # --- Softmax ---
    results["softmax"] = {
        "shape": f"[{seq_len},{hidden}]",
        "median_us": time_op(lambda: torch.softmax(x, dim=-1), device, warmup, iters),
    }

    # --- Scaled Dot-Product Attention ---
    q = torch.randn(1, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(1, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(1, num_heads, seq_len, head_dim, device=device)
    results["sdpa"] = {
        "shape": f"B=1, H={num_heads}, S={seq_len}, D={head_dim}",
        "median_us": time_op(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            device, warmup, iters,
        ),
    }

    # --- Embedding lookup ---
    emb = torch.nn.Embedding(50257, hidden).to(device)
    ids = torch.randint(0, 50257, (seq_len,), device=device)
    results["embedding"] = {
        "shape": f"vocab=50257, hidden={hidden}, seq={seq_len}",
        "median_us": time_op(lambda: emb(ids), device, warmup, iters),
    }

    # --- Full FFN block (up + gelu + down) ---
    ffn_up = torch.nn.Linear(hidden, hidden * 4, bias=True).to(device)
    ffn_down = torch.nn.Linear(hidden * 4, hidden, bias=True).to(device)

    def ffn_block():
        h = ffn_up(x)
        h = torch.nn.functional.gelu(h)
        return ffn_down(h)

    results["ffn_block"] = {
        "shape": f"[{seq_len},{hidden}] -> [{seq_len},{hidden*4}] -> [{seq_len},{hidden}]",
        "median_us": time_op(ffn_block, device, warmup, iters),
    }

    return results


def bench_forward(model_id, device, seq_len=128, warmup=10, iters=50):
    """Benchmark full forward pass through a HuggingFace model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return {"error": "transformers not installed"}

    results = {}

    # Load model
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    load_time = time.perf_counter() - t0
    results["load_time_s"] = load_time

    param_count = sum(p.numel() for p in model.parameters())
    results["param_count"] = param_count
    results["param_count_m"] = param_count / 1e6

    # Memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        results["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6

    # Generate input tokens
    input_ids = torch.randint(0, 50257, (1, seq_len), device=device)

    # Benchmark forward pass
    with torch.no_grad():
        median_us = time_op(
            lambda: model(input_ids),
            device, warmup, iters,
        )

    results["forward_pass"] = {
        "seq_len": seq_len,
        "median_us": median_us,
        "median_ms": median_us / 1000.0,
        "throughput_tokens_per_sec": seq_len / (median_us / 1e6),
    }

    if device.type == "cuda":
        results["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6

    return results


def main():
    parser = argparse.ArgumentParser(description="PyTorch/CUDA inference benchmark")
    parser.add_argument("--model", default="openai-community/gpt2", help="HF model ID")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--ops-only", action="store_true", help="Only benchmark ops")
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward")
    args = parser.parse_args()

    device = require_device(args.device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    output = {
        "framework": "pytorch",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        output["gpu_name"] = torch.cuda.get_device_name(0)
        output["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1e6

    if not args.forward_only:
        print(f"Benchmarking ops on {device}...", file=sys.stderr)
        output["ops"] = bench_ops(
            device,
            seq_len=args.seq_len,
            warmup=args.warmup,
            iters=args.iters,
        )

    if not args.ops_only:
        print(f"Benchmarking {args.model} forward on {device}...", file=sys.stderr)
        output["forward"] = bench_forward(
            args.model, device,
            seq_len=args.seq_len,
            warmup=min(args.warmup, 10),
            iters=min(args.iters, 50),
        )

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"PyTorch Benchmark — {device}")
        if "gpu_name" in output:
            print(f"GPU: {output['gpu_name']}")
        print(f"{'='*60}\n")

        if "ops" in output:
            print("Individual Operations:")
            print(f"  {'Operation':<20} {'Shape':<45} {'Median µs':>12}")
            print(f"  {'-'*20} {'-'*45} {'-'*12}")
            for name, data in output["ops"].items():
                print(f"  {name:<20} {data['shape']:<45} {data['median_us']:>10.1f}µs")

        if "forward" in output:
            fwd = output["forward"]
            print(f"\nForward Pass ({args.model}):")
            print(f"  Load time: {fwd.get('load_time_s', 0):.2f}s")
            print(f"  Parameters: {fwd.get('param_count_m', 0):.1f}M")
            if "forward_pass" in fwd:
                fp = fwd["forward_pass"]
                print(f"  Seq length: {fp['seq_len']}")
                print(f"  Median latency: {fp['median_ms']:.2f}ms")
                print(f"  Throughput: {fp['throughput_tokens_per_sec']:.0f} tokens/s")
            if "gpu_memory_peak_mb" in fwd:
                print(f"  Peak GPU memory: {fwd['gpu_memory_peak_mb']:.1f}MB")


if __name__ == "__main__":
    main()
