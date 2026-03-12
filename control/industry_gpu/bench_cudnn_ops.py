# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark: cuDNN ops via PyTorch functional API on CUDA.

Softmax, LayerNorm, GELU, Conv2d+MaxPool2d, Sigmoid — shapes matched
to neuralSpring's validation workloads.

Output tags: CUDNN_SOFTMAX_{N}, CUDNN_LAYERNORM_{M}x{N},
             CUDNN_GELU_{N}, CUDNN_CONV2D, CUDNN_SIGMOID_{N}
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from bench_cuda_common import require_cuda, seed_all, bench_cuda, emit, DEVICE

SOFTMAX_SIZES = [64, 256, 1024, 4096]
LAYERNORM_SHAPES = [(32, 128), (64, 256), (128, 512)]
GELU_SIZES = [1024, 4096, 16384]
SIGMOID_SIZES = [1024, 4096]


def main():
    require_cuda()
    seed_all()

    dev = torch.cuda.get_device_name(0)
    print(f"# cuDNN ops benchmark — {dev}, PyTorch {torch.__version__}")

    # ── Softmax ────────────────────────────────────────────────────────
    for n in SOFTMAX_SIZES:
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        us = bench_cuda(lambda: F.softmax(x, dim=0))
        emit(f"CUDNN_SOFTMAX_{n}", us)

    # Parity: softmax output should sum to 1.0
    x_sm = torch.randn(256, dtype=torch.float32, device=DEVICE)
    sm = F.softmax(x_sm, dim=0)
    print(f"CUDNN_SOFTMAX_256_SUM={sm.sum().item():.6f}")

    # ── LayerNorm ──────────────────────────────────────────────────────
    for m, n in LAYERNORM_SHAPES:
        x = torch.randn(m, n, dtype=torch.float32, device=DEVICE)
        norm_shape = [n]
        us = bench_cuda(lambda: F.layer_norm(x, norm_shape))
        emit(f"CUDNN_LAYERNORM_{m}x{n}", us)

    # Parity: layer-normed rows should have mean≈0, std≈1
    x_ln = torch.randn(32, 128, dtype=torch.float32, device=DEVICE)
    ln = F.layer_norm(x_ln, [128])
    print(f"CUDNN_LAYERNORM_MEAN={ln.mean().item():.6f}")
    print(f"CUDNN_LAYERNORM_STD={ln.std().item():.4f}")

    # ── GELU ───────────────────────────────────────────────────────────
    for n in GELU_SIZES:
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        us = bench_cuda(lambda: F.gelu(x))
        emit(f"CUDNN_GELU_{n}", us)

    # ── Conv2d + MaxPool2d ─────────────────────────────────────────────
    conv_w = torch.randn(64, 3, 5, 5, dtype=torch.float32, device=DEVICE)
    conv_b = torch.randn(64, dtype=torch.float32, device=DEVICE)
    inp = torch.randn(1, 3, 32, 32, dtype=torch.float32, device=DEVICE)

    def conv_pool():
        c = F.conv2d(inp, conv_w, conv_b)
        F.max_pool2d(c, 2)

    us = bench_cuda(conv_pool)
    emit("CUDNN_CONV2D", us)

    # ── Sigmoid ────────────────────────────────────────────────────────
    for n in SIGMOID_SIZES:
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        us = bench_cuda(lambda: torch.sigmoid(x))
        emit(f"CUDNN_SIGMOID_{n}", us)


if __name__ == "__main__":
    main()
