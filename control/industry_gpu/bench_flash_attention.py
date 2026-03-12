# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark: FlashAttention / cuDNN fused MHA via PyTorch.

PyTorch's `nn.MultiheadAttention` routes to cuDNN fused attention
(or FlashAttention when available) on CUDA.  Three configurations
matching neuralSpring's transformer validation workloads.

Output tags: MHA_{SEQ}x{D}x{H}
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from bench_cuda_common import require_cuda, seed_all, bench_cuda, emit, DEVICE

CONFIGS = [
    (32, 64, 4),    # seq_len, d_model, n_heads
    (64, 128, 8),
    (128, 256, 8),
]


def main():
    require_cuda()
    seed_all()

    dev = torch.cuda.get_device_name(0)
    print(f"# FlashAttention/MHA benchmark — {dev}, PyTorch {torch.__version__}")

    for seq_len, d_model, n_heads in CONFIGS:
        mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dtype=torch.float32,
            device=DEVICE,
        )
        mha.eval()

        q = torch.randn(1, seq_len, d_model, dtype=torch.float32, device=DEVICE)
        k = torch.randn(1, seq_len, d_model, dtype=torch.float32, device=DEVICE)
        v = torch.randn(1, seq_len, d_model, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            us = bench_cuda(lambda: mha(q, k, v, need_weights=False))
        emit(f"MHA_{seq_len}x{d_model}x{n_heads}", us)

    # Parity data: output Frobenius norm for smallest config
    seed_all()
    seq, d, h = CONFIGS[0]
    mha_check = nn.MultiheadAttention(
        embed_dim=d, num_heads=h, batch_first=True,
        dtype=torch.float32, device=DEVICE,
    )
    mha_check.eval()
    q = torch.randn(1, seq, d, dtype=torch.float32, device=DEVICE)
    k = torch.randn(1, seq, d, dtype=torch.float32, device=DEVICE)
    v = torch.randn(1, seq, d, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out, _ = mha_check(q, k, v, need_weights=False)
    fro = torch.norm(out, p="fro").item()
    print(f"MHA_{seq}x{d}x{h}_FRO={fro:.4f}")


if __name__ == "__main__":
    main()
