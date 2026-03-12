# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark: cuBLAS GEMM via PyTorch torch.mm on CUDA.

SGEMM (f32) at 6 scales + DGEMM (f64) at 3 scales.
PyTorch routes torch.mm to cuBLAS on CUDA, so these timings
directly measure the vendor library.

Output tags: CUBLAS_SGEMM_{N}, CUBLAS_DGEMM_{N}
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from bench_cuda_common import require_cuda, seed_all, bench_cuda, emit, DEVICE

SGEMM_SCALES = [64, 128, 256, 512, 1024, 2048]
DGEMM_SCALES = [64, 256, 1024]


def make_matrices(n: int, dtype: torch.dtype):
    """Create two deterministic n×n matrices on CUDA."""
    a = torch.randn(n, n, dtype=dtype, device=DEVICE)
    b = torch.randn(n, n, dtype=dtype, device=DEVICE)
    return a, b


def main():
    require_cuda()
    seed_all()

    dev = torch.cuda.get_device_name(0)
    print(f"# cuBLAS GEMM benchmark — {dev}, PyTorch {torch.__version__}")

    for n in SGEMM_SCALES:
        a, b = make_matrices(n, torch.float32)
        us = bench_cuda(lambda: torch.mm(a, b))
        emit(f"CUBLAS_SGEMM_{n}", us)

    for n in DGEMM_SCALES:
        a, b = make_matrices(n, torch.float64)
        us = bench_cuda(lambda: torch.mm(a, b))
        emit(f"CUBLAS_DGEMM_{n}", us)

    # Parity data: output actual matrix product for the smallest scale
    # so the Rust side can compare numerical results.
    seed_all()
    a32 = torch.randn(64, 64, dtype=torch.float32, device=DEVICE)
    b32 = torch.randn(64, 64, dtype=torch.float32, device=DEVICE)
    c32 = torch.mm(a32, b32)
    fro = torch.norm(c32, p="fro").item()
    print(f"CUBLAS_SGEMM_64_FRO={fro:.6f}")

    seed_all()
    a64 = torch.randn(64, 64, dtype=torch.float64, device=DEVICE)
    b64 = torch.randn(64, 64, dtype=torch.float64, device=DEVICE)
    c64 = torch.mm(a64, b64)
    fro64 = torch.norm(c64, p="fro").item()
    print(f"CUBLAS_DGEMM_64_FRO={fro64:.6f}")


if __name__ == "__main__":
    main()
