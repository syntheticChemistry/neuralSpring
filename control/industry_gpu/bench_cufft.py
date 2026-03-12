# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark: cuFFT via PyTorch torch.fft on CUDA.

Complex FFT (f32), Real-to-complex RFFT (f32), and f64 FFT at
multiple sizes.  PyTorch delegates to cuFFT on CUDA.

Parity check: FFT of a cosine at freq=5 — the magnitude spectrum
should peak at bin 5.

Output tags: CUFFT_FFT_{N}, CUFFT_RFFT_{N}, CUFFT_FFT_F64_{N}
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import math

import torch
from bench_cuda_common import require_cuda, seed_all, bench_cuda, emit, DEVICE

FFT_SIZES = [256, 1024, 4096, 16384, 65536]
F64_FFT_SIZES = [256, 1024, 4096]


def cosine_signal(n: int, freq: int, dtype: torch.dtype):
    """Generate a cosine signal at the given frequency."""
    t = torch.arange(n, dtype=dtype, device=DEVICE)
    return torch.cos(2 * math.pi * freq * t / n)


def main():
    require_cuda()
    seed_all()

    dev = torch.cuda.get_device_name(0)
    print(f"# cuFFT benchmark — {dev}, PyTorch {torch.__version__}")

    # ── Complex FFT (f32) ──────────────────────────────────────────────
    for n in FFT_SIZES:
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        us = bench_cuda(lambda: torch.fft.fft(x))
        emit(f"CUFFT_FFT_{n}", us)

    # ── Real-to-complex RFFT (f32) ─────────────────────────────────────
    for n in FFT_SIZES:
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        us = bench_cuda(lambda: torch.fft.rfft(x))
        emit(f"CUFFT_RFFT_{n}", us)

    # ── f64 FFT ────────────────────────────────────────────────────────
    for n in F64_FFT_SIZES:
        x = torch.randn(n, dtype=torch.float64, device=DEVICE)
        us = bench_cuda(lambda: torch.fft.fft(x))
        emit(f"CUFFT_FFT_F64_{n}", us)

    # ── Parity data: cosine at freq=5 ─────────────────────────────────
    n = 256
    freq = 5
    sig = cosine_signal(n, freq, torch.float32)
    spectrum = torch.fft.fft(sig)
    magnitudes = torch.abs(spectrum).cpu()
    peak_bin = torch.argmax(magnitudes[:n // 2]).item()
    peak_mag = magnitudes[freq].item()
    print(f"CUFFT_COSINE_PEAK_BIN={peak_bin}")
    print(f"CUFFT_COSINE_PEAK_MAG={peak_mag:.4f}")

    # RFFT parity: DC component of constant signal
    const_sig = torch.ones(256, dtype=torch.float32, device=DEVICE)
    rfft_out = torch.fft.rfft(const_sig)
    dc_re = rfft_out[0].real.item()
    print(f"CUFFT_RFFT_DC_RE={dc_re:.4f}")


if __name__ == "__main__":
    main()
