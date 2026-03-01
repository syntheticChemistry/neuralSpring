#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download pretrained model weights and save as safetensors for nS-01 Paper A.

Models (all public, permissive licenses):
  - ResNet-18  (torchvision, BSD-3-Clause)
  - ResNet-50  (torchvision, BSD-3-Clause)
  - ViT-B/16   (torchvision, BSD-3-Clause)
  - GPT-2 small (HuggingFace, MIT)
  - LeNet-5    (manually defined, tiny)

Usage:
    python scripts/download_pretrained.py

Output: control/weight_spectral/pretrained/<model>.safetensors
"""

import os, sys

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    print("Install deps: pip install torch torchvision safetensors transformers")
    sys.exit(1)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "control", "weight_spectral", "pretrained")
os.makedirs(OUT_DIR, exist_ok=True)


def save_state_dict(name, state_dict):
    path = os.path.join(OUT_DIR, f"{name}.safetensors")
    tensors = {}
    for k, v in state_dict.items():
        if v.ndim >= 2:
            tensors[k] = v.contiguous().float()
    save_file(tensors, path)
    total_params = sum(v.numel() for v in tensors.values())
    print(f"  {name}: {len(tensors)} weight matrices, {total_params:,} params -> {path}")


def download_resnet18():
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    save_state_dict("resnet18", model.state_dict())


def download_resnet50():
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    save_state_dict("resnet50", model.state_dict())


def download_vit_b16():
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    save_state_dict("vit_b_16", model.state_dict())


def download_gpt2():
    try:
        from transformers import GPT2Model
        model = GPT2Model.from_pretrained("gpt2")
        save_state_dict("gpt2", model.state_dict())
    except ImportError:
        print("  gpt2: skipped (install transformers: pip install transformers)")


def create_lenet5():
    import torch.nn as nn
    class LeNet5(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    torch.manual_seed(42)
    model = LeNet5()
    save_state_dict("lenet5", model.state_dict())


if __name__ == "__main__":
    print("Downloading pretrained models for nS-01 Paper A...")
    print()

    print("[1/5] ResNet-18")
    download_resnet18()

    print("[2/5] ResNet-50")
    download_resnet50()

    print("[3/5] ViT-B/16")
    download_vit_b16()

    print("[4/5] GPT-2 (small)")
    download_gpt2()

    print("[5/5] LeNet-5 (random init, seed=42)")
    create_lenet5()

    print()
    print(f"All models saved to {OUT_DIR}")
    print("Run: cargo run --release --bin validate_weight_spectral_real")
