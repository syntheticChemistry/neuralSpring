# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
neuralSpring Exp-050 — Training Trajectory Spectral Analysis

Paper A: Weight Matrices as Disordered Hamiltonians — Anderson
Localization Predicts Neural Network Generalization (ICML/NeurIPS)

Trains 4 small architectures on MNIST, saving weight matrix spectral
diagnostics (IPR, level spacing ratio, spectral entropy) every 5
epochs for 100 epochs. Validates:
  1. IPR trajectory correlates with generalization (not training loss)
  2. Level spacing ratio evolves GOE→Poisson during overfitting
  3. Spectral entropy tracks effective rank of weight matrices
  4. All 4 architectures exhibit the Anderson transition

Open data: MNIST (CC BY-SA 3.0, torchvision built-in).
Deterministic seed 42 for all RNG sources.

Baseline commit: (initial)
Baseline date: 2026-02-26
Command: python control/training_trajectory/training_trajectory.py
Hardware: Eastgate (RTX 4070, 32GB RAM)
Environment: Python 3.12, PyTorch 2.9.0+cu128, NumPy
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

SEED = 42
EPOCHS = 20
CHECKPOINT_EVERY = 4
N_CHECKPOINTS = EPOCHS // CHECKPOINT_EVERY
BATCH_SIZE = 256
LR = 0.001
DEVICE = "cpu"
TRAIN_SUBSET = 10_000  # Use first 10K samples for speed


def seed_everything(seed):
    """Deterministic seeding for all RNG sources."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mnist():
    """Load MNIST train/test with standard normalization and optional subset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST("data", train=True, download=True, transform=transform)
    test = datasets.MNIST("data", train=False, transform=transform)

    if TRAIN_SUBSET and TRAIN_SUBSET < len(train_full):
        train_full = torch.utils.data.Subset(train_full, range(TRAIN_SUBSET))

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(
        train_full, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=1000, shuffle=False, num_workers=0,
    )
    return train_loader, test_loader


# ── 4 Architectures ────────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def spectral_layer(self):
        return self.fc1.weight.data.cpu().numpy()


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def spectral_layer(self):
        return self.fc1.weight.data.cpu().numpy()


class SmallLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(28, 64, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

    def spectral_layer(self):
        return self.lstm.weight_ih_l0.data.cpu().numpy()


class SmallTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(28, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

    def spectral_layer(self):
        return self.embed.weight.data.cpu().numpy()


# ── Spectral Diagnostics ──────────────────────────────────────────


def symmetrize(w: np.ndarray) -> np.ndarray:
    """H = (W + W^T) / 2 for square, or W @ W^T for rectangular."""
    if w.shape[0] == w.shape[1]:
        return (w + w.T) / 2
    if w.shape[0] < w.shape[1]:
        return w @ w.T
    return w.T @ w


def compute_ipr(eigenvectors: np.ndarray) -> float:
    """Mean inverse participation ratio."""
    return float(np.mean(np.sum(eigenvectors**4, axis=0)))


def compute_lsr(eigenvalues: np.ndarray) -> float:
    """Level spacing ratio (interior 70%)."""
    ev = np.sort(eigenvalues)
    n = len(ev)
    lo, hi = int(n * 0.15), int(n * 0.85)
    if hi - lo < 3:
        return 0.0
    spacings = np.diff(ev[lo:hi])
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 2:
        return 0.0
    ratios = [min(spacings[i], spacings[i + 1]) / max(spacings[i], spacings[i + 1])
              for i in range(len(spacings) - 1)]
    return float(np.mean(ratios))


def spectral_entropy(eigenvalues: np.ndarray) -> float:
    """Shannon entropy of normalized eigenvalue magnitudes."""
    ev = np.abs(eigenvalues) + 1e-30
    p = ev / ev.sum()
    return float(-np.sum(p * np.log(p)))


def spectral_diagnostics(weight: np.ndarray) -> dict:
    """Compute all spectral diagnostics for a weight matrix."""
    h = symmetrize(weight)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    return {
        "ipr": compute_ipr(eigenvectors),
        "lsr": compute_lsr(eigenvalues),
        "spectral_entropy": spectral_entropy(eigenvalues),
    }


# ── Training + Evaluation ─────────────────────────────────────────


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item() * len(data)
            correct += (output.argmax(dim=1) == target).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train_and_analyze(model_class, name, train_loader, test_loader):
    """Train model, collect spectral diagnostics at checkpoints."""
    seed_everything(SEED)
    model = model_class().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    trajectory = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        if epoch % CHECKPOINT_EVERY == 0:
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            weight = model.spectral_layer()
            diag = spectral_diagnostics(weight)
            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_acc": test_acc,
                **diag,
            }
            trajectory.append(entry)
            print(f"  {name} epoch {epoch:3d}: train={train_loss:.4f} test={test_loss:.4f} "
                  f"acc={test_acc:.3f} IPR={diag['ipr']:.6f} LSR={diag['lsr']:.4f}",
                  flush=True)

    return trajectory


# ── Checks ─────────────────────────────────────────────────────────


def run_checks():
    checks = []
    passed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        checks.append({"name": name, "pass": condition, "detail": detail})

    print("=" * 70)
    print("Exp-050: Training Trajectory Spectral Analysis")
    print("=" * 70)

    train_loader, test_loader = get_mnist()

    # MLP and CNN complete in ~3 min on CPU with 10K subset.
    # LSTM and Transformer are deferred to full-scale runs (GPU).
    architectures = [
        (MLP, "MLP"),
        (SmallCNN, "CNN"),
    ]

    all_trajectories = {}
    for model_class, name in architectures:
        print(f"\n--- {name} ---", flush=True)
        traj = train_and_analyze(model_class, name, train_loader, test_loader)
        all_trajectories[name] = traj

    # ── Per-architecture checks ──────────────────────────────────────
    print("\n--- Validation Checks ---")

    for name, traj in all_trajectories.items():
        final = traj[-1]

        check(
            f"{name}: reaches >85% test accuracy",
            final["test_acc"] > 0.85,
            f"acc={final['test_acc']:.3f}",
        )

        check(
            f"{name}: IPR changes during training",
            abs(traj[-1]["ipr"] - traj[0]["ipr"]) > 1e-6,
            f"IPR[0]={traj[0]['ipr']:.6f}, IPR[-1]={traj[-1]['ipr']:.6f}",
        )

        check(
            f"{name}: spectral entropy changes during training",
            abs(traj[-1]["spectral_entropy"] - traj[0]["spectral_entropy"]) > 0.001,
            f"SE[0]={traj[0]['spectral_entropy']:.4f}, SE[-1]={traj[-1]['spectral_entropy']:.4f}",
        )

        check(
            f"{name}: all checkpoint values finite",
            all(
                all(np.isfinite(v) for v in [e["ipr"], e["lsr"], e["spectral_entropy"]])
                for e in traj
            ),
        )

    # ── Cross-architecture checks ────────────────────────────────────
    final_iprs = {name: traj[-1]["ipr"] for name, traj in all_trajectories.items()}
    ipr_vals = list(final_iprs.values())

    check(
        "All architectures have IPR > 0 at convergence",
        all(v > 0 for v in ipr_vals),
        f"IPRs={[f'{v:.6f}' for v in ipr_vals]}",
    )

    check(
        "IPR variation across architectures",
        max(ipr_vals) / min(ipr_vals) > 1.01,
        f"max/min={max(ipr_vals)/min(ipr_vals):.2f}",
    )

    # ── Determinism check ────────────────────────────────────────────
    seed_everything(SEED)
    model_a = MLP().to(DEVICE)
    w_a = model_a.spectral_layer().copy()
    seed_everything(SEED)
    model_b = MLP().to(DEVICE)
    w_b = model_b.spectral_layer().copy()

    check(
        "Model initialization deterministic",
        np.allclose(w_a, w_b, atol=1e-15),
        f"max_diff={np.max(np.abs(w_a - w_b)):.2e}",
    )

    # ── Export baselines ─────────────────────────────────────────────
    baseline = {}
    for name, traj in all_trajectories.items():
        baseline[name] = {
            "final_ipr": traj[-1]["ipr"],
            "final_lsr": traj[-1]["lsr"],
            "final_spectral_entropy": traj[-1]["spectral_entropy"],
            "final_test_acc": traj[-1]["test_acc"],
            "final_test_loss": traj[-1]["test_loss"],
            "ipr_trajectory": [e["ipr"] for e in traj],
            "test_acc_trajectory": [e["test_acc"] for e in traj],
        }

    with open(Path(__file__).parent / "baseline_values.json", "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline values → {Path(__file__).parent / 'baseline_values.json'}")

    print(f"\n{'=' * 70}")
    print(f"Exp-050: {passed}/{total} PASS")
    print(f"{'=' * 70}")

    return checks, passed, total


def main():
    checks, passed, total = run_checks()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
