# SPDX-License-Identifier: AGPL-3.0-or-later
#
# alphafold3_diffusion.py — NumPy reference implementations of diffusion model
# primitives for AlphaFold3 coralForge (nF-03 Phase A).
#
# Implements the core diffusion process from Abramson et al. "Accurate structure
# prediction for all molecules" Nature 630:493-500 (2024):
#
#   - Cosine noise schedule (beta, alpha, alpha_bar)
#   - Linear noise schedule
#   - Forward diffusion (q(x_t | x_0) = N(sqrt(a_bar) x_0, (1-a_bar) I))
#   - DDPM reverse step
#   - DDIM reverse step (deterministic)
#   - SE(3)-equivariant noise (center-of-mass removal)
#   - Pairformer transition FFN (Linear → GELU → Linear)
#   - pLDDT confidence head (Linear → sigmoid)
#
# Reference: Ho et al. "Denoising Diffusion Probabilistic Models" NeurIPS (2020)
#            Song et al. "Denoising Diffusion Implicit Models" ICLR (2021)
#
# Usage: python3 control/coral_forge/alphafold3_diffusion.py
# Output: control/coral_forge/diffusion_baselines.json

import json
import math
import sys
from pathlib import Path

import numpy as np

SEED = 42
N_RES = 12
N_ATOMS = N_RES * 3  # 3 backbone atoms per residue (N, CA, C)
T_STEPS = 50
D_PAIR = 8
D_HIDDEN = 16


# ═══════════════════════════════════════════════════════════════════
# Noise Schedules
# ═══════════════════════════════════════════════════════════════════

def cosine_beta_schedule(T, s=0.008):
    """Cosine schedule from Nichol & Dhariwal (2021)."""
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos((steps / T + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = f / f[0]
    # Clip to prevent singularities
    alpha_bar = np.clip(alpha_bar, 1e-10, 1.0)
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    betas = np.clip(betas, 1e-6, 0.999)
    return betas, alpha_bar[1:]


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear schedule from Ho et al. (2020)."""
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alpha_bar


# ═══════════════════════════════════════════════════════════════════
# Forward Diffusion
# ═══════════════════════════════════════════════════════════════════

def forward_diffusion(x_0, t, alpha_bar, rng):
    """q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)"""
    a_bar_t = alpha_bar[t]
    noise = rng.standard_normal(x_0.shape)
    x_t = math.sqrt(a_bar_t) * x_0 + math.sqrt(1.0 - a_bar_t) * noise
    return x_t, noise


# ═══════════════════════════════════════════════════════════════════
# DDPM Reverse Step
# ═══════════════════════════════════════════════════════════════════

def ddpm_reverse_step(x_t, predicted_noise, t, betas, alpha_bar, rng):
    """
    DDPM: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-a_bar_t) * eps)
                   + sigma_t * z
    """
    beta_t = betas[t]
    alpha_t = 1.0 - beta_t
    a_bar_t = alpha_bar[t]

    # Mean
    coeff_x = 1.0 / math.sqrt(alpha_t)
    coeff_eps = beta_t / math.sqrt(1.0 - a_bar_t)
    mean = coeff_x * (x_t - coeff_eps * predicted_noise)

    if t == 0:
        return mean

    # Variance (simplified: sigma_t^2 = beta_t)
    sigma_t = math.sqrt(beta_t)
    z = rng.standard_normal(x_t.shape)
    return mean + sigma_t * z


# ═══════════════════════════════════════════════════════════════════
# DDIM Reverse Step (deterministic)
# ═══════════════════════════════════════════════════════════════════

def ddim_reverse_step(x_t, predicted_noise, t, alpha_bar):
    """
    DDIM: x_{t-1} = sqrt(a_bar_{t-1}) * predicted_x_0
                   + sqrt(1 - a_bar_{t-1}) * predicted_noise
    where predicted_x_0 = (x_t - sqrt(1-a_bar_t) * eps) / sqrt(a_bar_t)
    """
    a_bar_t = alpha_bar[t]
    a_bar_prev = alpha_bar[t - 1] if t > 0 else 1.0

    # Predict clean data
    pred_x_0 = (x_t - math.sqrt(1.0 - a_bar_t) * predicted_noise) / math.sqrt(a_bar_t)

    # Direction pointing to x_t
    dir_xt = math.sqrt(1.0 - a_bar_prev) * predicted_noise

    x_prev = math.sqrt(a_bar_prev) * pred_x_0 + dir_xt
    return x_prev, pred_x_0


# ═══════════════════════════════════════════════════════════════════
# SE(3) Equivariant Operations
# ═══════════════════════════════════════════════════════════════════

def remove_center_of_mass(coords):
    """Subtract center of mass to ensure translation invariance."""
    com = coords.mean(axis=0)
    return coords - com, com


def apply_random_rotation(coords, rng):
    """Apply a random SO(3) rotation for equivariance testing."""
    # Generate random rotation via QR decomposition of random matrix
    m = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(m)
    # Ensure proper rotation (det=+1)
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return coords @ q.T, q


def se3_equivariant_noise(coords, t, alpha_bar, rng):
    """
    Add noise in a SE(3)-equivariant manner:
    1. Remove center of mass
    2. Add isotropic Gaussian noise
    3. Result is centered (zero COM)
    """
    centered, com = remove_center_of_mass(coords)
    noisy, noise = forward_diffusion(centered, t, alpha_bar, rng)
    noisy_centered, _ = remove_center_of_mass(noisy)
    return noisy_centered, noise, com


# ═══════════════════════════════════════════════════════════════════
# Pairformer Transition FFN
# ═══════════════════════════════════════════════════════════════════

def gelu(x):
    """GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def pair_transition_ffn(pair_repr, w1, b1, w2, b2):
    """
    Linear → GELU → Linear
    pair_repr: [N, N, D_pair]
    w1: [D_pair, D_hidden], b1: [D_hidden]
    w2: [D_hidden, D_pair], b2: [D_pair]
    """
    n = pair_repr.shape[0]
    d_pair = pair_repr.shape[2]
    flat = pair_repr.reshape(-1, d_pair)
    h = gelu(flat @ w1 + b1)
    out = h @ w2 + b2
    return out.reshape(n, n, -1)


# ═══════════════════════════════════════════════════════════════════
# Confidence Heads
# ═══════════════════════════════════════════════════════════════════

def plddt_head(single_repr, w, b):
    """pLDDT: Linear → sigmoid → per-residue confidence [0, 1]."""
    logits = single_repr @ w + b
    return 1.0 / (1.0 + np.exp(-logits))


def pae_head(pair_repr, w, b, n_bins=64):
    """PAE: pair → Linear → softmax over bins → expected distance."""
    n = pair_repr.shape[0]
    d = pair_repr.shape[2]
    flat = pair_repr.reshape(-1, d)
    logits = flat @ w + b  # [N*N, n_bins]
    # Row-wise softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # Expected bin value
    bin_centers = np.linspace(0.0, 31.75, n_bins)
    expected = (probs * bin_centers).sum(axis=1)
    return expected.reshape(n, n), probs.reshape(n, n, n_bins)


# ═══════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════

def run_tests():
    rng = np.random.default_rng(SEED)
    results = {}
    n_pass = 0
    n_fail = 0

    def check(name, condition, detail=""):
        nonlocal n_pass, n_fail
        if condition:
            n_pass += 1
            print(f"  [PASS] {name}")
        else:
            n_fail += 1
            print(f"  [FAIL] {name}: {detail}")
        return condition

    # ─── Noise schedules ────────────────────────────────────────────
    print("\n--- Noise Schedules ---\n")

    cos_betas, cos_abar = cosine_beta_schedule(T_STEPS)
    check("cosine: T betas generated", len(cos_betas) == T_STEPS)
    check("cosine: alpha_bar monotonically decreasing",
          np.all(np.diff(cos_abar) <= 0),
          f"min diff = {np.diff(cos_abar).min():.6e}")
    check("cosine: alpha_bar[0] near 1",
          cos_abar[0] > 0.99,
          f"alpha_bar[0] = {cos_abar[0]:.6f}")
    check("cosine: alpha_bar[-1] near 0",
          cos_abar[-1] < 0.1,
          f"alpha_bar[-1] = {cos_abar[-1]:.6f}")

    lin_betas, lin_abar = linear_beta_schedule(T_STEPS)
    check("linear: alpha_bar monotonically decreasing",
          np.all(np.diff(lin_abar) <= 0))
    check("linear: betas in [1e-4, 0.02]",
          lin_betas[0] >= 1e-4 and lin_betas[-1] <= 0.02 + 1e-10)

    results["cosine_betas"] = cos_betas.tolist()
    results["cosine_alpha_bar"] = cos_abar.tolist()
    results["linear_betas"] = lin_betas.tolist()
    results["linear_alpha_bar"] = lin_abar.tolist()

    # ─── Forward diffusion ──────────────────────────────────────────
    print("\n--- Forward Diffusion ---\n")

    x_0 = rng.standard_normal((N_ATOMS, 3))
    t_mid = T_STEPS // 2
    x_mid, noise_mid = forward_diffusion(x_0, t_mid, cos_abar, np.random.default_rng(SEED + 1))
    check("forward: x_t shape preserved", x_mid.shape == x_0.shape)
    check("forward: x_t differs from x_0",
          np.linalg.norm(x_mid - x_0) > 1.0,
          f"diff = {np.linalg.norm(x_mid - x_0):.4f}")

    # At t=0, noise contribution is sqrt(1 - alpha_bar[0]) ≈ small
    x_0_noisy, _ = forward_diffusion(x_0, 0, cos_abar, np.random.default_rng(SEED + 2))
    noise_scale_t0 = math.sqrt(1.0 - cos_abar[0])
    check("forward: t=0 → noise scale small",
          noise_scale_t0 < 0.15,
          f"noise_scale = {noise_scale_t0:.6e}")
    check("forward: t=0 → x_t ≈ x_0 (within noise budget)",
          np.abs(x_0_noisy - x_0).max() < noise_scale_t0 * 5.0,
          f"max diff = {np.abs(x_0_noisy - x_0).max():.6e}, 5σ budget = {noise_scale_t0 * 5.0:.6e}")

    results["x_0"] = x_0.tolist()
    results["x_mid"] = x_mid.tolist()
    results["noise_mid"] = noise_mid.tolist()
    results["t_mid"] = t_mid

    # ─── DDPM reverse step ──────────────────────────────────────────
    print("\n--- DDPM Reverse Step ---\n")

    x_prev = ddpm_reverse_step(
        x_mid, noise_mid, t_mid, cos_betas, cos_abar,
        np.random.default_rng(SEED + 3)
    )
    check("ddpm: x_prev shape preserved", x_prev.shape == x_mid.shape)
    check("ddpm: x_prev finite", np.all(np.isfinite(x_prev)))
    check("ddpm: x_prev moved toward x_0",
          np.linalg.norm(x_prev - x_0) < np.linalg.norm(x_mid - x_0),
          f"prev_dist={np.linalg.norm(x_prev - x_0):.4f}, mid_dist={np.linalg.norm(x_mid - x_0):.4f}")

    results["ddpm_x_prev"] = x_prev.tolist()

    # ─── DDIM reverse step ──────────────────────────────────────────
    print("\n--- DDIM Reverse Step ---\n")

    ddim_prev, ddim_x0 = ddim_reverse_step(x_mid, noise_mid, t_mid, cos_abar)
    check("ddim: x_prev finite", np.all(np.isfinite(ddim_prev)))
    check("ddim: pred_x_0 finite", np.all(np.isfinite(ddim_x0)))
    check("ddim: deterministic (no randomness)",
          np.allclose(
              ddim_reverse_step(x_mid, noise_mid, t_mid, cos_abar)[0],
              ddim_prev
          ))

    results["ddim_x_prev"] = ddim_prev.tolist()
    results["ddim_pred_x0"] = ddim_x0.tolist()

    # ─── SE(3) equivariance ─────────────────────────────────────────
    print("\n--- SE(3) Equivariant Noise ---\n")

    coords = rng.standard_normal((N_ATOMS, 3))
    noisy, noise, com = se3_equivariant_noise(coords, t_mid, cos_abar, np.random.default_rng(SEED + 4))
    check("se3: output centered (COM ≈ 0)",
          np.linalg.norm(noisy.mean(axis=0)) < 1e-10,
          f"COM norm = {np.linalg.norm(noisy.mean(axis=0)):.6e}")

    # Rotation equivariance: rotate then noise vs noise then rotate should give same result
    rng_a = np.random.default_rng(SEED + 5)
    rng_b = np.random.default_rng(SEED + 5)
    centered_coords, _ = remove_center_of_mass(coords)
    noisy_a, _, _ = se3_equivariant_noise(coords, t_mid, cos_abar, rng_a)
    # Shift coords by arbitrary translation — should give same centered result
    shifted_coords = coords + np.array([100.0, -50.0, 200.0])
    noisy_b, _, _ = se3_equivariant_noise(shifted_coords, t_mid, cos_abar, rng_b)
    check("se3: translation invariance (shifted coords → same centered result)",
          np.allclose(noisy_a, noisy_b, atol=1e-10),
          f"max diff = {np.abs(noisy_a - noisy_b).max():.6e}")

    results["se3_noisy"] = noisy.tolist()
    results["se3_com"] = com.tolist()

    # ─── Pair transition FFN ────────────────────────────────────────
    print("\n--- Pairformer Transition FFN ---\n")

    pair_repr = rng.standard_normal((N_RES, N_RES, D_PAIR))
    w1 = rng.standard_normal((D_PAIR, D_HIDDEN)) * 0.1
    b1 = np.zeros(D_HIDDEN)
    w2 = rng.standard_normal((D_HIDDEN, D_PAIR)) * 0.1
    b2 = np.zeros(D_PAIR)
    ffn_out = pair_transition_ffn(pair_repr, w1, b1, w2, b2)
    check("ffn: output shape", ffn_out.shape == (N_RES, N_RES, D_PAIR))
    check("ffn: output finite", np.all(np.isfinite(ffn_out)))
    check("ffn: output differs from input",
          np.linalg.norm(ffn_out - pair_repr) > 1e-6)

    results["pair_repr"] = pair_repr.tolist()
    results["ffn_w1"] = w1.tolist()
    results["ffn_b1"] = b1.tolist()
    results["ffn_w2"] = w2.tolist()
    results["ffn_b2"] = b2.tolist()
    results["ffn_out"] = ffn_out.tolist()

    # ─── pLDDT confidence head ──────────────────────────────────────
    print("\n--- Confidence: pLDDT ---\n")

    single_repr = rng.standard_normal((N_RES, D_PAIR))
    w_plddt = rng.standard_normal((D_PAIR, 1)) * 0.1
    b_plddt = np.zeros(1)
    plddt = plddt_head(single_repr, w_plddt, b_plddt)
    check("plddt: output shape [N_RES, 1]", plddt.shape == (N_RES, 1))
    check("plddt: values in [0, 1]",
          np.all(plddt >= 0) and np.all(plddt <= 1),
          f"range = [{plddt.min():.4f}, {plddt.max():.4f}]")

    results["plddt"] = plddt.flatten().tolist()
    results["plddt_single"] = single_repr.tolist()
    results["plddt_w"] = w_plddt.tolist()
    results["plddt_b"] = b_plddt.tolist()

    # ─── PAE confidence head ────────────────────────────────────────
    print("\n--- Confidence: PAE ---\n")

    w_pae = rng.standard_normal((D_PAIR, 64)) * 0.1
    b_pae = np.zeros(64)
    pae_expected, pae_probs = pae_head(pair_repr, w_pae, b_pae)
    check("pae: expected shape [N, N]", pae_expected.shape == (N_RES, N_RES))
    check("pae: probs sum to 1 per pair",
          np.allclose(pae_probs.sum(axis=2), 1.0, atol=1e-10))
    check("pae: expected values non-negative",
          np.all(pae_expected >= 0))

    results["pae_expected"] = pae_expected.tolist()
    results["pae_w"] = w_pae.tolist()
    results["pae_b"] = b_pae.tolist()

    # ─── Full diffusion loop (forward + reverse) ────────────────────
    print("\n--- Full Diffusion Loop (T steps) ---\n")

    clean_coords = rng.standard_normal((N_ATOMS, 3)) * 5.0
    clean_centered, _ = remove_center_of_mass(clean_coords)

    # Forward: progressively add noise
    _, full_abar = cosine_beta_schedule(T_STEPS)
    x_T, _ = forward_diffusion(clean_centered, T_STEPS - 1, full_abar, np.random.default_rng(SEED + 10))
    check("loop: x_T is noisy (far from x_0)",
          np.linalg.norm(x_T - clean_centered) > 5.0)

    # DDIM reverse: denoise T steps with perfect noise prediction
    # (oracle — use actual noise as prediction)
    x_curr = x_T.copy()
    ddim_trajectory = [np.linalg.norm(x_curr - clean_centered)]
    for t in range(T_STEPS - 1, 0, -1):
        # Oracle noise prediction (in practice this would be the model output)
        a_bar_t = full_abar[t]
        predicted_eps = (x_curr - math.sqrt(a_bar_t) * clean_centered) / math.sqrt(1.0 - a_bar_t)
        x_curr, _ = ddim_reverse_step(x_curr, predicted_eps, t, full_abar)
        ddim_trajectory.append(np.linalg.norm(x_curr - clean_centered))

    check("loop: DDIM oracle converges to x_0",
          ddim_trajectory[-1] < ddim_trajectory[0] * 0.01,
          f"final/initial = {ddim_trajectory[-1]/ddim_trajectory[0]:.4f}")
    check("loop: trajectory monotonically decreasing (mostly)",
          sum(1 for i in range(1, len(ddim_trajectory))
              if ddim_trajectory[i] < ddim_trajectory[i-1]) > len(ddim_trajectory) * 0.8)

    results["ddim_trajectory"] = ddim_trajectory
    results["clean_centered"] = clean_centered.tolist()

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n=== alphafold3_diffusion: {n_pass}/{n_pass + n_fail} PASS, {n_fail} FAIL ===")

    # Write baselines
    out_path = Path(__file__).parent / "diffusion_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baselines written to {out_path}")

    return n_fail == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
