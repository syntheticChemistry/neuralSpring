// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated linear algebra: matmul, transpose, norms, commutator.

#![expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "GPU linalg converts f64→f32 for hardware; tensor variables differ by operation suffix"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

/// GPU matrix multiplication C = A × B for n×n matrices.
///
/// Replaces `spectral_commutativity::mat_mul` (triple-nested CPU loop).
/// Uses `Tensor::matmul` which dispatches through `BarraCUDA`'s 4-tier
/// kernel router (`Naive`/`Tiled16`/`CpuTiled32`/`GpuEvolved32`).
///
/// # Errors
///
/// Returns an error if GPU tensor creation or matmul fails.
pub fn mat_mul_gpu(
    a: &[f64],
    b: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("mat_mul_gpu A upload: {e}"))?;
    let b_t = Tensor::from_data(&b_f32, vec![n, n], device.clone())
        .map_err(|e| format!("mat_mul_gpu B upload: {e}"))?;

    let c_t = a_t
        .matmul(&b_t)
        .map_err(|e| format!("mat_mul_gpu matmul: {e}"))?;

    let c_f32 = c_t
        .to_vec()
        .map_err(|e| format!("mat_mul_gpu readback: {e}"))?;

    Ok(c_f32.into_iter().map(f64::from).collect())
}

/// GPU Frobenius norm: sqrt(sum of squares).
///
/// Replaces `spectral_commutativity::frobenius_norm` (CPU `.iter().sum()`).
/// Uses `Tensor::norm` (L2 norm reduction).
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn frobenius_norm_gpu(a: &[f64], device: &Arc<WgpuDevice>) -> Result<f64, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let n = a_f32.len();

    let a_t = Tensor::from_data(&a_f32, vec![n], device.clone())
        .map_err(|e| format!("frobenius_norm_gpu upload: {e}"))?;

    let norm_t = a_t
        .norm()
        .map_err(|e| format!("frobenius_norm_gpu norm: {e}"))?;

    let result = norm_t
        .to_vec()
        .map_err(|e| format!("frobenius_norm_gpu readback: {e}"))?;

    Ok(f64::from(result[0]))
}

/// GPU transpose for n×n matrix.
///
/// Replaces `spectral_commutativity::transpose` (CPU double loop).
/// Uses `Tensor::transpose`.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn transpose_gpu(a: &[f64], n: usize, device: &Arc<WgpuDevice>) -> Result<Vec<f64>, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("transpose_gpu upload: {e}"))?;

    let t_t = a_t
        .transpose()
        .map_err(|e| format!("transpose_gpu transpose: {e}"))?;

    let t_f32 = t_t
        .to_vec()
        .map_err(|e| format!("transpose_gpu readback: {e}"))?;

    Ok(t_f32.into_iter().map(f64::from).collect())
}

/// GPU commutator `[A,B]` = AB - BA.
///
/// Replaces `spectral_commutativity::commutator`.
/// Two GPU matmuls + elementwise subtract, all on GPU.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn commutator_gpu(
    a: &[f64],
    b: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu A: {e}"))?;
    let b_t = Tensor::from_data(&b_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu B: {e}"))?;

    let b_t2 = Tensor::from_data(&b_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu B2: {e}"))?;
    let a_t2 = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("commutator_gpu A2: {e}"))?;

    let ab = a_t
        .matmul(&b_t)
        .map_err(|e| format!("commutator_gpu AB: {e}"))?;
    let ba = b_t2
        .matmul(&a_t2)
        .map_err(|e| format!("commutator_gpu BA: {e}"))?;

    let diff = ab
        .sub(&ba)
        .map_err(|e| format!("commutator_gpu sub: {e}"))?;

    let out = diff
        .to_vec()
        .map_err(|e| format!("commutator_gpu readback: {e}"))?;

    Ok(out.into_iter().map(f64::from).collect())
}

/// GPU distance to normal: ||A*A - AA*||\_F / (2||A||\_F).
///
/// Replaces `spectral_commutativity::distance_to_normal`.
/// Full computation on GPU: transpose, two matmuls, subtract, norms.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn distance_to_normal_gpu(
    a: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let norm = frobenius_norm_gpu(a, device)?;
    if norm < crate::primitives::LOG_GUARD {
        return Ok(0.0);
    }

    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

    let a_for_at = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_at: {e}"))?;
    let at = a_for_at
        .transpose()
        .map_err(|e| format!("distance_to_normal_gpu transpose: {e}"))?;

    let a_for_ata = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_ata: {e}"))?;
    let ata = at
        .matmul(&a_for_ata)
        .map_err(|e| format!("distance_to_normal_gpu AtA: {e}"))?;

    let a_for_aat = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_aat: {e}"))?;
    let a_for_aat_t = Tensor::from_data(&a_f32, vec![n, n], device.clone())
        .map_err(|e| format!("distance_to_normal_gpu A_aat2: {e}"))?;
    let at_for_aat = a_for_aat_t
        .transpose()
        .map_err(|e| format!("distance_to_normal_gpu At_aat: {e}"))?;
    let aat = a_for_aat
        .matmul(&at_for_aat)
        .map_err(|e| format!("distance_to_normal_gpu AAt: {e}"))?;

    let diff = ata
        .sub(&aat)
        .map_err(|e| format!("distance_to_normal_gpu sub: {e}"))?;

    let diff_norm = diff
        .norm()
        .map_err(|e| format!("distance_to_normal_gpu norm: {e}"))?;

    let result = diff_norm
        .to_vec()
        .map_err(|e| format!("distance_to_normal_gpu readback: {e}"))?;

    Ok(f64::from(result[0]) / (2.0 * norm))
}
