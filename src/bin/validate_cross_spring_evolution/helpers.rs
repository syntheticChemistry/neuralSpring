// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared test vector generation.

pub fn gen_f64_vec(n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * scale).collect()
}
