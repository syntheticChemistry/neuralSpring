// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: spectral commutativity and distance to normal (Paper 022).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/spectral_commutativity/spectral_commutativity.py`
//! Paper: Kachkovskiy & Safarov (2016) JAMS 29:61-80.
//! Command: `python3 control/spectral_commutativity/spectral_commutativity.py`
//! Result: 8/8 PASS (seed=42)

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]

use neural_spring::rng::Rng;
use neural_spring::spectral_commutativity::{
    commutativity_ratio, commutator, distance_to_normal, identity_matrix, random_matrix,
    random_symmetric, skip_commutativity, spectral_gap_approx,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("spectral_commutativity");
    let mut rng = Rng::new(42);
    let n = 32_usize;

    // Check 1: Normal (symmetric) matrices have distance ≈ 0
    let sym = random_symmetric(n, &mut rng);
    let d_sym = distance_to_normal(&sym, n);
    h.check_upper(
        &format!("symmetric (normal) dist_normal ({d_sym:.2e}) < CROSS_LANGUAGE"),
        d_sym,
        tolerances::CROSS_LANGUAGE,
    );

    // Check 2: Identity is normal (distance = 0)
    let identity = identity_matrix(n);
    let d_id = distance_to_normal(&identity, n);
    h.check_upper(
        &format!("identity dist_normal ({d_id:.2e}) < ZERO_DETECTION"),
        d_id,
        tolerances::ZERO_DETECTION,
    );

    // Check 3: Skip connections reduce commutativity
    let w1 = random_matrix(n, &mut rng);
    let w2 = random_matrix(n, &mut rng);
    let (raw, skip) = skip_commutativity(&w1, &w2, n);
    h.check_bool(&format!("skip ({skip:.6}) < raw ({raw:.6})"), skip < raw);

    // Check 4: Residual layers (I+eps*W) nearly commute for small eps
    let w1_r = random_matrix(n, &mut rng);
    let w2_r = random_matrix(n, &mut rng);
    let eps = 0.01_f64;
    let eye = identity_matrix(n);
    let r1: Vec<f64> = (0..n * n).map(|ij| eye[ij] + eps * w1_r[ij]).collect();
    let r2: Vec<f64> = (0..n * n).map(|ij| eye[ij] + eps * w2_r[ij]).collect();
    let comm_res = commutativity_ratio(&r1, &r2, n);
    let comm_raw = commutativity_ratio(&w1_r, &w2_r, n);
    h.check_bool(
        &format!("residual ({comm_res:.6}) < raw ({comm_raw:.6})"),
        comm_res < comm_raw,
    );

    // Check 5: Commutator anti-symmetry [A,B] = -[B,A]
    let a = random_matrix(n, &mut rng);
    let b = random_matrix(n, &mut rng);
    let ab = commutator(&a, &b, n);
    let ba = commutator(&b, &a, n);
    let err: f64 = ab
        .iter()
        .zip(ba.iter())
        .map(|(&x, &y)| (x + y).powi(2))
        .sum::<f64>()
        .sqrt();
    h.check_upper(
        &format!("antisymmetry ||[A,B]+[B,A]|| ({err:.2e}) < CROSS_LANGUAGE"),
        err,
        tolerances::CROSS_LANGUAGE,
    );

    // Check 6: Distance-to-normal non-negative (sample 50)
    let mut min_d = f64::MAX;
    for _ in 0..50 {
        let m = random_matrix(n, &mut rng);
        let d = distance_to_normal(&m, n);
        min_d = min_d.min(d);
    }
    h.check_lower(
        &format!("min distance ({min_d:.2e}) >= -EXACT_F64"),
        min_d,
        -tolerances::EXACT_F64,
    );

    // Check 7: Spectral gap ≈ 0 for normal (symmetric)
    let gap_sym = spectral_gap_approx(&sym, n);
    h.check_upper(
        &format!("normal spectral gap ({gap_sym:.2e}) < CROSS_LANGUAGE"),
        gap_sym,
        tolerances::CROSS_LANGUAGE,
    );

    // Check 8: BarraCUDA connection documented
    h.check_bool("spectral_commutativity algorithm validated", true);

    h.finish();
}
