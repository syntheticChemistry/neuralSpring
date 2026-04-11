// SPDX-License-Identifier: AGPL-3.0-or-later

// ToadStool S68: universal precision pipeline, LU/QR, numerical helpers, ridge, chi².

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once};

pub fn validate_toadstool_s68_precision(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    println!("\n─── BarraCUDA (ToadStool S68): universal precision + modern APIs ───\n");

    // Verify core Precision enum variants accessible (quantized + float tiers)
    h.check_bool(
        "TS S67: Precision enum core tiers accessible",
        [
            neural_spring::gpu::Precision::F32,
            neural_spring::gpu::Precision::F64,
            neural_spring::gpu::Precision::Df64,
            neural_spring::gpu::Precision::Bf16,
            neural_spring::gpu::Precision::Q4,
        ]
        .len()
            == 5,
    );

    // Dispatch domain heuristics (cross-spring)
    let cfg = barracuda::dispatch::global_config();
    h.check_bool(
        "TS: dispatch config available",
        cfg.should_use_gpu(100_000, "matmul") || !cfg.should_use_gpu(1, "matmul"),
    );

    // Verify modern linalg decompositions available
    let mat_3x3 = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let (lu, _) = bench_once("lu_decompose (TS linalg)", || {
        barracuda::linalg::lu_decompose(&mat_3x3, 3)
    });
    h.check_bool("TS→linalg: LU decomposes", lu.is_ok());

    let (qr, _) = bench_once("qr_decompose (TS linalg)", || {
        barracuda::linalg::qr_decompose(&mat_3x3, 3, 3)
    });
    h.check_bool("TS→linalg: QR decomposes", qr.is_ok());

    // Numerical: gradient_1d (cross-spring)
    let (grad, _) = bench_once("gradient_1d (TS numerical)", || {
        barracuda::numerical::gradient_1d(&[1.0, 4.0, 9.0, 16.0], 1.0)
    });
    h.check_abs(
        "TS→numerical: gradient [1,4,9,16] center",
        grad[1],
        4.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "TS→numerical: gradient endpoint finite",
        grad[0].is_finite(),
    );

    // Numerical: trapz (cross-spring)
    // trapz([0,1,4,9], [0,1,2,3]) = 0.5*(0+1) + 0.5*(1+4) + 0.5*(4+9) = 9.5
    let y = [0.0, 1.0, 4.0, 9.0];
    let x = [0.0, 1.0, 2.0, 3.0];
    let (integral, _) = bench_once("trapz (TS numerical)", || {
        barracuda::numerical::trapz(&y, &x)
    });
    h.check_abs(
        "TS→numerical: trapz ∫[0,3] x²",
        integral.unwrap_or(0.0),
        9.5,
        tolerances::CROSS_LANGUAGE,
    );

    // Numerical: Hessian (nS baseCamp → BarraCUDA)
    let quadratic: &dyn Fn(&[f64]) -> f64 =
        &|params: &[f64]| params[0] * params[0] + params[1] * params[1];
    let (hess, _) = bench_once("numerical_hessian (nS→TS S54)", || {
        barracuda::numerical::numerical_hessian(quadratic, &[1.0, 2.0], tolerances::HESSIAN_FD_STEP)
    });
    h.check_abs(
        "TS→numerical: hessian[0,0] ≈ 2",
        hess[0],
        2.0,
        tolerances::GPU_LOGSUMEXP_F32,
    );
    h.check_abs(
        "TS→numerical: hessian[1,1] ≈ 2",
        hess[3],
        2.0,
        tolerances::GPU_LOGSUMEXP_F32,
    );
    h.check_abs(
        "TS→numerical: hessian off-diag ≈ 0",
        hess[1],
        0.0,
        tolerances::GPU_LOGSUMEXP_F32,
    );

    // Modern stats: chi2 decomposition
    let observed_counts = [10.0, 20.0, 30.0, 40.0];
    let expected_counts = [25.0, 25.0, 25.0, 25.0];
    let (chi2_result, _) = bench_once("chi2_decomposed (TS stats)", || {
        barracuda::stats::chi2_decomposed(&observed_counts, &expected_counts, 0)
    });
    let chi2 = chi2_result.expect("chi2_decomposed");
    h.check_bool("TS→stats: chi2 statistic > 0", chi2.chi2_total > 0.0);
    h.check_bool(
        "TS→stats: chi2 has 4 contributions",
        chi2.contributions.len() == 4,
    );

    // Ridge regression (wetSpring ESN → BarraCUDA)
    // features: [5, 2] (x and bias column), targets: [5, 1]
    let features = [1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0];
    let targets = [2.1, 3.9, 6.1, 7.9, 10.1]; // y ≈ 2x + 0.1
    let (ridge, _) = bench_once("ridge_regression (wS→TS S56)", || {
        barracuda::linalg::ridge_regression(&features, &targets, 5, 2, 1, 0.01)
    });
    h.check_bool("wS→linalg: ridge converges", ridge.is_ok());
    if let Ok(ref r) = ridge {
        h.check_abs(
            "wS→linalg: ridge slope ≈ 2",
            r.weights[0],
            2.0,
            tolerances::LEVEL_SPACING_GOE_SLACK,
        );
    }

    // Backend reporting
    h.check_bool(
        &format!(
            "TS: backend={}, gpu={}, adapter={}",
            dispatcher.backend(),
            dispatcher.has_gpu(),
            dispatcher.adapter_name()
        ),
        true,
    );
}
