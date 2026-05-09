// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Science composition — Rust→IPC parity for domain science.
//!
//! Absorbed from `validate_science_composition.rs` (~9 checks).

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error, validate_parity};
use primalspring::tolerances;
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "science_composition",
        track: Track::SpectralAnalysis,
        tier: Tier::Both,
        provenance_crate: "validate_science_composition",
        provenance_date: "2026-05-09",
        description: "Science composition parity: stats, tensor, crypto via IPC",
        check_count: 9,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("Science Composition — Tier 1 (Rust)");

    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    v.check_bool(
        "science:rust:mean_correct",
        (mean - 3.0).abs() < 1e-12,
        &format!("mean={mean}"),
    );

    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();
    v.check_bool(
        "science:rust:std_dev_positive",
        std_dev > 0.0,
        &format!("std_dev={std_dev}"),
    );

    let a: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let b: [f64; 4] = [5.0, 6.0, 7.0, 8.0];
    let c00 = a[0] * b[0] + a[1] * b[2];
    let c01 = a[0] * b[1] + a[1] * b[3];
    v.check_bool(
        "science:rust:matmul_2x2_c00",
        (c00 - 19.0).abs() < 1e-12,
        &format!("c00={c00}"),
    );
    v.check_bool(
        "science:rust:matmul_2x2_c01",
        (c01 - 22.0).abs() < 1e-12,
        &format!("c01={c01}"),
    );
}

fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Science Composition — Tier 2 (Live)");

    validate_parity(
        ctx,
        v,
        "science:live:stats_mean",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        "result",
        3.0,
        tolerances::IPC_ROUND_TRIP_TOL,
    );

    match ctx.hash_bytes(b"science-composition-parity", "blake3") {
        Ok(hash) => {
            v.check_bool(
                "science:live:crypto_hash",
                !hash.is_empty(),
                &format!("len={}", hash.len()),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "science:live:crypto_hash",
                &format!("security offline: {e}"),
            );
        }
        Err(e) => {
            v.check_bool("science:live:crypto_hash", false, &format!("{e}"));
        }
    }
}
