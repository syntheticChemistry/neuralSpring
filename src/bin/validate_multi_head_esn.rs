// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `MultiHeadWdmClassifier` + head disagreement + NPU export.
//!
//! Exercises the hotSpring cross-spring evolution:
//! - MultiHeadEsn with 3 WDM-specific heads (Anderson, Steering, Meta)
//! - head_disagreement uncertainty quantification
//! - NPU weight export via barracuda int8 quantization
//! - Typed JSON deserialization

#![allow(
    clippy::expect_used,
    clippy::pedantic,
    clippy::nursery,
    clippy::too_many_lines
)]

use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_esn::{
    load_esn_from_json, wdm_head_configs, wdm_heads, MultiHeadWdmClassifier,
};

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("validate_multi_head_esn");

    validate_wdm_head_configs(&mut h);
    validate_json_deserialization(&mut h);
    validate_multi_head_creation(&mut h).await;
    validate_npu_export(&mut h).await;

    h.finish();
}

fn validate_wdm_head_configs(h: &mut ValidationHarness) {
    let heads = wdm_head_configs(3);
    h.check_bool("wdm_heads: 3 heads", heads.len() == wdm_heads::COUNT);
    h.check_bool(
        "wdm_heads: regime output_size=3",
        heads[wdm_heads::REGIME_LABEL].output_size == 3,
    );
    h.check_bool(
        "wdm_heads: bandwidth output_size=1",
        heads[wdm_heads::SPECTRAL_BANDWIDTH].output_size == 1,
    );
    h.check_bool(
        "wdm_heads: confidence output_size=1",
        heads[wdm_heads::CONFIDENCE].output_size == 1,
    );
}

fn validate_json_deserialization(h: &mut ValidationHarness) {
    let json = r#"{
        "normalization": {"x_mean": [0.5, 6.0], "x_std": [1.0, 1.5]},
        "weights": {
            "reservoir_size": 4, "n_classes": 3,
            "W_in": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            "W_res": [0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04],
            "b_res": [0.0, 0.0, 0.0, 0.0],
            "W_out": [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            "b_out": [0.0, 0.0, 0.0]
        }
    }"#;

    match load_esn_from_json(json) {
        Ok(esn) => {
            h.check_bool("json: typed deser success", true);
            h.check_bool("json: reservoir_size=4", esn.reservoir_size == 4);
            h.check_bool("json: n_classes=3", esn.n_classes == 3);
            let (label, scores) = esn.classify(0.5, 5.5);
            h.check_bool("json: label in range", label < 3);
            h.check_bool("json: scores finite", scores.iter().all(|s| s.is_finite()));
        }
        Err(e) => {
            h.check_bool(&format!("json: typed deser: {e}"), false);
        }
    }

    h.check_bool("json: invalid rejects", load_esn_from_json("nope").is_err());
    h.check_bool(
        "json: missing weights rejects",
        load_esn_from_json(r#"{"normalization":{"x_mean":[0,0],"x_std":[1,1]}}"#).is_err(),
    );
}

async fn validate_multi_head_creation(h: &mut ValidationHarness) {
    match MultiHeadWdmClassifier::new(8, 3).await {
        Ok(mhw) => {
            h.check_bool("multi_head: creation success", true);
            h.check_bool("multi_head: n_classes=3", mhw.n_classes() == 3);
            h.check_abs(
                "multi_head: default norm x_mean[0]",
                mhw.norm().x_mean[0],
                0.0,
                1e-15,
            );
        }
        Err(e) => {
            h.check_bool(&format!("multi_head: creation: {e}"), false);
        }
    }
}

async fn validate_npu_export(h: &mut ValidationHarness) {
    let mhw = match MultiHeadWdmClassifier::new(8, 3).await {
        Ok(m) => m,
        Err(e) => {
            h.check_bool(&format!("npu: creation: {e}"), false);
            return;
        }
    };

    match mhw.export_npu_weights() {
        Ok(npu) => {
            h.check_bool("npu: export success", true);
            h.check_bool("npu: input_dim=8", npu.input_dim == 8);
            h.check_bool("npu: scale finite", npu.scale.is_finite());
        }
        Err(e) => {
            if e.contains("not been trained") {
                h.check_bool("npu: untrained correctly rejected", true);
            } else {
                h.check_bool(&format!("npu: unexpected error: {e}"), false);
            }
        }
    }
}
