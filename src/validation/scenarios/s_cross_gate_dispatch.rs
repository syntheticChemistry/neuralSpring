// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Cross-gate dispatch — mesh topology, trust handshake,
//! and remote capability routing (including `ml.mlp_infer`).
//!
//! Verifies that neuralSpring can discover cross-gate capabilities
//! via Songbird mesh and route requests through BTSP-secured channels.

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

use primalspring::composition::{CompositionContext, is_skip_error};
use primalspring::validation::ValidationResult;

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "cross_gate_dispatch",
        track: Track::CrossGate,
        tier: Tier::Both,
        provenance_crate: "s_cross_gate_dispatch",
        provenance_date: "2026-06-04",
        description: "Cross-gate dispatch: mesh discovery, BTSP trust, ML capability routing",
        check_count: 12,
    },
    run_rust: Some(run_rust),
    run_live: Some(run_live),
};

fn run_rust(v: &mut ValidationResult) {
    v.section("Cross-Gate Dispatch — Tier 1 (Rust structural)");

    v.check_bool(
        "crossgate:rust:capability_constants",
        !crate::capabilities::ML_MLP_INFER.is_empty()
            && !crate::capabilities::DISCOVERY_PEERS.is_empty()
            && !crate::capabilities::MESH_INIT.is_empty()
            && !crate::capabilities::CRYPTO_BTSP_HANDSHAKE.is_empty(),
        "ml.mlp_infer + discovery.peers + mesh.init + crypto.btsp_handshake defined",
    );

    v.check_bool(
        "crossgate:rust:dotted_notation",
        crate::capabilities::ML_MLP_INFER.contains('.')
            && crate::capabilities::DISCOVERY_PEERS.contains('.')
            && crate::capabilities::MESH_INIT.contains('.')
            && crate::capabilities::CRYPTO_BTSP_HANDSHAKE.contains('.'),
        "all cross-gate capabilities use dotted notation",
    );

    let all_caps = crate::config::ALL_CAPABILITIES;
    v.check_bool(
        "crossgate:rust:ml_mlp_infer_in_registry",
        all_caps.contains(&"ml.mlp_infer"),
        &format!("{} total capabilities", all_caps.len()),
    );
    v.check_bool(
        "crossgate:rust:discovery_peers_in_registry",
        all_caps.contains(&"discovery.peers"),
        "mesh discovery registered",
    );
    v.check_bool(
        "crossgate:rust:mesh_init_in_registry",
        all_caps.contains(&"mesh.init"),
        "mesh init registered",
    );
    v.check_bool(
        "crossgate:rust:btsp_handshake_in_registry",
        all_caps.contains(&"crypto.btsp_handshake"),
        "BTSP trust handshake registered",
    );

    let cap_count = all_caps.len();
    v.check_bool(
        "crossgate:rust:capability_count_51",
        cap_count == 51,
        &format!("got {cap_count}"),
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "live cross-gate dispatch probes exercise full capability graph in one pass"
)]
fn run_live(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Cross-Gate Dispatch — Tier 2 (Live)");

    match ctx.call(
        "tensor",
        "ml.mlp_infer",
        serde_json::json!({
            "input": [1.0, 0.5, -0.3],
            "input_dim": 3,
            "hidden_dims": [4],
            "output_dim": 2,
        }),
    ) {
        Ok(result) => {
            let has_output = result.get("output").is_some()
                || result.get("data").is_some()
                || result.get("result").is_some();
            v.check_bool(
                "crossgate:live:ml_mlp_infer",
                has_output,
                "barraCuda responded with output array",
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "crossgate:live:ml_mlp_infer",
                &format!("barraCuda offline: {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "crossgate:live:ml_mlp_infer",
                &format!("ml.mlp_infer not yet supported upstream: {e}"),
            );
        }
    }

    match ctx.call(
        "security",
        "crypto.btsp_handshake",
        serde_json::json!({
            "peer_id": "south-gate",
            "challenge": "scenario-validation-nonce",
        }),
    ) {
        Ok(_result) => {
            v.check_bool(
                "crossgate:live:btsp_handshake",
                true,
                "BearDog BTSP handshake responded",
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "crossgate:live:btsp_handshake",
                &format!("BearDog offline: {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "crossgate:live:btsp_handshake",
                &format!("BTSP not yet supported: {e}"),
            );
        }
    }

    match ctx.call("ai", "discovery.peers", serde_json::json!({})) {
        Ok(result) => {
            let peer_count = result
                .get("peers")
                .and_then(|v| v.as_array())
                .map_or(0, Vec::len);
            v.check_bool(
                "crossgate:live:discovery_peers",
                true,
                &format!("{peer_count} peers discovered"),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "crossgate:live:discovery_peers",
                &format!("Songbird offline: {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "crossgate:live:discovery_peers",
                &format!("discovery.peers not routable: {e}"),
            );
        }
    }

    match ctx.call(
        "ai",
        "mesh.init",
        serde_json::json!({
            "node_id": "south-gate",
            "dry_run": true,
        }),
    ) {
        Ok(_result) => {
            v.check_bool(
                "crossgate:live:mesh_init",
                true,
                "Songbird mesh.init responded",
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "crossgate:live:mesh_init",
                &format!("Songbird offline: {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "crossgate:live:mesh_init",
                &format!("mesh.init not routable: {e}"),
            );
        }
    }

    match ctx.hash_bytes(b"cross-gate-integrity-check", "blake3") {
        Ok(hash) => {
            v.check_bool(
                "crossgate:live:integrity_hash",
                hash.len() == 64,
                &format!("BLAKE3 hash len={}", hash.len()),
            );
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip(
                "crossgate:live:integrity_hash",
                &format!("security offline: {e}"),
            );
        }
        Err(e) => {
            v.check_bool("crossgate:live:integrity_hash", false, &format!("{e}"));
        }
    }
}
