// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp094: neuralSpring NUCLEUS Composition Parity
//!
//! Replicates primalSpring's exp094 pattern for the neuralSpring niche.
//! Validates the full NUCLEUS composition pipeline:
//!
//!   Tower (`BearDog` security + `Songbird` discovery)
//!   → Node (`barraCuda` tensor + `coralReef` shader + `toadStool` dispatch)
//!   → Nest (`NestGate` storage + provenance trio)
//!   → Niche (neuralSpring science capabilities)
//!
//! Pattern: discover → call → extract → compare → report
//!
//! Environment:
//!   `REMOTE_GATE_HOST` — enables TCP/gateway mode (Docker/benchScale)
//!   `BIOMEOS_PORT`     — biomeOS TCP port (default 9800)
//!   `FAMILY_ID`        — primal family for socket scoping

use primalspring::composition::{CompositionContext, validate_parity};
use primalspring::tolerances;
use primalspring::validation::ValidationResult;

fn main() {
    ValidationResult::new("neuralSpring Exp094 — NUCLEUS Composition Parity")
        .with_provenance("exp094_neuralspring_composition_parity", "2026-05-08")
        .run(
            "NUCLEUS base + neuralSpring niche parity (Tower + Node + Nest + Science)",
            |v| {
                let mut ctx = CompositionContext::from_live_discovery_with_fallback();
                let caps = ctx.available_capabilities();

                v.section("Discovery");
                v.check_bool(
                    "capabilities_found",
                    !caps.is_empty(),
                    &format!(
                        "discovered {} capabilities: {}",
                        caps.len(),
                        caps.join(", ")
                    ),
                );

                nucleus_base(&mut ctx, v);
                niche_science_parity(&mut ctx, v);
                niche_inference_probe(&mut ctx, v);
                cross_atomic_pipeline(&mut ctx, v);
            },
        );
}

// ═══════════════════════════════════════════════════════════════════════
// NUCLEUS Base (Tower + Node + Nest) — adapted from exp095 template
// ═══════════════════════════════════════════════════════════════════════

fn nucleus_base(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    tower_atomic(ctx, v);
    node_atomic(ctx, v);
    nest_atomic(ctx, v);
}

fn tower_atomic(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Tower Atomic (BearDog + Songbird)");

    for (name, cap) in [
        ("beardog_alive", "security"),
        ("songbird_alive", "discovery"),
    ] {
        match ctx.health_check(cap) {
            Ok(alive) => v.check_bool(name, alive, &format!("{cap} health normalized")),
            Err(e) if e.is_connection_error() => {
                v.check_skip(name, &format!("{cap} not running: {e}"));
            }
            Err(e) => v.check_bool(name, false, &format!("{cap} error: {e}")),
        }
    }

    match ctx.hash_bytes(b"neuralSpring composition parity test", "blake3") {
        Ok(hash) => {
            v.check_bool(
                "crypto_hash_nonempty",
                !hash.is_empty(),
                &format!(
                    "BLAKE3: {}... (len={})",
                    &hash[..hash.len().min(16)],
                    hash.len()
                ),
            );
            let deterministic = ctx
                .hash_bytes(b"neuralSpring composition parity test", "blake3")
                .is_ok_and(|h2| h2 == hash);
            v.check_bool(
                "crypto_hash_deterministic",
                deterministic,
                "same input → same hash",
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "crypto_hash_nonempty",
                &format!("security not available: {e}"),
            );
            v.check_skip("crypto_hash_deterministic", "security not available");
        }
        Err(e) => {
            v.check_bool("crypto_hash_nonempty", false, &format!("hash error: {e}"));
            v.check_skip("crypto_hash_deterministic", "prior call failed");
        }
    }

    for cap in ["security", "compute", "storage"] {
        let name = format!("resolve_{cap}");
        match ctx.resolve_capability(cap) {
            Ok(result) => {
                let found = result
                    .get("found")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                    || result.get("endpoint").is_some()
                    || result.get("socket").is_some();
                v.check_bool(&name, found, &format!("resolved {cap}: {result}"));
            }
            Err(e) if e.is_connection_error() => {
                v.check_skip(&name, &format!("discovery not available: {e}"));
            }
            Err(e) => v.check_bool(&name, false, &format!("resolve gap: {e}")),
        }
    }
}

fn node_atomic(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Node Atomic (barraCuda + coralReef + toadStool)");

    validate_parity(
        ctx,
        v,
        "tensor_stats_mean",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        "result",
        3.0,
        tolerances::CPU_GPU_PARITY_TOL,
    );

    match ctx.call(
        "shader",
        "shader.compile.capabilities",
        serde_json::json!({}),
    ) {
        Ok(result) => {
            let has_archs = result
                .get("supported_archs")
                .and_then(|a| a.as_array())
                .is_some_and(|a| !a.is_empty());
            v.check_bool(
                "shader_supported_archs",
                has_archs,
                &format!("archs: {result}"),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "shader_supported_archs",
                &format!("shader not available: {e}"),
            );
        }
        Err(e) => v.check_bool(
            "shader_supported_archs",
            false,
            &format!("shader error: {e}"),
        ),
    }

    match ctx.health_check("compute") {
        Ok(alive) => v.check_bool(
            "compute_dispatch_alive",
            alive,
            "toadStool health normalized",
        ),
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "compute_dispatch_alive",
                &format!("compute not available: {e}"),
            );
        }
        Err(e) => v.check_bool(
            "compute_dispatch_alive",
            false,
            &format!("compute error: {e}"),
        ),
    }
}

fn nest_atomic(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Nest Atomic (NestGate + provenance trio)");

    let family_id = std::env::var("FAMILY_ID").unwrap_or_else(|_| "nucleus01".to_owned());
    let test_key = "exp094_ns_parity_roundtrip";
    let test_value = "neuralspring_composition_validation_2026";

    let store_result = ctx
        .call(
            "storage",
            "storage.store",
            serde_json::json!({"family_id": family_id, "key": test_key, "value": test_value}),
        )
        .or_else(|_| {
            ctx.call(
                "storage",
                "storage.put",
                serde_json::json!({"family_id": family_id, "key": test_key, "value": test_value}),
            )
        });

    match store_result {
        Ok(_) => {
            let retrieve_result = ctx
                .call(
                    "storage",
                    "storage.retrieve",
                    serde_json::json!({"family_id": family_id, "key": test_key}),
                )
                .or_else(|_| {
                    ctx.call(
                        "storage",
                        "storage.get",
                        serde_json::json!({"family_id": family_id, "key": test_key}),
                    )
                });
            match retrieve_result {
                Ok(result) => {
                    let val = result.get("value").and_then(|v| v.as_str()).unwrap_or("");
                    v.check_bool(
                        "storage_roundtrip_match",
                        val == test_value,
                        &format!("stored={test_value}, retrieved={val}"),
                    );
                }
                Err(e) => {
                    v.check_bool(
                        "storage_roundtrip_match",
                        false,
                        &format!("retrieve error: {e}"),
                    );
                }
            }
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "storage_roundtrip_match",
                &format!("storage not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "storage_roundtrip_match",
                false,
                &format!("store error: {e}"),
            );
        }
    }

    for (name, cap) in [("sweetgrass_alive", "commit"), ("rhizocrypt_alive", "dag")] {
        match ctx.health_check(cap) {
            Ok(alive) => v.check_bool(name, alive, &format!("{cap} health normalized")),
            Err(e) if e.is_connection_error() => {
                v.check_skip(name, &format!("{cap} not available: {e}"));
            }
            Err(e) => v.check_bool(name, false, &format!("{cap} error: {e}")),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Niche: neuralSpring Science Parity
// ═══════════════════════════════════════════════════════════════════════

fn niche_science_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Niche — Science Domain Parity");

    // stats.mean parity: Python np.mean([2, 4, 6, 8, 10]) = 6.0
    validate_parity(
        ctx,
        v,
        "science_mean_5elem",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [2.0, 4.0, 6.0, 8.0, 10.0]}),
        "result",
        6.0,
        tolerances::EXACT_PARITY_TOL,
    );

    // stats.std_dev parity: Python np.std([2,4,4,4,5,5,7,9], ddof=1) ≈ 2.1381
    validate_parity(
        ctx,
        v,
        "science_std_dev_8elem",
        "tensor",
        "stats.std_dev",
        serde_json::json!({"data": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}),
        "result",
        2.138_089_935_299_395,
        1e-6,
    );

    // Probe spectral analysis capability availability
    match ctx.call(
        "science",
        "science.spectral_analysis",
        serde_json::json!({"matrix": [[1.0, 0.5], [0.5, 1.0]], "mode": "eigenvalues"}),
    ) {
        Ok(result) => {
            let has_eigenvalues =
                result.get("eigenvalues").is_some() || result.get("result").is_some();
            v.check_bool(
                "spectral_analysis_available",
                has_eigenvalues,
                &format!("spectral analysis responded: {result}"),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "spectral_analysis_available",
                &format!("science primal not running: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "spectral_analysis_available",
                false,
                &format!("spectral error: {e}"),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Niche: Inference Probe (Squirrel composition)
// ═══════════════════════════════════════════════════════════════════════

fn niche_inference_probe(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("Niche — Inference Probes");

    for method in ["inference.complete", "inference.embed", "inference.models"] {
        let name = method.replace('.', "_");
        match ctx.call("ai", method, serde_json::json!({"probe": true})) {
            Ok(result) => {
                v.check_bool(
                    &format!("{name}_reachable"),
                    true,
                    &format!("{method} responded: {result}"),
                );
            }
            Err(e) if e.is_connection_error() => {
                v.check_skip(
                    &format!("{name}_reachable"),
                    &format!("{method} not available (Squirrel not running): {e}"),
                );
            }
            Err(e) => {
                v.check_skip(
                    &format!("{name}_reachable"),
                    &format!("{method} error (non-connection): {e}"),
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Cross-Atomic Pipeline: hash → store → retrieve
// ═══════════════════════════════════════════════════════════════════════

fn cross_atomic_pipeline(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    v.section("NUCLEUS Cross-Atomic Pipeline");

    let payload = b"neuralSpring_exp094_cross_atomic_2026";
    match ctx.hash_bytes(payload, "blake3") {
        Ok(hash) => {
            v.check_bool(
                "cross_tower_hash",
                !hash.is_empty(),
                &format!("BLAKE3: {}...", &hash[..hash.len().min(16)]),
            );

            let family_id = std::env::var("FAMILY_ID").unwrap_or_else(|_| "nucleus01".to_owned());
            let key = "exp094_ns_cross_atomic_hash";
            match ctx.call(
                "storage",
                "storage.store",
                serde_json::json!({"family_id": family_id, "key": key, "value": hash}),
            ) {
                Ok(_) => {
                    match ctx.call(
                        "storage",
                        "storage.retrieve",
                        serde_json::json!({"family_id": family_id, "key": key}),
                    ) {
                        Ok(retrieved) => {
                            let val = retrieved
                                .get("value")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            v.check_bool(
                                "cross_nest_roundtrip",
                                val == hash,
                                "hash stored and retrieved matches",
                            );
                        }
                        Err(e) => {
                            v.check_bool(
                                "cross_nest_roundtrip",
                                false,
                                &format!("retrieve error: {e}"),
                            );
                        }
                    }
                }
                Err(e) if e.is_connection_error() => {
                    v.check_skip(
                        "cross_nest_roundtrip",
                        &format!("storage not available: {e}"),
                    );
                }
                Err(e) => {
                    v.check_bool("cross_nest_roundtrip", false, &format!("store error: {e}"));
                }
            }
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip("cross_tower_hash", &format!("security not available: {e}"));
            v.check_skip("cross_nest_roundtrip", "tower unavailable");
        }
        Err(e) => {
            v.check_bool("cross_tower_hash", false, &format!("hash error: {e}"));
            v.check_skip("cross_nest_roundtrip", "tower failed");
        }
    }
}
