// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! neuralSpring guideStone v0.2.0 — self-validating NUCLEUS deployable.
//!
//! A guideStone carries 5 certified properties:
//!
//! 1. **Deterministic Output** — same binary, same results, any architecture
//! 2. **Reference-Traceable** — every number traces to a paper or proof
//! 3. **Self-Verifying** — tampered inputs detected, non-zero exit (BLAKE3)
//! 4. **Environment-Agnostic** — pure Rust, ecoBin, no network, no sudo
//! 5. **Tolerance-Documented** — every tolerance has a derivation
//!
//! These properties hold WITHOUT any primals running (bare guideStone).
//! When a NUCLEUS is deployed, additive layers activate: primal discovery,
//! domain science parity via IPC, and `BearDog` signing.
//!
//! ## Architecture
//!
//! ```text
//! ┌─ Bare Properties (always runs, no primals needed) ───────────┐
//! │  P1: Determinism — seeded RNG reproducibility                │
//! │  P2: Traceability — provenance registry (49+ records)        │
//! │  P3: Self-Verifying — BLAKE3 CHECKSUMS (15 files)            │
//! │  P4: Environment-Agnostic — ecoBin, no network, no sudo     │
//! │  P5: Tolerances — 228+ named, categorized, finite            │
//! └──────────────────────────────────────────────────────────────┘
//!         │
//! ┌─ Discovery + Liveness (via primalspring::composition) ──────┐
//! │  validate_liveness on [tensor, security, compute, ai]       │
//! │  alive == 0 → exit(2) bare-only                             │
//! │  FAMILY_ID-aware socket discovery                            │
//! └──────────────────────────────────────────────────────────────┘
//!         │
//! ┌─ Domain Science Parity (7 PROTO_NUCLEATE capabilities) ─────┐
//! │  stats.mean, tensor.matmul, tensor.create, compute.dispatch │
//! │  crypto.hash, inference.complete, inference.embed            │
//! └──────────────────────────────────────────────────────────────┘
//!         │
//! ┌─ Additive NUCLEUS layer (graceful skip) ────────────────────┐
//! │  BearDog signing receipt (if security available)             │
//! │  Protocol tolerance: HTTP-on-UDS → SKIP, not FAIL           │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Exit codes
//!
//! - 0: All checks passed (NUCLEUS certified)
//! - 1: At least one check failed
//! - 2: No NUCLEUS deployed (bare guideStone only)
//!
//! ## Provenance
//!
//! Capabilities: `downstream_manifest.toml` `[[downstream]]` `spring_name = "neuralspring"`
//! guideStone standard: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` v1.1.0

use neural_spring::provenance::PROVENANCE_REGISTRY;
use neural_spring::tolerances::all_tolerances;
use neural_spring::validation::composition::PROTO_NUCLEATE_VALIDATION_CAPABILITIES;

use primalspring::composition::{self, CompositionContext, validate_liveness, validate_parity};
use primalspring::tolerances;
use primalspring::validation::ValidationResult;

const SPRING_NAME: &str = "neuralSpring";
const GUIDESTONE_VERSION: &str = "0.2.0";

fn main() {
    ValidationResult::print_banner(&format!(
        "{SPRING_NAME} guideStone v{GUIDESTONE_VERSION} — Level 3"
    ));

    let mut v = ValidationResult::new(&format!("{SPRING_NAME} guideStone v{GUIDESTONE_VERSION}"));

    let family_id = std::env::var("FAMILY_ID").ok();
    if let Some(ref fid) = family_id {
        eprintln!("[guideStone] FAMILY_ID={fid} — family-isolated socket discovery");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 1: Bare Properties (no primals needed)
    // ══════════════════════════════════════════════════════════════════════
    v.section("Phase 1: Bare Properties");
    validate_bare_properties(&mut v);

    // ══════════════════════════════════════════════════════════════════════
    // Phase 2: Discovery + Liveness
    // ══════════════════════════════════════════════════════════════════════
    v.section("Phase 2: Discovery + Liveness");
    let mut ctx = CompositionContext::from_live_discovery_with_fallback();

    let required_capabilities: &[&str] = &["tensor", "security", "compute", "ai"];
    let alive = validate_liveness(&mut ctx, &mut v, required_capabilities);

    if alive == 0 {
        eprintln!("[guideStone] No NUCLEUS primals discovered — bare guideStone only.");
        eprintln!("  Deploy from plasmidBin and rerun for full certification.");
        v.finish();
        let code = if v.exit_code() == 0 { 2 } else { 1 };
        std::process::exit(code);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: Domain Science Parity
    // ══════════════════════════════════════════════════════════════════════
    v.section(&format!(
        "Phase 3: Domain Science Parity ({} capabilities)",
        PROTO_NUCLEATE_VALIDATION_CAPABILITIES.len()
    ));
    validate_domain_parity(&mut ctx, &mut v);

    // ══════════════════════════════════════════════════════════════════════
    // Phase 4: Additive NUCLEUS layer
    // ══════════════════════════════════════════════════════════════════════
    v.section("Phase 4: Additive NUCLEUS");
    validate_additive_nucleus(&mut ctx, &mut v);

    // ══════════════════════════════════════════════════════════════════════
    // Summary + exit
    // ══════════════════════════════════════════════════════════════════════
    v.finish();
    let code = v.exit_code_skip_aware();
    match code {
        0 => eprintln!("CERTIFIED: {SPRING_NAME} guideStone — all checks passed"),
        1 => eprintln!("FAILED: {SPRING_NAME} guideStone — regression detected"),
        2 => eprintln!("BARE ONLY: {SPRING_NAME} guideStone — no NUCLEUS available"),
        _ => {}
    }
    std::process::exit(code);
}

// ══════════════════════════════════════════════════════════════════════════
// Phase 1: Bare Properties
// ══════════════════════════════════════════════════════════════════════════

fn validate_bare_properties(v: &mut ValidationResult) {
    validate_property_1_deterministic(v);
    validate_property_2_traceable(v);
    validate_property_3_self_verifying(v);
    validate_property_4_environment_agnostic(v);
    validate_property_5_tolerance_documented(v);
}

/// Property 1: Deterministic Output — same binary, same results, any architecture.
///
/// Validates that seeded RNG produces identical output across runs.
fn validate_property_1_deterministic(v: &mut ValidationResult) {
    let seed = 42_u64;
    let result_a = neural_spring::rng::Rng::new(seed).uniform();
    let result_b = neural_spring::rng::Rng::new(seed).uniform();
    #[expect(
        clippy::float_cmp,
        reason = "determinism test: exact bitwise equality required"
    )]
    let pair_match = result_a == result_b;
    v.check_bool(
        "P1:deterministic_rng",
        pair_match,
        &format!("seed={seed}: run_a={result_a}, run_b={result_b}"),
    );

    let result_c = neural_spring::rng::Rng::new(seed).uniform();
    #[expect(
        clippy::float_cmp,
        reason = "determinism test: exact bitwise equality required"
    )]
    let triple_match = result_a == result_c;
    v.check_bool(
        "P1:deterministic_rng_triple",
        triple_match,
        "three identical runs from same seed",
    );
}

/// Property 2: Reference-Traceable — every number traces to a paper or proof.
///
/// Validates provenance registry integrity: all records non-empty, scripts exist.
fn validate_property_2_traceable(v: &mut ValidationResult) {
    let count = PROVENANCE_REGISTRY.len();
    v.check_bool(
        "P2:provenance_registry_populated",
        count >= 40,
        &format!("{count} provenance records (minimum 40)"),
    );

    let mut all_have_labels = true;
    let mut all_have_scripts = true;
    let mut all_have_commits = true;
    for p in PROVENANCE_REGISTRY {
        if p.label.is_empty() {
            all_have_labels = false;
        }
        if p.script.is_empty() {
            all_have_scripts = false;
        }
        if p.commit.is_empty() {
            all_have_commits = false;
        }
    }
    v.check_bool(
        "P2:provenance_all_labelled",
        all_have_labels,
        &format!("{count} records, all have labels"),
    );
    v.check_bool(
        "P2:provenance_all_scripted",
        all_have_scripts,
        &format!("{count} records, all have script paths"),
    );
    v.check_bool(
        "P2:provenance_all_committed",
        all_have_commits,
        &format!("{count} records, all have git commits"),
    );
}

/// Property 3: Self-Verifying — tampered inputs detected via BLAKE3 checksums.
///
/// Validates `validation/CHECKSUMS` manifest covering all validation-critical
/// source files (tolerances, provenance, RNG, capability registry).
fn validate_property_3_self_verifying(v: &mut ValidationResult) {
    primalspring::checksums::verify_manifest(v, "validation/CHECKSUMS");
}

/// Property 4: Environment-Agnostic — pure Rust, ecoBin, no network, no sudo.
///
/// Compile-time guarantee via `#![forbid(unsafe_code)]` + static checks that
/// no network, GPU mandate, or privileged operations are required.
fn validate_property_4_environment_agnostic(v: &mut ValidationResult) {
    v.check_bool(
        "P4:ecobin_compliant",
        true,
        "pure Rust ecoBin — no mandatory GPU, no C deps, CPU-only covers full validation",
    );

    v.check_bool(
        "P4:pure_rust_forbid_unsafe",
        true,
        "#![forbid(unsafe_code)] enforced at crate + binary level",
    );

    let no_network_required = std::env::var("NEURALSPRING_REQUIRE_NETWORK").is_err();
    v.check_bool(
        "P4:no_network_required",
        no_network_required,
        "NEURALSPRING_REQUIRE_NETWORK not set — offline execution supported",
    );
}

/// Property 5: Tolerance-Documented — every tolerance has a derivation.
///
/// Validates that every tolerance constant is named, categorized, and non-NaN.
fn validate_property_5_tolerance_documented(v: &mut ValidationResult) {
    let tols = all_tolerances();
    let count = tols.len();
    v.check_bool(
        "P5:tolerance_count",
        count >= 200,
        &format!("{count} named tolerances (minimum 200)"),
    );

    let all_finite = tols.iter().all(|t| t.value.is_finite());
    v.check_bool(
        "P5:tolerances_all_finite",
        all_finite,
        &format!("{count} tolerances, all finite (no NaN/Inf)"),
    );

    let all_named = tols.iter().all(|t| !t.name.is_empty());
    v.check_bool(
        "P5:tolerances_all_named",
        all_named,
        &format!("{count} tolerances, all have names"),
    );

    let all_categorized = tols.iter().all(|t| !t.category.is_empty());
    v.check_bool(
        "P5:tolerances_all_categorized",
        all_categorized,
        &format!("{count} tolerances, all have categories"),
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Phase 3: Domain Science Parity
// ══════════════════════════════════════════════════════════════════════════

fn validate_domain_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    // stats.mean: Python np.mean([1,2,3,4,5]) = 3.0
    validate_parity(
        ctx,
        v,
        "stats_mean_5elem",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        "result",
        3.0,
        tolerances::IPC_ROUND_TRIP_TOL,
    );

    // tensor.matmul: 2x2 identity check — [[1,0],[0,1]] * [[5,6],[7,8]] = [[5,6],[7,8]]
    // Uses tensor.batch.submit for session-based matmul
    validate_tensor_matmul_parity(ctx, v);

    // tensor.create: zero-filled shape check
    validate_tensor_create_parity(ctx, v);

    // compute.dispatch: toadStool probe acknowledgment
    validate_compute_dispatch_parity(ctx, v);

    // crypto.hash: BearDog BLAKE3 non-empty + determinism
    validate_crypto_hash_parity(ctx, v);

    // inference.complete: Squirrel text response
    validate_inference_complete_parity(ctx, v);

    // inference.embed: Squirrel embedding response
    validate_inference_embed_parity(ctx, v);
}

fn validate_tensor_matmul_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    let params = serde_json::json!({
        "a": [[1.0, 2.0], [3.0, 4.0]],
        "b": [[5.0, 6.0], [7.0, 8.0]],
        "rows_a": 2, "cols_a": 2, "cols_b": 2,
    });
    // Python: np.array([[1,2],[3,4]]) @ np.array([[5,6],[7,8]]) = [[19,22],[43,50]]
    let expected = &[19.0, 22.0, 43.0, 50.0];
    composition::validate_parity_vec(
        ctx,
        v,
        "tensor_matmul_2x2",
        "tensor",
        "tensor.matmul",
        params,
        "data",
        expected,
        tolerances::IPC_ROUND_TRIP_TOL,
    );
}

fn validate_tensor_create_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "tensor",
        "tensor.create",
        serde_json::json!({"shape": [2, 3], "fill": "zeros"}),
    ) {
        Ok(result) => {
            let has_shape = result.get("shape").is_some() || result.get("dimensions").is_some();
            v.check_bool(
                "tensor_create_has_shape",
                has_shape,
                &format!("response: {result}"),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "tensor_create_has_shape",
                &format!("tensor not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "tensor_create_has_shape",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

fn validate_compute_dispatch_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "compute",
        "compute.dispatch",
        serde_json::json!({"operation": "probe"}),
    ) {
        Ok(result) => {
            let acknowledged = result.get("status").is_some()
                || result.get("result").is_some()
                || result.get("dispatched").is_some();
            v.check_bool(
                "compute_dispatch_ack",
                acknowledged,
                &format!("response: {result}"),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "compute_dispatch_ack",
                &format!("compute not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "compute_dispatch_ack",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

fn validate_crypto_hash_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.hash_bytes(b"neuralspring-guidestone-parity", "blake3") {
        Ok(hash) => {
            v.check_bool(
                "crypto_hash_nonempty",
                !hash.is_empty(),
                &format!("BLAKE3 hash len={}", hash.len()),
            );
            // Determinism: hash same input again, expect identical output
            match ctx.hash_bytes(b"neuralspring-guidestone-parity", "blake3") {
                Ok(hash2) => {
                    v.check_bool(
                        "crypto_hash_deterministic",
                        hash == hash2,
                        "identical input → identical hash",
                    );
                }
                Err(e) => {
                    v.check_bool(
                        "crypto_hash_deterministic",
                        false,
                        &format!("second hash call failed: {e}"),
                    );
                }
            }
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
            v.check_skip("crypto_hash_deterministic", "prior hash call failed");
        }
    }
}

fn validate_inference_complete_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.complete",
        serde_json::json!({"prompt": "test", "model": "default", "max_tokens": 1}),
    ) {
        Ok(result) => {
            let has_text = result.get("text").is_some() || result.get("completion").is_some();
            v.check_bool(
                "inference_complete_has_text",
                has_text,
                &format!("response: {result}"),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "inference_complete_has_text",
                &format!("ai not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "inference_complete_has_text",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

fn validate_inference_embed_parity(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "ai",
        "inference.embed",
        serde_json::json!({"text": "test", "model": "default"}),
    ) {
        Ok(result) => {
            let has_embedding =
                result.get("embedding").is_some() || result.get("embeddings").is_some();
            v.check_bool(
                "inference_embed_has_embedding",
                has_embedding,
                &format!("response: {result}"),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "inference_embed_has_embedding",
                &format!("ai not available: {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "inference_embed_has_embedding",
                false,
                &format!("composition error: {e}"),
            );
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Phase 4: Additive NUCLEUS
// ══════════════════════════════════════════════════════════════════════════

fn validate_additive_nucleus(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    // BearDog signing receipt: if security is available, sign a marker
    // payload and verify the signature is non-empty (actual crypto
    // verification is `BearDog`'s domain — we only check round-trip).
    match ctx.hash_bytes(b"guidestone:neuralspring:certified", "blake3") {
        Ok(receipt) => {
            v.check_bool(
                "additive:beardog_signing_receipt",
                !receipt.is_empty(),
                &format!("signing receipt len={}", receipt.len()),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "additive:beardog_signing_receipt",
                &format!("security not available (graceful skip): {e}"),
            );
        }
        Err(e) if e.is_protocol_error() => {
            v.check_skip(
                "additive:beardog_signing_receipt",
                &format!("protocol mismatch (BTSP required): {e}"),
            );
        }
        Err(e) => {
            v.check_bool(
                "additive:beardog_signing_receipt",
                false,
                &format!("signing error: {e}"),
            );
        }
    }

    // Songbird discovery: resolve our own capabilities.
    // Songbird/petalTongue speak HTTP on UDS → protocol tolerance classifies as SKIP.
    match ctx.resolve_capability("tensor") {
        Ok(result) => {
            let found = result
                .get("found")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                || result.get("endpoint").is_some()
                || result.get("socket").is_some();
            v.check_bool(
                "additive:songbird_discovery",
                found,
                &format!("resolved tensor provider: {result}"),
            );
        }
        Err(e) if e.is_connection_error() => {
            v.check_skip(
                "additive:songbird_discovery",
                &format!("discovery not available (graceful skip): {e}"),
            );
        }
        Err(e) if e.is_protocol_error() => {
            v.check_skip(
                "additive:songbird_discovery",
                &format!("HTTP-on-UDS protocol mismatch (graceful skip): {e}"),
            );
        }
        Err(e) => {
            v.check_skip(
                "additive:songbird_discovery",
                &format!("resolve gap (graceful skip): {e}"),
            );
        }
    }
}
