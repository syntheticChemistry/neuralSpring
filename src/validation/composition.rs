// SPDX-License-Identifier: AGPL-3.0-or-later

//! Composition validation infrastructure for NUCLEUS proto-nucleate patterns.
//!
//! Extends the hotSpring-style validation harness with primal-composition
//! primitives: capability-based discovery, honest skip (exit 2), JSON-RPC
//! probes, and bonding validation.
//!
//! ## Evolution context
//!
//! Python baselines validated Rust correctness. Now Rust + Python baselines
//! validate NUCLEUS composition patterns — primals discoverable via IPC,
//! capabilities routable via `by_capability`, bonds enforced per atomic
//! boundary.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config;
use crate::primal_names;

/// Result of attempting to discover a primal's Unix socket.
#[derive(Debug, Clone)]
pub enum DiscoveryResult {
    /// Socket found at the given path.
    Found(PathBuf),
    /// Primal not running — no socket found.
    NotFound {
        /// Primal name used in the search.
        primal: String,
        /// Directories that were probed.
        searched: Vec<PathBuf>,
    },
}

impl DiscoveryResult {
    /// Returns `true` if the primal was discovered.
    #[must_use]
    pub const fn is_found(&self) -> bool {
        matches!(self, Self::Found(_))
    }
}

/// Discover a primal's Unix socket using the biomeOS 5-tier discovery order.
///
/// 1. `$BIOMEOS_ORCHESTRATOR_SOCKET` env override
/// 2. `$XDG_RUNTIME_DIR/biomeos/{primal}*.sock`
/// 3. `/tmp/biomeos/{primal}*.sock`
/// 4. `$XDG_RUNTIME_DIR/{primal}/*.sock` (legacy)
/// 5. `/tmp/{primal}-*.sock` (legacy)
///
/// Socket matching uses both the niche name (e.g. `neuralspring`) and the
/// hyphenated `CARGO_PKG_NAME` form (e.g. `neural-spring`) to handle springs
/// whose binary name differs from their niche name.
#[must_use]
pub fn discover_primal_socket(primal: &str) -> DiscoveryResult {
    let mut searched = Vec::new();

    // Tier 0: explicit orchestrator socket override
    if let Ok(override_path) = std::env::var(config::ENV_BIOMEOS_ORCHESTRATOR) {
        let p = PathBuf::from(&override_path);
        searched.push(p.clone());
        if p.exists() {
            return DiscoveryResult::Found(p);
        }
    }

    let alt_name = primal_to_pkg_name(primal);

    if let Ok(xdg) = std::env::var(config::ENV_XDG_RUNTIME_DIR) {
        let biomeos_dir = PathBuf::from(&xdg).join(config::BIOMEOS_SOCKET_SUBDIR);
        searched.push(biomeos_dir.clone());
        if let Some(sock) = find_socket_in_dir(&biomeos_dir, primal, alt_name.as_deref()) {
            return DiscoveryResult::Found(sock);
        }

        let legacy_dir = PathBuf::from(&xdg).join(primal);
        searched.push(legacy_dir.clone());
        if let Some(sock) = find_socket_in_dir(&legacy_dir, primal, alt_name.as_deref()) {
            return DiscoveryResult::Found(sock);
        }
    }

    let tmp_biomeos = std::env::temp_dir().join(config::BIOMEOS_SOCKET_SUBDIR);
    searched.push(tmp_biomeos.clone());
    if let Some(sock) = find_socket_in_dir(&tmp_biomeos, primal, alt_name.as_deref()) {
        return DiscoveryResult::Found(sock);
    }

    let tmp_legacy = std::env::temp_dir();
    searched.push(tmp_legacy.clone());
    if let Some(sock) = find_socket_in_dir(&tmp_legacy, primal, alt_name.as_deref()) {
        return DiscoveryResult::Found(sock);
    }

    DiscoveryResult::NotFound {
        primal: primal.to_string(),
        searched,
    }
}

/// Convert a niche name (e.g. `neuralspring`) to its probable `CARGO_PKG_NAME`
/// form (e.g. `neural-spring`). Returns `None` if the name contains no
/// recognizable camelCase boundary (i.e. it's already a simple name like
/// `beardog` that doesn't need an alternate form).
fn primal_to_pkg_name(niche: &str) -> Option<String> {
    let known = [
        ("neuralspring", "neural-spring"),
        ("hotspring", "hot-spring"),
        ("wetspring", "wet-spring"),
        ("groundspring", "ground-spring"),
        ("airspring", "air-spring"),
        ("healthspring", "health-spring"),
        ("ludospring", "ludo-spring"),
        ("primalspring", "primal-spring"),
        ("esotericwebb", "esoteric-webb"),
    ];
    for &(niche_name, pkg_name) in &known {
        if niche == niche_name {
            return Some(pkg_name.to_string());
        }
    }
    None
}

fn find_socket_in_dir(dir: &Path, primal: &str, alt_name: Option<&str>) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.ends_with(".sock") {
            continue;
        }
        if name_str.contains(primal) {
            return Some(entry.path());
        }
        if let Some(alt) = alt_name {
            if name_str.contains(alt) {
                return Some(entry.path());
            }
        }
    }
    None
}

/// Send a JSON-RPC 2.0 request over a Unix socket and return the response.
///
/// Uses newline-delimited framing per `PRIMAL_IPC_PROTOCOL.md`.
///
/// # Errors
///
/// Returns an error if the socket cannot be connected, the request fails
/// to send, or the response cannot be parsed.
pub fn json_rpc_call(
    socket: &Path,
    method: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    let stream =
        UnixStream::connect(socket).map_err(|e| format!("connect {}: {e}", socket.display()))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| format!("set_read_timeout: {e}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| format!("set_write_timeout: {e}"))?;

    json_rpc_on_stream(stream, method, params)
}

fn json_rpc_on_stream(
    mut stream: UnixStream,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let mut payload = serde_json::to_vec(&request).map_err(|e| format!("serialize: {e}"))?;
    payload.push(b'\n');

    stream
        .write_all(&payload)
        .map_err(|e| format!("write: {e}"))?;
    stream.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    let resp: serde_json::Value =
        serde_json::from_str(line.trim()).map_err(|e| format!("parse: {e}"))?;

    if let Some(err) = resp.get("error") {
        return Err(format!("RPC error: {err}"));
    }

    resp.get("result")
        .cloned()
        .ok_or_else(|| "response missing 'result' field".to_string())
}

/// Probe a primal's `health.liveness` endpoint.
///
/// Returns `Ok(())` if the primal responds, `Err` with reason otherwise.
///
/// # Errors
///
/// Returns an error if the primal is unreachable or responds with an error.
pub fn probe_liveness(socket: &Path, timeout: Duration) -> Result<(), String> {
    json_rpc_call(socket, "health.liveness", &serde_json::json!({}), timeout)?;
    Ok(())
}

/// Probe a primal's `capabilities.list` endpoint.
///
/// Returns the list of capability strings the primal advertises.
///
/// # Errors
///
/// Returns an error if the primal is unreachable or doesn't advertise.
pub fn probe_capabilities(socket: &Path, timeout: Duration) -> Result<Vec<String>, String> {
    let result = json_rpc_call(socket, "capabilities.list", &serde_json::json!({}), timeout)?;

    let caps = result
        .get("capabilities")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(caps)
}

/// Call a primal capability by name and return the raw result.
///
/// # Errors
///
/// Returns an error if the call fails.
pub fn call_capability(
    socket: &Path,
    capability: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    json_rpc_call(socket, capability, params, timeout)
}

/// Proto-nucleate node descriptor for composition validation.
#[derive(Debug, Clone)]
pub struct ProtoNucleateNode {
    /// Node name (lowercase discovery hint).
    pub name: &'static str,
    /// Primary capability used for `by_capability` discovery.
    pub by_capability: &'static str,
    /// Whether this node is required for the composition to be valid.
    pub required: bool,
}

/// The proto-nucleate graph for neuralSpring inference composition.
///
/// Derived from `primalSpring/graphs/downstream/downstream_manifest.toml` neuralspring entry.
#[must_use]
pub fn inference_proto_nucleate_nodes() -> Vec<ProtoNucleateNode> {
    vec![
        ProtoNucleateNode {
            name: primal_names::BIOMEOS,
            by_capability: "graph.deploy",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::BEARDOG,
            by_capability: "security",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::SONGBIRD,
            by_capability: "discovery",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::CORALREEF,
            by_capability: "shader.compile.wgsl",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::TOADSTOOL,
            by_capability: "compute.dispatch.submit",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::SQUIRREL,
            by_capability: "ai.query",
            required: false,
        },
        ProtoNucleateNode {
            name: primal_names::NESTGATE,
            by_capability: "storage.retrieve",
            required: false,
        },
    ]
}

/// Bonding policy for NUCLEUS composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondType {
    /// Shared trust domain — all primals freely exchange capabilities.
    Metallic,
    /// Different families — limited cross-call surface.
    Ionic,
    /// Same family seed — full meld routing.
    Covalent,
    /// Read-only ephemeral connections.
    Weak,
}

impl std::fmt::Display for BondType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metallic => f.write_str("Metallic"),
            Self::Ionic => f.write_str("Ionic"),
            Self::Covalent => f.write_str("Covalent"),
            Self::Weak => f.write_str("Weak"),
        }
    }
}

/// Skip-aware exit code: 0 = pass, 1 = fail, 2 = all skipped.
///
/// Follows primalSpring's `exit_code_skip_aware` pattern — composition
/// validators that find no live primals exit 2 (honest skip), not 0.
#[must_use]
pub const fn exit_code_skip_aware(passed: usize, failed: usize, skipped: usize) -> i32 {
    if failed > 0 {
        1
    } else if passed > 0 {
        0
    } else if skipped > 0 {
        2
    } else {
        1
    }
}

/// Science capability baseline for Rust→IPC parity validation.
///
/// Each baseline defines:
/// - The JSON-RPC method name
/// - The params to send
/// - A closure that computes the expected result via direct Rust calls
/// - Tolerance for numeric comparison
///
/// This is the third validation tier: Python validated Rust, now Rust validates IPC.
#[derive(Clone)]
pub struct ScienceBaseline {
    /// JSON-RPC method name (e.g. `science.spectral_analysis`).
    pub method: &'static str,
    /// JSON-RPC params to send.
    pub params: serde_json::Value,
    /// Keys in the IPC response to validate (each maps to a known Rust value).
    pub expected: Vec<(&'static str, f64)>,
    /// Absolute tolerance for numeric comparison.
    pub tolerance: f64,
}

/// Canonical science baselines for Rust→IPC parity.
///
/// Each entry exercises a science capability via IPC and compares to the
/// known Rust result computed with identical parameters (same seed, dim, disorder).
#[must_use]
pub fn science_baselines() -> Vec<ScienceBaseline> {
    use crate::anderson_localization::{anderson_hamiltonian_random, mean_ipr};
    use crate::eigh::eigh_householder_qr;
    use crate::rng::Rng;
    use crate::tolerances;
    use crate::weight_spectral;

    // Baseline 1: science.spectral_analysis (dim=16, disorder=2.0, seed=42)
    let spectral = {
        let n = 16;
        let w = 2.0;
        let seed = 42;
        let mut rng = Rng::new(seed);
        let h = anderson_hamiltonian_random(n, 1.0, w, &mut rng);
        let decomp = eigh_householder_qr(&h, n);
        let ipr_val = mean_ipr(&decomp.eigenvectors, n);
        let mut evals = decomp.eigenvalues;
        evals.sort_by(f64::total_cmp);
        let lsr = weight_spectral::level_spacing_ratio(&evals);
        let bw = weight_spectral::spectral_bandwidth(&evals);

        ScienceBaseline {
            method: "science.spectral_analysis",
            params: serde_json::json!({ "dim": n, "disorder": w, "seed": seed }),
            expected: vec![
                ("mean_ipr", ipr_val),
                ("level_spacing_ratio", lsr),
                ("bandwidth", bw),
            ],
            tolerance: tolerances::SPECIAL_FUNCTION_F64,
        }
    };

    // Baseline 2: science.ipr (uniform wavefunction, IPR = 1/n)
    let ipr_uniform = {
        let n = 8_usize;
        let n_f64 = 8.0_f64;
        let amp = 1.0 / n_f64.sqrt();
        let wf: Vec<f64> = vec![amp; n];
        let expected_ipr = crate::anderson_localization::ipr(&wf);

        ScienceBaseline {
            method: "science.ipr",
            params: serde_json::json!({ "wavefunction": wf }),
            expected: vec![("ipr", expected_ipr)],
            tolerance: tolerances::EXACT_F64,
        }
    };

    // Baseline 3: science.hessian_eigen (quadratic surface, dim=10)
    let hessian_quad = {
        let n = 10;
        let mut hessian = vec![0.0; n * n];
        for i in 0..n {
            #[expect(clippy::cast_precision_loss, reason = "small index → f64")]
            let v = (i + 1) as f64;
            hessian[i * n + i] = v;
        }
        let _decomp = eigh_householder_qr(&hessian, n);
        #[expect(clippy::cast_precision_loss, reason = "small index → f64")]
        let expected_trace = (1..=n).map(|i| i as f64).sum::<f64>();

        ScienceBaseline {
            method: "science.hessian_eigen",
            params: serde_json::json!({ "dim": n, "surface_type": "quadratic" }),
            expected: vec![("trace", expected_trace)],
            tolerance: tolerances::SPECIAL_FUNCTION_F64,
        }
    };

    // Baseline 4: science.disorder_sweep (lattice_size=10, seed=42)
    let disorder_sweep = {
        let n = 10;
        let seed = 42_u64;
        let w_vals = vec![1.0, 4.0, 16.0];
        let mut rng = Rng::new(seed);
        let iprs = crate::anderson_localization::disorder_sweep(n, 1.0, &w_vals, &mut rng);

        ScienceBaseline {
            method: "science.disorder_sweep",
            params: serde_json::json!({
                "lattice_size": n,
                "disorder_values": w_vals,
                "seed": seed,
                "hopping": 1.0,
            }),
            expected: iprs
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    // ipr_values[i] — we'll check these in the binary by index
                    match i {
                        0 => ("ipr_w1", v),
                        1 => ("ipr_w4", v),
                        _ => ("ipr_w16", v),
                    }
                })
                .collect(),
            tolerance: tolerances::SPECIAL_FUNCTION_F64,
        }
    };

    vec![spectral, ipr_uniform, hessian_quad, disorder_sweep]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_not_found_for_fake_primal() {
        let result = discover_primal_socket("nonexistent_primal_xyz");
        assert!(!result.is_found());
    }

    #[test]
    fn exit_code_all_pass() {
        assert_eq!(exit_code_skip_aware(5, 0, 0), 0);
    }

    #[test]
    fn exit_code_any_fail() {
        assert_eq!(exit_code_skip_aware(3, 1, 0), 1);
    }

    #[test]
    fn exit_code_all_skip() {
        assert_eq!(exit_code_skip_aware(0, 0, 5), 2);
    }

    #[test]
    fn exit_code_pass_and_skip() {
        assert_eq!(exit_code_skip_aware(2, 0, 3), 0);
    }

    #[test]
    fn exit_code_empty() {
        assert_eq!(exit_code_skip_aware(0, 0, 0), 1);
    }

    #[test]
    fn proto_nucleate_nodes_cover_key_primals() {
        let nodes = inference_proto_nucleate_nodes();
        let names: Vec<&str> = nodes.iter().map(|n| n.name).collect();
        assert!(names.contains(&"toadstool"));
        assert!(names.contains(&"squirrel"));
        assert!(names.contains(&"coralreef"));
        assert!(names.contains(&"biomeos"));
    }

    #[test]
    fn bond_type_display() {
        assert_eq!(BondType::Metallic.to_string(), "Metallic");
        assert_eq!(BondType::Ionic.to_string(), "Ionic");
    }

    #[test]
    fn science_baselines_non_empty() {
        let baselines = science_baselines();
        assert!(baselines.len() >= 4, "should have at least 4 baselines");
        for b in &baselines {
            assert!(!b.method.is_empty());
            assert!(!b.expected.is_empty());
            assert!(b.tolerance > 0.0);
        }
    }

    #[test]
    fn science_baselines_deterministic() {
        let b1 = science_baselines();
        let b2 = science_baselines();
        for (a, b) in b1.iter().zip(b2.iter()) {
            assert_eq!(a.method, b.method);
            for ((k1, v1), (k2, v2)) in a.expected.iter().zip(b.expected.iter()) {
                assert_eq!(k1, k2);
                assert_eq!(
                    v1.to_bits(),
                    v2.to_bits(),
                    "baseline {k1} must be deterministic"
                );
            }
        }
    }

    #[test]
    fn primal_to_pkg_name_known() {
        assert_eq!(
            primal_to_pkg_name("neuralspring"),
            Some("neural-spring".to_string())
        );
        assert_eq!(
            primal_to_pkg_name("hotspring"),
            Some("hot-spring".to_string())
        );
        assert_eq!(primal_to_pkg_name("beardog"), None);
    }

    #[test]
    fn primal_to_pkg_name_unknown() {
        assert_eq!(primal_to_pkg_name("unknownprimal"), None);
    }
}
