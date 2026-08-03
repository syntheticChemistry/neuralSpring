// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified IPC facade over per-primal modules.
//!
//! [`IpcMathClient`] routes each method call through the capability
//! router rather than hardcoded primal names.

use std::time::Duration;

use crate::capabilities;
use crate::error::IpcError;

use super::DEFAULT_TIMEOUT;
use super::barracuda;
use super::beardog;
use super::coralreef;
use super::health::{IpcLivenessReport, probe_all};
use super::nestgate;
use super::router::CapabilityRouter;
use super::skunkbat;
use super::squirrel;
use super::toadstool;

/// Discovered primal socket paths for IPC dispatch.
///
/// Unified facade over per-primal IPC modules. Uses capability-based
/// routing: each method call resolves through the [`CapabilityRouter`]
/// rather than a fixed primal name. Missing primals produce `Err` at
/// call time with an honest discovery message.
pub struct IpcMathClient {
    router: CapabilityRouter,
    timeout: Duration,
}

impl IpcMathClient {
    /// Discover all math-relevant primals and return a connected client.
    ///
    /// Missing primals are recorded in the capability router — calls to
    /// their methods will return `Err` with an honest "not discovered"
    /// message.
    #[must_use]
    pub fn discover() -> Self {
        Self {
            router: CapabilityRouter::from_hints(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Override the default IPC timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Whether the barraCuda primal was discovered.
    #[must_use]
    pub fn has_barracuda(&self) -> bool {
        self.router.get(capabilities::STATS_MEAN).is_some()
    }

    /// Whether the toadStool primal was discovered.
    #[must_use]
    pub fn has_toadstool(&self) -> bool {
        self.router.get(capabilities::COMPUTE_DISPATCH).is_some()
    }

    /// Whether the `BearDog` primal was discovered.
    #[must_use]
    pub fn has_beardog(&self) -> bool {
        self.router.get(capabilities::CRYPTO_HASH).is_some()
    }

    /// Whether the coralReef primal was discovered.
    #[must_use]
    pub fn has_coralreef(&self) -> bool {
        self.router.get(capabilities::SHADER_COMPILE_WGSL).is_some()
    }

    /// Whether the Squirrel primal was discovered.
    #[must_use]
    pub fn has_squirrel(&self) -> bool {
        self.router.get(capabilities::INFERENCE_COMPLETE).is_some()
    }

    /// Probe liveness of all discovered primals.
    #[must_use]
    pub fn probe_all(&self) -> IpcLivenessReport {
        probe_all(&self.router, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // barraCuda surface (routed via stats.*/tensor.* capabilities)
    // ═══════════════════════════════════════════════════════════════

    /// `stats.mean` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `stats.mean` or the call fails.
    pub fn stats_mean(&self, data: &[f64]) -> Result<f64, IpcError> {
        barracuda::stats_mean(
            self.router.require(capabilities::STATS_MEAN)?,
            data,
            self.timeout,
        )
    }

    /// `stats.std_dev` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `stats.std_dev` or the call fails.
    pub fn stats_std_dev(&self, data: &[f64]) -> Result<f64, IpcError> {
        barracuda::stats_std_dev(
            self.router.require(capabilities::STATS_STD_DEV)?,
            data,
            self.timeout,
        )
    }

    /// `stats.weighted_mean` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn stats_weighted_mean(&self, data: &[f64], weights: &[f64]) -> Result<f64, IpcError> {
        barracuda::stats_weighted_mean(
            self.router.require(capabilities::STATS_WEIGHTED_MEAN)?,
            data,
            weights,
            self.timeout,
        )
    }

    /// `tensor.matmul` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn tensor_matmul(
        &self,
        a: &[f64],
        b: &[f64],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f64>, IpcError> {
        barracuda::tensor_matmul(
            self.router.require(capabilities::TENSOR_MATMUL)?,
            a,
            b,
            rows_a,
            cols_a,
            cols_b,
            self.timeout,
        )
    }

    /// `tensor.create` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn tensor_create(
        &self,
        shape: &[usize],
        fill: &str,
    ) -> Result<serde_json::Value, IpcError> {
        barracuda::tensor_create(
            self.router.require(capabilities::TENSOR_CREATE)?,
            shape,
            fill,
            self.timeout,
        )
    }

    /// `barracuda.precision.route` — query optimal precision strategy
    /// for a domain operation.
    ///
    /// Returns the recommended precision tier (f32/f64/DF64/mixed),
    /// FMA safety, sovereign compiler requirement, and optional rationale.
    ///
    /// # Errors
    ///
    /// Returns an error if barraCuda is not discovered or the call fails.
    pub fn precision_route(
        &self,
        domain: &str,
        hardware_hint: Option<&str>,
    ) -> Result<barracuda::PrecisionRouteResult, IpcError> {
        barracuda::precision_route(
            self.router.require(capabilities::PRECISION_ROUTE)?,
            domain,
            hardware_hint,
            self.timeout,
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // toadStool surface (routed via compute.dispatch)
    // ═══════════════════════════════════════════════════════════════

    /// `compute.dispatch` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn compute_dispatch(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, IpcError> {
        toadstool::compute_dispatch(
            self.router.require(capabilities::COMPUTE_DISPATCH)?,
            params,
            self.timeout,
        )
    }

    /// `node.compute` — composed compute pipeline via signal dispatch.
    ///
    /// Sends a `node.compute` signal through biomeOS, which decomposes
    /// into: `toadStool.compute.dispatch → coralReef.shader.compile.wgsl → barraCuda.tensor.matmul`.
    /// biomeOS manages the graph sequencing and error handling.
    ///
    /// Prefer this over [`compute_dispatch`](Self::compute_dispatch) when
    /// running inside a biomeOS composition for full pipeline orchestration.
    ///
    /// # Errors
    ///
    /// Returns an error if signal dispatch fails or biomeOS is unavailable.
    #[cfg(feature = "primalspring")]
    pub fn dispatch_compute_signal(
        ctx: &mut primalspring::composition::CompositionContext,
        workload: &serde_json::Value,
        shader: Option<&str>,
    ) -> Result<serde_json::Value, IpcError> {
        let mut params = serde_json::json!({ "workload": workload });
        if let Some(s) = shader {
            params["shader"] = serde_json::Value::String(s.to_owned());
        }
        ctx.dispatch("node.compute", &params)
            .map_err(|e| IpcError::Other(format!("node.compute dispatch: {e}")))
    }

    /// `toadstool.validate` — Tier 2 workload pre-flight validation.
    ///
    /// Validates a workload TOML before dispatch. Returns a structured
    /// result with validity, GPU availability, precision tier, estimated
    /// dispatch time, warnings, and required capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if toadStool is not discovered or the call fails.
    pub fn validate_workload(
        &self,
        workload_path: &str,
        dry_run: bool,
    ) -> Result<toadstool::ValidateResult, IpcError> {
        toadstool::validate(
            self.router.require(capabilities::TOADSTOOL_VALIDATE)?,
            workload_path,
            dry_run,
            self.timeout,
        )
    }

    /// `toadstool.list_workloads` — list available workloads.
    ///
    /// # Errors
    ///
    /// Returns an error if toadStool is not discovered or the call fails.
    pub fn list_workloads(&self) -> Result<serde_json::Value, IpcError> {
        toadstool::list_workloads(
            self.router
                .require(capabilities::TOADSTOOL_LIST_WORKLOADS)?,
            self.timeout,
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // BearDog surface (routed via crypto.hash)
    // ═══════════════════════════════════════════════════════════════

    /// `crypto.hash` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn crypto_hash(&self, algorithm: &str, data: &str) -> Result<String, IpcError> {
        beardog::crypto_hash(
            self.router.require(capabilities::CRYPTO_HASH)?,
            algorithm,
            data,
            self.timeout,
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // Squirrel surface (routed via inference.*)
    // ═══════════════════════════════════════════════════════════════

    /// `inference.complete` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn inference_complete(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, IpcError> {
        squirrel::inference_complete(
            self.router.require(capabilities::INFERENCE_COMPLETE)?,
            params,
            self.timeout,
        )
    }

    /// `inference.embed` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn inference_embed(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, IpcError> {
        squirrel::inference_embed(
            self.router.require(capabilities::INFERENCE_EMBED)?,
            params,
            self.timeout,
        )
    }

    /// `inference.models` — list available models via Squirrel.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn inference_models(&self) -> Result<serde_json::Value, IpcError> {
        squirrel::inference_models(
            self.router.require(capabilities::INFERENCE_MODELS)?,
            self.timeout,
        )
    }

    /// Register this spring as an inference provider with Squirrel.
    ///
    /// After registration, Squirrel routes matching inference requests
    /// back to `socket_path` for this provider.
    ///
    /// # Errors
    ///
    /// Returns an error if Squirrel is not discovered or registration fails.
    pub fn register_as_provider(
        &self,
        provider_id: &str,
        socket_path: &str,
        supported_capabilities: &[&str],
    ) -> Result<serde_json::Value, IpcError> {
        squirrel::register_provider(
            self.router
                .require(capabilities::INFERENCE_REGISTER_PROVIDER)?,
            provider_id,
            socket_path,
            supported_capabilities,
            self.timeout,
        )
    }

    /// Unregister this spring as an inference provider from Squirrel.
    ///
    /// # Errors
    ///
    /// Returns an error if Squirrel is not discovered or unregistration fails.
    pub fn unregister_provider(&self, provider_id: &str) -> Result<serde_json::Value, IpcError> {
        squirrel::unregister_provider(
            self.router
                .require(capabilities::INFERENCE_UNREGISTER_PROVIDER)?,
            provider_id,
            self.timeout,
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // coralReef surface (routed via shader.compile.*)
    // ═══════════════════════════════════════════════════════════════

    /// `shader.compile.wgsl` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn shader_compile_wgsl(
        &self,
        source: &str,
        label: &str,
    ) -> Result<serde_json::Value, IpcError> {
        coralreef::shader_compile_wgsl(
            self.router.require(capabilities::SHADER_COMPILE_WGSL)?,
            source,
            label,
            self.timeout,
        )
    }

    /// `shader.compile.capabilities` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn shader_capabilities(&self) -> Result<serde_json::Value, IpcError> {
        coralreef::shader_capabilities(
            self.router
                .require(capabilities::SHADER_COMPILE_CAPABILITIES)?,
            self.timeout,
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // skunkBat surface (routed via security.audit_log)
    // ═══════════════════════════════════════════════════════════════

    /// `security.audit_log` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn audit_log(
        &self,
        event_type: &str,
        source: &str,
        payload: &serde_json::Value,
    ) -> Result<serde_json::Value, IpcError> {
        skunkbat::audit_log(
            self.router.require(capabilities::SECURITY_AUDIT_LOG)?,
            event_type,
            source,
            payload,
            self.timeout,
        )
    }

    /// Whether the skunkBat primal was discovered.
    #[must_use]
    pub fn has_skunkbat(&self) -> bool {
        self.router.get(capabilities::SECURITY_AUDIT_LOG).is_some()
    }

    /// Whether the `NestGate` primal was discovered.
    #[must_use]
    pub fn has_nestgate(&self) -> bool {
        self.router.get(capabilities::CONTENT_PUT).is_some()
    }

    // ═══════════════════════════════════════════════════════════════
    // NestGate surface (routed via content.*)
    // ═══════════════════════════════════════════════════════════════

    /// `content.put` — store content-addressed data via `NestGate`.
    ///
    /// Returns the BLAKE3 hash and metadata on success.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `content.put` or the call fails.
    pub fn content_put(
        &self,
        data_base64: &str,
        content_type: Option<&str>,
    ) -> Result<nestgate::ContentPutResult, IpcError> {
        nestgate::content_put(
            self.router.require(capabilities::CONTENT_PUT)?,
            data_base64,
            content_type,
            self.timeout,
        )
    }

    /// `content.get` — retrieve content-addressed data by BLAKE3 hash.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `content.get` or the call fails.
    pub fn content_get(&self, hash: &str) -> Result<nestgate::ContentGetResult, IpcError> {
        nestgate::content_get(
            self.router.require(capabilities::CONTENT_GET)?,
            hash,
            self.timeout,
        )
    }

    /// `content.exists` — check whether content-addressed data exists.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `content.exists` or the call fails.
    pub fn content_exists(&self, hash: &str) -> Result<bool, IpcError> {
        nestgate::content_exists(
            self.router.require(capabilities::CONTENT_EXISTS)?,
            hash,
            self.timeout,
        )
    }

    /// Number of unique primals discovered.
    #[must_use]
    pub fn discovered_count(&self) -> usize {
        self.router.discovered_primals().len()
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::error::IpcError;
    use crate::ipc::health::PrimalSlot;
    use serial_test::serial;

    #[test]
    #[serial]
    fn discover_returns_none_for_absent_primals() {
        let client = IpcMathClient::discover();
        let report = client.probe_all();
        assert_eq!(report.alive_count(), 0);
        assert!(!report.is_alive(PrimalSlot::Barracuda));
    }

    #[test]
    fn client_require_returns_err_for_missing_primals() {
        let client = IpcMathClient::discover();
        assert!(client.stats_mean(&[1.0]).is_err());
        assert!(client.crypto_hash("blake3", "test").is_err());
        assert!(client.compute_dispatch(&serde_json::json!({})).is_err());
        assert!(client.inference_complete(&serde_json::json!({})).is_err());
        assert!(client.shader_capabilities().is_err());
        assert!(
            client
                .audit_log("test", "neuralspring", &serde_json::json!({}))
                .is_err()
        );
        assert!(client.content_put("dGVzdA==", None).is_err());
        assert!(client.content_get("deadbeef").is_err());
        assert!(client.content_exists("deadbeef").is_err());
        assert!(client.validate_workload("/tmp/test.toml", true).is_err());
        assert!(client.list_workloads().is_err());
        assert!(client.precision_route("lattice_qcd", None).is_err());
    }

    #[test]
    fn with_timeout_overrides_default() {
        let client = IpcMathClient::discover().with_timeout(Duration::from_millis(500));
        assert_eq!(client.timeout, Duration::from_millis(500));
    }

    #[test]
    #[serial]
    fn has_coralreef_false_when_absent() {
        let client = IpcMathClient::discover();
        assert!(!client.has_coralreef());
    }

    #[test]
    #[serial]
    fn has_nestgate_false_when_absent() {
        let client = IpcMathClient::discover();
        assert!(!client.has_nestgate());
    }

    #[test]
    #[serial]
    fn not_discovered_error_for_missing_barracuda() {
        let client = IpcMathClient::discover();
        let err = client.stats_mean(&[1.0, 2.0]).expect_err("not discovered");
        assert!(matches!(err, IpcError::NotDiscovered { .. }));
        assert!(err.to_string().contains("barracuda"));
    }

    #[test]
    #[serial]
    fn client_discovered_count_zero_without_primals() {
        let client = IpcMathClient::discover();
        assert_eq!(client.discovered_count(), 0);
    }

    #[test]
    #[serial]
    fn has_primal_flags_false_when_absent() {
        let client = IpcMathClient::discover();
        assert!(!client.has_barracuda());
        assert!(!client.has_toadstool());
        assert!(!client.has_beardog());
        assert!(!client.has_squirrel());
        assert!(!client.has_skunkbat());
        assert!(!client.has_coralreef());
        assert!(!client.has_nestgate());
    }

    #[test]
    fn client_additional_methods_err_when_not_discovered() {
        let client = IpcMathClient::discover();
        assert!(client.stats_std_dev(&[1.0]).is_err());
        assert!(
            client
                .stats_weighted_mean(&[1.0, 2.0], &[0.5, 0.5])
                .is_err()
        );
        assert!(client.tensor_matmul(&[1.0], &[1.0], 1, 1, 1).is_err());
        assert!(client.tensor_create(&[2, 2], "zeros").is_err());
        assert!(client.inference_embed(&serde_json::json!({})).is_err());
        assert!(client.inference_models().is_err());
        assert!(
            client
                .register_as_provider("ns", "/tmp/x.sock", &["inference.complete"])
                .is_err()
        );
        assert!(client.unregister_provider("ns").is_err());
        assert!(
            client
                .shader_compile_wgsl("@compute fn main() {}", "t")
                .is_err()
        );
    }

    fn mock_rpc_socket(response_body: &str) -> (std::path::PathBuf, std::thread::JoinHandle<()>) {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixListener;

        let dir = std::env::temp_dir().join(format!(
            "ns_ipc_rpc_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let sock_path = dir.join("mock.sock");
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("bind mock socket");
        let response = format!("{response_body}\n");
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let mut reader = BufReader::new(&stream);
                let mut line = String::new();
                reader.read_line(&mut line).ok();
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });
        (sock_path, handle)
    }

    fn cleanup_mock_socket(sock_path: &std::path::Path) {
        std::fs::remove_file(sock_path).ok();
        if let Some(parent) = sock_path.parent() {
            std::fs::remove_dir(parent).ok();
        }
    }

    #[test]
    #[serial]
    fn orchestrator_override_marks_primals_discovered() {
        let dir = std::env::temp_dir().join(format!(
            "ns_ipc_orch_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let sock = dir.join("orchestrator.sock");
        std::fs::write(&sock, b"").expect("touch");

        temp_env::with_var(
            crate::config::ENV_BIOMEOS_ORCHESTRATOR,
            Some(sock.to_string_lossy().as_ref()),
            || {
                let client = IpcMathClient::discover();
                assert!(client.has_barracuda());
                assert!(client.has_toadstool());
                assert_eq!(client.discovered_count(), 1);
            },
        );

        std::fs::remove_file(&sock).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[serial]
    fn discovered_fake_file_returns_transport_error() {
        let dir = std::env::temp_dir().join(format!(
            "ns_ipc_fake_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let fake = dir.join("not-a-socket");
        std::fs::write(&fake, b"not a unix socket").expect("write fake");

        temp_env::with_var(
            crate::config::ENV_BIOMEOS_ORCHESTRATOR,
            Some(fake.to_string_lossy().as_ref()),
            || {
                let client = IpcMathClient::discover().with_timeout(Duration::from_millis(200));
                let err = client
                    .stats_mean(&[1.0, 2.0, 3.0])
                    .expect_err("connect fails");
                assert!(matches!(err, IpcError::Transport { .. }));
            },
        );

        std::fs::remove_file(&fake).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[serial]
    fn mock_socket_rpc_error_propagates_through_client() {
        let (sock, handle) = mock_rpc_socket(
            r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"internal"},"id":1}"#,
        );
        temp_env::with_var(
            crate::config::ENV_BIOMEOS_ORCHESTRATOR,
            Some(sock.to_string_lossy().as_ref()),
            || {
                std::thread::yield_now();
                let client = IpcMathClient::discover().with_timeout(Duration::from_secs(5));
                let err = client
                    .stats_mean(&[1.0, 2.0])
                    .expect_err("rpc error propagates");
                assert!(matches!(err, IpcError::Protocol { .. }));
                assert!(err.to_string().contains("RPC error"));
            },
        );
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    #[serial]
    fn mock_socket_missing_numeric_result_returns_protocol_error() {
        let (sock, handle) =
            mock_rpc_socket(r#"{"jsonrpc":"2.0","result":{"status":"ok"},"id":1}"#);
        temp_env::with_var(
            crate::config::ENV_BIOMEOS_ORCHESTRATOR,
            Some(sock.to_string_lossy().as_ref()),
            || {
                std::thread::yield_now();
                let client = IpcMathClient::discover().with_timeout(Duration::from_secs(5));
                let err = client.stats_mean(&[1.0]).expect_err("missing mean field");
                assert!(matches!(err, IpcError::Protocol { .. }));
                assert!(err.to_string().contains("missing numeric result"));
            },
        );
        handle.join().expect("mock thread");
        cleanup_mock_socket(&sock);
    }

    #[test]
    #[serial]
    fn probe_all_reports_dead_for_non_listening_socket() {
        let dir = std::env::temp_dir().join(format!(
            "ns_ipc_dead_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let fake = dir.join("dead.sock");
        std::fs::write(&fake, b"").expect("touch");

        temp_env::with_var(
            crate::config::ENV_BIOMEOS_ORCHESTRATOR,
            Some(fake.to_string_lossy().as_ref()),
            || {
                let client = IpcMathClient::discover().with_timeout(Duration::from_millis(100));
                let report = client.probe_all();
                assert_eq!(report.alive_count(), 0);
                assert!(!report.is_alive(PrimalSlot::Barracuda));
            },
        );

        std::fs::remove_file(&fake).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
