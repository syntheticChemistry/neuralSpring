// SPDX-License-Identifier: AGPL-3.0-or-later

//! petalTongue Scenario and Streaming Validator
//!
//! Validates the full petalTongue visualization integration:
//!   1. Scenario builders produce valid structures
//!   2. `DataChannel` variants exercised
//!   3. `full_study()` combines all sub-studies
//!   4. JSON serialization round-trips
//!   5. IPC client param builders produce correct JSON-RPC payloads
//!   6. `StreamSession` lifecycle and statistics
//!   7. Mock socket roundtrip (render, append, gauge, replace)
//!   8. Scenario edge graph structure
//!
//! ## Provenance
//!
//! Validation class: Infrastructure (visualization layer)
//! No Python baseline — validates schema, serialization, and IPC contracts.

#![expect(
    clippy::too_many_lines,
    clippy::expect_used,
    reason = "validation binary — comprehensive visualization integration tests"
)]

use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::visualization::{
    self, DataChannel, NeuralScenario, PetalTonguePushClient, SessionStats, StreamSession,
};

fn mock_response(listener: &UnixListener) -> serde_json::Value {
    let (mut stream, _) = listener.accept().expect("accept");
    let mut buf = vec![0u8; 65536];
    let n = stream.read(&mut buf).expect("read");
    let request: serde_json::Value = serde_json::from_slice(&buf[..n]).expect("parse");
    let response = serde_json::json!({"jsonrpc": "2.0", "result": "ok", "id": 1});
    stream
        .write_all(serde_json::to_vec(&response).expect("ser").as_slice())
        .expect("write");
    request
}

fn test_socket(name: &str) -> (PathBuf, UnixListener) {
    let dir = std::env::temp_dir().join(format!(
        "ns_pt_{name}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("{name}.sock"));
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind");
    (path, listener)
}

fn cleanup_socket(path: &std::path::Path) {
    std::fs::remove_file(path).ok();
    if let Some(parent) = path.parent() {
        std::fs::remove_dir(parent).ok();
    }
}

fn count_channels(scenario: &NeuralScenario) -> usize {
    scenario
        .ecosystem
        .primals
        .iter()
        .flat_map(|p| p.data_channels.iter())
        .count()
}

fn has_channel_type(scenario: &NeuralScenario, check: fn(&DataChannel) -> bool) -> bool {
    scenario
        .ecosystem
        .primals
        .iter()
        .flat_map(|p| p.data_channels.iter())
        .any(check)
}

fn main() {
    let mut h = ValidationHarness::new("petalTongue Visualization Integration");

    // ═══════════════════════════════════════════════════════════════════
    // 1. Spectral study
    // ═══════════════════════════════════════════════════════════════════
    let (spectral, spectral_edges) = visualization::spectral_study();
    h.check_abs(
        "spectral.channels > 0",
        if count_channels(&spectral) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "spectral.has_edges",
        if spectral_edges.is_empty() { 0.0 } else { 1.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "spectral.has_timeseries",
        if has_channel_type(&spectral, |c| matches!(c, DataChannel::TimeSeries { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 2. Training study
    // ═══════════════════════════════════════════════════════════════════
    let (training, _training_edges) = visualization::training_study();
    h.check_abs(
        "training.channels > 0",
        if count_channels(&training) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 3. Coordination study
    // ═══════════════════════════════════════════════════════════════════
    let (coordination, _coord_edges) = visualization::coordination_study();
    h.check_abs(
        "coordination.channels > 0",
        if count_channels(&coordination) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 4. Folding study
    // ═══════════════════════════════════════════════════════════════════
    let (folding, _folding_edges) = visualization::folding_study();
    h.check_abs(
        "folding.channels > 0",
        if count_channels(&folding) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "folding.has_bar",
        if has_channel_type(&folding, |c| matches!(c, DataChannel::Bar { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 5. Provenance study
    // ═══════════════════════════════════════════════════════════════════
    let (provenance, _prov_edges) = visualization::provenance_study();
    h.check_abs(
        "provenance.channels > 0",
        if count_channels(&provenance) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 6. Full study combiner
    // ═══════════════════════════════════════════════════════════════════
    let (full, full_edges) = visualization::full_study();
    let full_channels = count_channels(&full);
    h.check_abs(
        "full.channels >= 5",
        if full_channels >= 5 { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "full.primals > 1",
        if full.ecosystem.primals.len() > 1 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "full.edges_combined",
        if full_edges.len() >= spectral_edges.len() {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 7. Scenario with edges JSON structure
    // ═══════════════════════════════════════════════════════════════════
    let graph_json_str = visualization::scenario_with_edges_json(&spectral, &spectral_edges);
    let graph_json: serde_json::Value =
        serde_json::from_str(&graph_json_str).unwrap_or(serde_json::Value::Null);
    h.check_abs(
        "edges.json_valid",
        if graph_json.is_object() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "edges.has_edges_field",
        if graph_json
            .get("edges")
            .is_some_and(serde_json::Value::is_array)
        {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 8. Session statistics (unit)
    // ═══════════════════════════════════════════════════════════════════
    let stats = SessionStats {
        messages_sent: 50,
        bytes_sent: 4000,
        errors: 5,
        uptime_ms: 2000,
    };
    h.check_abs(
        "stats.mps",
        stats.messages_per_second(),
        25.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs("stats.error_rate", stats.error_rate(), 5.0 / 55.0, 1e-10);

    let zero_stats = SessionStats {
        messages_sent: 0,
        bytes_sent: 0,
        errors: 0,
        uptime_ms: 0,
    };
    h.check_abs(
        "stats.zero_mps",
        zero_stats.messages_per_second(),
        0.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 9. StreamSession resume (no socket needed)
    // ═══════════════════════════════════════════════════════════════════
    let client = PetalTonguePushClient::new(PathBuf::from("/tmp/nonexistent_ns_pt_test.sock"));
    let session = StreamSession::resume(client, "validate-pt-session");
    h.check_abs(
        "session.id",
        if session.session_id() == "validate-pt-session" {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "session.no_backpressure",
        if session.backpressure_active() {
            0.0
        } else {
            1.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    let initial_stats = session.stats();
    h.check_abs(
        "session.initial_messages",
        f64::from(u32::try_from(initial_stats.messages_sent).unwrap_or(u32::MAX)),
        0.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 10. Mock socket: push_render roundtrip
    // ═══════════════════════════════════════════════════════════════════
    let (sock_path, listener) = test_socket("pt_render");
    let pt_client = PetalTonguePushClient::new(sock_path.clone());

    let handle = std::thread::spawn(move || mock_response(&listener));
    let render_result = pt_client.push_render("sess-v1", "Validate Render", &spectral);
    let request = handle.join().expect("mock thread");

    h.check_abs(
        "mock.render_ok",
        if render_result.is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "mock.render_method",
        if request.get("method").and_then(|m| m.as_str()) == Some("visualization.render") {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "mock.render_domain",
        if request
            .get("params")
            .and_then(|p| p.get("domain"))
            .and_then(|d| d.as_str())
            == Some("neural")
        {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    cleanup_socket(&sock_path);

    // ═══════════════════════════════════════════════════════════════════
    // 11. Mock socket: push_append roundtrip
    // ═══════════════════════════════════════════════════════════════════
    let (sock_path, listener) = test_socket("pt_append");
    let pt_client = PetalTonguePushClient::new(sock_path.clone());

    let handle = std::thread::spawn(move || mock_response(&listener));
    let append_result = pt_client.push_append("sess-v2", "series-1", &[1.0, 2.0], &[10.0, 20.0]);
    let request = handle.join().expect("mock thread");

    h.check_abs(
        "mock.append_ok",
        if append_result.is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "mock.append_op_type",
        if request
            .get("params")
            .and_then(|p| p.get("operation"))
            .and_then(|o| o.get("type"))
            .and_then(|t| t.as_str())
            == Some("append")
        {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    cleanup_socket(&sock_path);

    // ═══════════════════════════════════════════════════════════════════
    // 12. Mock socket: push_replace roundtrip
    // ═══════════════════════════════════════════════════════════════════
    let (sock_path, listener) = test_socket("pt_replace");
    let pt_client = PetalTonguePushClient::new(sock_path.clone());

    let handle = std::thread::spawn(move || mock_response(&listener));
    let replace_data = serde_json::json!({"matrix": [[1, 2], [3, 4]]});
    let replace_result = pt_client.push_replace("sess-v3", "heatmap-1", &replace_data);
    let request = handle.join().expect("mock thread");

    h.check_abs(
        "mock.replace_ok",
        if replace_result.is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "mock.replace_op_type",
        if request
            .get("params")
            .and_then(|p| p.get("operation"))
            .and_then(|o| o.get("type"))
            .and_then(|t| t.as_str())
            == Some("replace")
        {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    cleanup_socket(&sock_path);

    // ═══════════════════════════════════════════════════════════════════
    // 13. Mock socket: StreamSession start + append
    // ═══════════════════════════════════════════════════════════════════
    let spectral2 = visualization::spectral_study().0;
    let (sock_path, listener) = test_socket("pt_stream");
    let pt_client = PetalTonguePushClient::new(sock_path.clone());

    let handle = std::thread::spawn(move || {
        mock_response(&listener);
        mock_response(&listener)
    });
    let session = StreamSession::start(pt_client, "stream-v1", "Stream Test", &spectral2)
        .expect("session start");
    let append_result = session.append("series-1", &[1.0], &[0.25]);
    let _requests = handle.join().expect("mock thread");

    h.check_abs(
        "stream.start_ok",
        1.0,
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "stream.append_ok",
        if append_result.is_ok() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    let stream_stats = session.stats();
    h.check_abs(
        "stream.messages_sent",
        f64::from(u32::try_from(stream_stats.messages_sent).unwrap_or(u32::MAX)),
        2.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    cleanup_socket(&sock_path);

    // ═══════════════════════════════════════════════════════════════════
    // 14. Connection failure tracking
    // ═══════════════════════════════════════════════════════════════════
    let bad_client =
        PetalTonguePushClient::new(PathBuf::from("/tmp/nonexistent_ns_pt_validate.sock"));
    let bad_session = StreamSession::resume(bad_client, "bad-session");
    let fail_result = bad_session.append("x", &[1.0], &[2.0]);
    h.check_abs(
        "fail.append_errors",
        if fail_result.is_err() { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    let fail_stats = bad_session.stats();
    h.check_abs(
        "fail.error_count",
        f64::from(u32::try_from(fail_stats.errors).unwrap_or(u32::MAX)),
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    h.finish();
}
