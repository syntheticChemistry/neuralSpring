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
//!   7. Local IPC roundtrip (render, append, gauge, replace)
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

/// Accept one JSON-RPC request on a local validation socket and reply "ok".
///
/// This is a real (not mocked) IPC roundtrip over a Unix socket — the
/// validator creates a local listener, the `PetalTonguePushClient` connects,
/// and this function verifies the request payload matches the expected schema.
fn accept_and_reply(listener: &UnixListener) -> serde_json::Value {
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
    // 6. HMM study
    // ═══════════════════════════════════════════════════════════════════
    let (hmm, hmm_edges) = visualization::hmm_study();
    h.check_abs(
        "hmm.channels > 0",
        if count_channels(&hmm) > 0 { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "hmm.has_heatmap",
        if has_channel_type(&hmm, |c| matches!(c, DataChannel::Heatmap { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "hmm.has_edges",
        if hmm_edges.is_empty() { 0.0 } else { 1.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 7. Game theory study
    // ═══════════════════════════════════════════════════════════════════
    let (gt, gt_edges) = visualization::game_theory_study();
    h.check_abs(
        "game_theory.channels > 0",
        if count_channels(&gt) > 0 { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "game_theory.has_heatmap",
        if has_channel_type(&gt, |c| matches!(c, DataChannel::Heatmap { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "game_theory.has_gauge",
        if has_channel_type(&gt, |c| matches!(c, DataChannel::Gauge { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "game_theory.has_edges",
        if gt_edges.is_empty() { 0.0 } else { 1.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 8. WDM study
    // ═══════════════════════════════════════════════════════════════════
    let (wdm, wdm_edges) = visualization::wdm_study();
    h.check_abs(
        "wdm.channels > 0",
        if count_channels(&wdm) > 0 { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "wdm.has_scatter3d",
        if has_channel_type(&wdm, |c| matches!(c, DataChannel::Scatter3D { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "wdm.has_edges",
        if wdm_edges.is_empty() { 0.0 } else { 1.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 9. Glucose study
    // ═══════════════════════════════════════════════════════════════════
    let (glucose, _glucose_edges) = visualization::glucose_study();
    h.check_abs(
        "glucose.channels > 0",
        if count_channels(&glucose) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "glucose.has_distribution",
        if has_channel_type(&glucose, |c| matches!(c, DataChannel::Distribution { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "glucose.has_gauge",
        if has_channel_type(&glucose, |c| matches!(c, DataChannel::Gauge { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 10. Immunological study
    // ═══════════════════════════════════════════════════════════════════
    let (immuno, _immuno_edges) = visualization::immunological_study();
    h.check_abs(
        "immuno.channels > 0",
        if count_channels(&immuno) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "immuno.has_spectrum",
        if has_channel_type(&immuno, |c| matches!(c, DataChannel::Spectrum { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "immuno.has_distribution",
        if has_channel_type(&immuno, |c| matches!(c, DataChannel::Distribution { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 11. Population study
    // ═══════════════════════════════════════════════════════════════════
    let (pop, pop_edges) = visualization::population_study();
    h.check_abs(
        "population.channels > 0",
        if count_channels(&pop) > 0 { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "population.has_heatmap",
        if has_channel_type(&pop, |c| matches!(c, DataChannel::Heatmap { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "population.has_scatter3d",
        if has_channel_type(&pop, |c| matches!(c, DataChannel::Scatter3D { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "population.has_edges",
        if pop_edges.is_empty() { 0.0 } else { 1.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 12. Loss landscape study
    // ═══════════════════════════════════════════════════════════════════
    let (landscape, landscape_edges) = visualization::loss_landscape_study();
    h.check_abs(
        "landscape.channels > 0",
        if count_channels(&landscape) > 0 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "landscape.has_fieldmap",
        if has_channel_type(&landscape, |c| matches!(c, DataChannel::FieldMap { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "landscape.has_spectrum",
        if has_channel_type(&landscape, |c| matches!(c, DataChannel::Spectrum { .. })) {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "landscape.has_edges",
        if landscape_edges.is_empty() { 0.0 } else { 1.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 13. Full study combiner (all 12 tracks)
    // ═══════════════════════════════════════════════════════════════════
    let (full, full_edges) = visualization::full_study();
    let full_channels = count_channels(&full);
    h.check_abs(
        "full.channels >= 20",
        if full_channels >= 20 { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "full.primals >= 12",
        if full.ecosystem.primals.len() >= 12 {
            1.0
        } else {
            0.0
        },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "full.edges >= 10",
        if full_edges.len() >= 10 { 1.0 } else { 0.0 },
        1.0,
        tolerances::BOOLEAN_VALIDATION_SLACK,
    );
    h.check_abs(
        "full.all_8_channel_types",
        if has_channel_type(&full, |c| matches!(c, DataChannel::Heatmap { .. }))
            && has_channel_type(&full, |c| matches!(c, DataChannel::Distribution { .. }))
            && has_channel_type(&full, |c| matches!(c, DataChannel::FieldMap { .. }))
            && has_channel_type(&full, |c| matches!(c, DataChannel::Scatter3D { .. }))
            && has_channel_type(&full, |c| matches!(c, DataChannel::Spectrum { .. }))
            && has_channel_type(&full, |c| matches!(c, DataChannel::TimeSeries { .. }))
            && has_channel_type(&full, |c| matches!(c, DataChannel::Gauge { .. }))
            && has_channel_type(&full, |c| matches!(c, DataChannel::Bar { .. }))
        {
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
    h.check_abs(
        "stats.error_rate",
        stats.error_rate(),
        5.0 / 55.0,
        tolerances::CROSS_LANGUAGE,
    );

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
    let client =
        PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent_ns_pt_test.sock"));
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

    let handle = std::thread::spawn(move || accept_and_reply(&listener));
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

    let handle = std::thread::spawn(move || accept_and_reply(&listener));
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

    let handle = std::thread::spawn(move || accept_and_reply(&listener));
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
        accept_and_reply(&listener);
        accept_and_reply(&listener)
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
        PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent_ns_pt_validate.sock"));
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
