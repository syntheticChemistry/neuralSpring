// SPDX-License-Identifier: AGPL-3.0-or-later

//! biomeOS lifecycle management: registration, deregistration, heartbeat.

use std::path::PathBuf;

use super::discovery::{forward_to_primal_raw, resolve_socket_dir};
use super::{heartbeat_interval_secs, orchestrator_socket, ALL_CAPABILITIES, PRIMAL_NAME};

pub async fn register_with_biomeos(our_socket: &std::path::Path) {
    let biomeos_socket = resolve_socket_dir().join(orchestrator_socket());
    if !biomeos_socket.exists() {
        log::info!(
            "No biomeOS orchestrator at {}, running standalone",
            biomeos_socket.display()
        );
        return;
    }

    let reg_result = forward_to_primal_raw(
        &biomeos_socket,
        "nucleus.register",
        &serde_json::json!({
            "name": PRIMAL_NAME,
            "socket_path": our_socket.to_string_lossy(),
            "pid": std::process::id(),
        }),
    )
    .await;

    match reg_result {
        Ok(_) => log::info!("Registered with biomeOS NUCLEUS"),
        Err(e) => log::warn!("nucleus.register failed (non-fatal): {e}"),
    }

    for cap in ALL_CAPABILITIES {
        let cap_result = forward_to_primal_raw(
            &biomeos_socket,
            "capability.register",
            &serde_json::json!({
                "primal": PRIMAL_NAME,
                "capability": cap,
                "socket_path": our_socket.to_string_lossy(),
            }),
        )
        .await;

        if let Err(e) = cap_result {
            log::warn!("capability.register({cap}) failed (non-fatal): {e}");
        }
    }

    log::info!(
        "All {} capabilities registered with biomeOS",
        ALL_CAPABILITIES.len()
    );
}

pub async fn deregister_from_nucleus(our_socket: &std::path::Path) {
    let biomeos_socket = resolve_socket_dir().join(orchestrator_socket());
    if !biomeos_socket.exists() {
        return;
    }
    let _ = forward_to_primal_raw(
        &biomeos_socket,
        "nucleus.deregister",
        &serde_json::json!({
            "name": PRIMAL_NAME,
            "socket_path": our_socket.to_string_lossy(),
        }),
    )
    .await;
    log::info!("Deregistered from biomeOS NUCLEUS");
}

pub async fn heartbeat_loop(our_socket: PathBuf) {
    let biomeos_socket = resolve_socket_dir().join(orchestrator_socket());
    let mut interval =
        tokio::time::interval(std::time::Duration::from_secs(heartbeat_interval_secs()));

    loop {
        interval.tick().await;

        if !biomeos_socket.exists() {
            continue;
        }

        let _ = forward_to_primal_raw(
            &biomeos_socket,
            "nucleus.heartbeat",
            &serde_json::json!({
                "name": PRIMAL_NAME,
                "socket_path": our_socket.to_string_lossy(),
                "status": "healthy",
            }),
        )
        .await;
    }
}
