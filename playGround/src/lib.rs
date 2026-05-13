// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring playGround — application sandbox for primal IPC, model
//! inference, and the ecoPrimals compute triangle (`coralReef` → `ToadStool` →
//! `barraCuda`).
//!
//! This crate provides:
//! - [`ipc_client`]: Reusable JSON-RPC 2.0 client over Unix domain sockets
//! - [`biomeos_client`]: Typed biomeOS orchestrator client (nucleus.*, capability.*)
//! - [`squirrel_client`]: Typed Squirrel MCP client (ai.query, tool.execute)
//! - [`primal_client`]: Typed neuralSpring primal client (science.* capabilities)
//! - [`toadstool_client`]: Typed `ToadStool` compute client (compute.submit, gpu.dispatch)
//! - [`coralreef_client`]: Typed coralReef shader compiler client (shader.compile.wgsl)
//! - [`mcp_tools`]: MCP tool definitions for all 14 science capabilities
//! - [`secrets`]: API key loading from testing-secrets
//! - [`songbird_http`]: Tower Atomic HTTP via Songbird IPC (zero C deps)
//! - [`hf_hub`]: `HuggingFace` Hub download client (via Songbird)
//! - [`model_config`]: HF model config parser
//! - [`inference`]: GPU inference via barraCuda shaders

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![expect(
    clippy::missing_errors_doc,
    reason = "playground — evolving API surface"
)]

pub mod biomeos_client;
pub mod coralreef_client;
pub mod discovery;
pub mod hf_hub;
pub mod inference;
pub mod ipc_client;
pub mod mcp_tools;
pub mod model_config;
pub mod primal_client;
pub mod secrets;
pub mod songbird_http;
pub mod squirrel_client;
pub mod toadstool_client;
