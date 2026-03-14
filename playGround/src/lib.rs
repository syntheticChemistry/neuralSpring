// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring playGround — application sandbox for Squirrel MCP integration,
//! interactive experiment running, model inference, and primal IPC.
//!
//! This crate provides:
//! - [`ipc_client`]: Reusable JSON-RPC 2.0 client over Unix domain sockets
//! - [`squirrel_client`]: Typed Squirrel MCP client (ai.query, tool.execute)
//! - [`primal_client`]: Typed neuralSpring primal client (science.* capabilities)
//! - [`mcp_tools`]: MCP tool definitions for all 14 science capabilities
//! - [`secrets`]: API key loading from testing-secrets
//! - [`hf_hub`]: `HuggingFace` Hub download client
//! - [`model_config`]: HF model config parser
//! - [`inference`]: GPU inference via barraCuda shaders

#![forbid(unsafe_code)]
#![expect(
    clippy::missing_errors_doc,
    reason = "playground — evolving API surface"
)]

pub mod hf_hub;
pub mod inference;
pub mod ipc_client;
pub mod mcp_tools;
pub mod model_config;
pub mod primal_client;
pub mod secrets;
pub mod squirrel_client;
