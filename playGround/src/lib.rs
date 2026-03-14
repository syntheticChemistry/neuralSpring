// SPDX-License-Identifier: AGPL-3.0-or-later

//! neuralSpring playGround — application sandbox for Squirrel MCP integration,
//! interactive experiment running, and primal IPC.
//!
//! This crate provides:
//! - [`ipc_client`]: Reusable JSON-RPC 2.0 client over Unix domain sockets
//! - [`squirrel_client`]: Typed Squirrel MCP client (ai.query, tool.execute)
//! - [`primal_client`]: Typed neuralSpring primal client (science.* capabilities)
//! - [`mcp_tools`]: MCP tool definitions for all 14 science capabilities

#![forbid(unsafe_code)]
#![expect(
    clippy::missing_errors_doc,
    reason = "playground — evolving API surface"
)]

pub mod ipc_client;
pub mod mcp_tools;
pub mod primal_client;
pub mod squirrel_client;
