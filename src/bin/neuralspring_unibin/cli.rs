// SPDX-License-Identifier: AGPL-3.0-or-later

//! CLI definition for the neuralSpring UniBin.

use clap::{Parser, Subcommand};

/// neuralSpring UniBin — eukaryotic single-binary deployment.
///
/// Replaces the prokaryotic multi-binary topology with a single
/// binary exposing certification, validation, serve, status, and
/// version subcommands.
#[derive(Parser)]
#[command(name = "neuralspring-unibin", version, about)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Run certification layers (L0-L3).
    Certify {
        /// Maximum certification layer (0=bare, 1=discovery, 2=parity, 3=nucleus).
        #[arg(long, default_value_t = 3)]
        layer: u8,

        /// Run only bare properties (alias for --layer 0).
        #[arg(long, conflicts_with = "layer")]
        bare: bool,
    },

    /// Run validation scenarios.
    Validate {
        /// Filter by track (e.g. "nucleus-composition", "spectral-analysis").
        #[arg(long)]
        track: Option<String>,

        /// Run a specific scenario by ID.
        #[arg(long)]
        scenario: Option<String>,

        /// Filter by tier: "rust", "live", "both".
        #[arg(long)]
        tier: Option<String>,

        /// List available scenarios without running them.
        #[arg(long)]
        list: bool,
    },

    /// Start the JSON-RPC IPC server (primal serve mode).
    Serve,

    /// Show capability discovery summary.
    Status,

    /// Print version and exit.
    Version,
}
