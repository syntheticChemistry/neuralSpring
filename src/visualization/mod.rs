// SPDX-License-Identifier: AGPL-3.0-or-later

//! petalTongue visualization integration for neuralSpring.
//!
//! Follows healthSpring's push-based `DataBinding` pattern: scenario
//! builders call real neuralSpring functions and package results as
//! typed [`DataChannel`] payloads that petalTongue renders via
//! `visualization.render` / `visualization.render.stream`.
//!
//! Domain: `"neural"` (triggers electric blue/magenta palette).
//!
//! ## Modules
//!
//! - [`types`] — petalTongue-compatible schema (all 8 `DataBinding` variants)
//! - [`ipc_push`] — `PetalTonguePushClient` for runtime socket discovery
//! - [`scenarios`] — per-domain scenario builders + `full_study()` combiner

pub mod ipc_push;
pub mod scenarios;
pub mod stream;
pub mod types;

pub use ipc_push::{PetalTonguePushClient, PushError, PushResult};
pub use scenarios::{
    coordination_study, folding_study, full_study, provenance_study, scenario_with_edges_json,
    spectral_study, training_study,
};
pub use stream::{SessionStats, StreamSession};
pub use types::{DataChannel, NeuralScenario, ScenarioEdge, ScenarioNode, ThresholdRange};
