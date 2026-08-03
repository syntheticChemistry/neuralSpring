// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS pipeline executor for neuralSpring.
//!
//! Bridges metalForge [`PipelineGraph`](neural_spring_forge::graph::PipelineGraph) to actual neuralSpring computation.
//! Follows the NUCLEUS atomic model:
//!
//! - **Tower**: Capability discovery — resolves stage capabilities to local functions
//! - **Node**: Compute dispatch — executes each stage (CPU or GPU via `Dispatcher`)
//! - **Nest**: Provenance — records substrate, timing, outputs per stage
//!
//! ## Usage
//!
//! ```no_run
//! use neural_spring::nucleus_pipeline::{execute_composition_pipeline, PipelineReport};
//! let report = execute_composition_pipeline().expect("pipeline execution");
//! assert!(report.all_passed());
//! ```

mod dispatch;
mod error;
mod executor;
mod report;

pub use dispatch::{
    PIPELINE_CAPABILITIES, dispatch_capability, dispatch_capability_gpu, is_pipeline_capability,
};
pub use error::PipelineError;
#[cfg(feature = "primalspring")]
pub use executor::execute_graph_live;
pub use executor::{
    execute_composition_pipeline, execute_composition_pipeline_gpu, execute_graph,
    execute_graph_gpu,
};
pub use report::PipelineReport;
