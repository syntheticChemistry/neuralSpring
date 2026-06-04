// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed errors for [`PipelineGraph`](neural_spring_forge::graph::PipelineGraph) execution.

use std::fmt;

/// Errors from pipeline graph execution.
///
/// Returned when a `PipelineGraph` cannot be topologically sorted
/// (cycle detected) or references a stage ID not present in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// The graph contains a cycle and cannot be topologically sorted.
    CyclicGraph {
        /// Name of the pipeline that failed validation.
        pipeline: String,
    },
    /// A stage ID from the topological order was not found in the graph.
    MissingStage {
        /// The stage ID that could not be resolved.
        stage_id: String,
        /// Name of the pipeline containing the missing reference.
        pipeline: String,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CyclicGraph { pipeline } => {
                write!(f, "pipeline '{pipeline}' contains a cycle")
            }
            Self::MissingStage { stage_id, pipeline } => {
                write!(f, "stage '{stage_id}' not found in pipeline '{pipeline}'")
            }
        }
    }
}

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_error_display() {
        let err = PipelineError::CyclicGraph {
            pipeline: "test".to_string(),
        };
        assert!(err.to_string().contains("cycle"));

        let err = PipelineError::MissingStage {
            stage_id: "ghost".to_string(),
            pipeline: "test".to_string(),
        };
        assert!(err.to_string().contains("ghost"));
    }
}
