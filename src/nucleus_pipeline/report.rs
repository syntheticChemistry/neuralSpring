// SPDX-License-Identifier: AGPL-3.0-or-later

//! [`PipelineReport`] — provenance summary for a completed pipeline run.

use neural_spring_forge::graph::PipelineExecution;

/// A completed pipeline report with provenance metadata.
#[derive(Debug)]
pub struct PipelineReport {
    /// Per-stage results, timings, and outputs from the run.
    pub execution: PipelineExecution,
    /// Name of the `PipelineGraph` that was executed.
    pub pipeline_name: String,
    /// Summary label for substrate mix (CPU, GPU, or mixed).
    pub substrate_used: String,
    /// Number of stages in topological execution order.
    pub total_stages: usize,
    /// How many stages executed on GPU vs CPU.
    pub gpu_stages: usize,
    /// Stages that ran on the CPU path (including GPU fallback).
    pub cpu_stages: usize,
}

impl PipelineReport {
    /// Whether all stages passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.execution.all_passed()
    }

    /// Total elapsed microseconds.
    #[must_use]
    pub fn total_us(&self) -> f64 {
        self.execution.total_elapsed_us()
    }
}

#[cfg(test)]
mod tests {
    use crate::nucleus_pipeline::execute_composition_pipeline;

    #[test]
    #[expect(clippy::expect_used, reason = "test assertion")]
    fn pipeline_report_total_us_delegates_to_execution() {
        let report = execute_composition_pipeline().expect("composition pipeline");
        assert!((report.total_us() - report.execution.total_elapsed_us()).abs() < f64::EPSILON);
    }
}
