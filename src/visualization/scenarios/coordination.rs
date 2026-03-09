// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-agent coordination scenario builder.
//!
//! Generates a lattice of agents and sweeps disorder to measure how
//! spectral properties (IPR, LSR, algebraic connectivity) evolve.

use crate::agent_coordination::{coordination_spectral_analysis, generate_lattice_agents};
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge};

use super::{node, scaffold, scatter3d, timeseries};

/// Build the agent coordination study scenario.
///
/// Produces 1 node with a `Scatter3D` (disorder vs IPR vs algebraic
/// connectivity) and a `TimeSeries` (disorder vs level spacing ratio).
#[must_use]
pub fn coordination_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Multi-Agent Coordination Spectral Analysis",
        "Quorum sensing spectral diagnostics: IPR, LSR, algebraic connectivity vs disorder (nS-05)",
    );

    let n_agents = 16;
    let dim = 2;
    let comm_range = 3.0;
    let disorder_vals = vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0];
    let cap_var = 1.0;

    let mut rng = Rng::new(42);
    let agents = generate_lattice_agents(n_agents, dim, cap_var, &mut rng);

    let mut disorders = Vec::with_capacity(disorder_vals.len());
    let mut iprs = Vec::with_capacity(disorder_vals.len());
    let mut lsrs = Vec::with_capacity(disorder_vals.len());
    let mut connectivities = Vec::with_capacity(disorder_vals.len());
    let mut labels = Vec::with_capacity(disorder_vals.len());

    for &w in &disorder_vals {
        let cr = coordination_spectral_analysis(&agents, comm_range, w);
        disorders.push(w);
        iprs.push(cr.mean_ipr);
        lsrs.push(cr.level_spacing_ratio);
        connectivities.push(cr.algebraic_connectivity);
        labels.push(format!("W={w}"));
    }

    s.ecosystem.primals.push(node(
        "agent_coordination",
        &format!("Agent Coordination ({n_agents} agents, dim={dim})"),
        "compute",
        0.0,
        0.0,
        &["science.agent_coordination"],
        vec![
            scatter3d(
                "coordination-phase-space",
                "Coordination Phase Space",
                "dimensionless",
                disorders.clone(),
                iprs,
                connectivities,
                labels,
            ),
            timeseries(
                "disorder-vs-lsr",
                "Disorder vs Level Spacing Ratio",
                "Disorder (W)",
                "LSR",
                "dimensionless",
                disorders,
                lsrs,
            ),
        ],
        vec![],
    ));

    (s, Vec::new())
}
