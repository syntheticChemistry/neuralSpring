// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-agent AI coordination as quorum sensing.
//!
//! baseCamp Sub-thesis 05: Multi-Agent AI Coordination as QS.
//!
//! Applies the Anderson localization framework for quorum sensing
//! to decentralized multi-agent AI systems. Predicts coordination
//! phase transitions based on interaction topology and agent
//! heterogeneity.
//!
//! ## Grounding papers
//!
//! - `SwarmSys` (2025) "Decentralized Swarm-Inspired Agents"
//! - Emergent Collective Memory (2025) — stigmergic coordination
//! - Foreback & Dolson (2025) — Paper 015 (heterogeneous swarm)
//!
//! ## Validated primitives
//!
//! - [`crate::anderson_localization`] — IPR, Hamiltonian construction
//! - [`crate::game_theory`] — replicator dynamics, spatial cooperation
//! - [`crate::swarm_robotics`] — swarm fitness evaluation
//! - [`crate::eigh::eigh_householder_qr`] — eigendecomposition

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use crate::anderson_localization::mean_ipr;
use crate::eigh::eigh_householder_qr;
use crate::rng::Rng;

/// Agent in the coordination model.
#[derive(Debug, Clone)]
pub struct Agent {
    /// Position (x, y) or (x, y, z) depending on dimension.
    pub position: Vec<f64>,
    /// Agent capability parameter (heterogeneity source).
    pub capability: f64,
    /// Coordination signal level emitted by this agent.
    pub signal_level: f64,
    /// Whether this agent is cooperating (true) or defecting (false).
    pub cooperating: bool,
}

/// Build the agent interaction graph as a weighted adjacency matrix.
///
/// Agents within `comm_range` of each other have edge weight
/// proportional to 1/distance. Returns flat row-major n×n matrix.
#[must_use]
pub fn interaction_graph(agents: &[Agent], comm_range: f64) -> Vec<f64> {
    let n = agents.len();
    let mut adj = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&agents[i].position, &agents[j].position);
            if dist < comm_range && dist > 1e-300 {
                let weight = 1.0 / dist;
                adj[i * n + j] = weight;
                adj[j * n + i] = weight;
            }
        }
    }
    adj
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the graph Laplacian from an adjacency matrix.
///
/// L = D - A where D is the degree matrix (diagonal with row sums).
/// Returns flat row-major n×n matrix.
#[must_use]
pub fn graph_laplacian(adjacency: &[f64], n: usize) -> Vec<f64> {
    let mut laplacian = vec![0.0; n * n];
    for i in 0..n {
        let degree: f64 = (0..n).map(|j| adjacency[i * n + j]).sum();
        laplacian[i * n + i] = degree;
        for j in 0..n {
            laplacian[i * n + j] -= adjacency[i * n + j];
        }
    }
    laplacian
}

/// Add disorder from agent heterogeneity to the Laplacian.
///
/// Returns H = L + W * diag(heterogeneity), where heterogeneity
/// is derived from agent capability variance.
#[must_use]
pub fn disordered_laplacian(
    laplacian: &[f64],
    n: usize,
    agents: &[Agent],
    disorder_strength: f64,
) -> Vec<f64> {
    let mut h = laplacian.to_vec();
    let mean_cap = agents.iter().map(|a| a.capability).sum::<f64>() / n.max(1) as f64;
    for i in 0..n {
        let deviation = agents[i].capability - mean_cap;
        h[i * n + i] += disorder_strength * deviation;
    }
    h
}

/// Coordination analysis via Anderson localization on the interaction graph.
///
/// Computes the IPR and level spacing ratio of the disordered Laplacian
/// to predict whether coordination will succeed (delocalized = coordinated)
/// or fail (localized = fragmented).
#[must_use]
pub fn coordination_spectral_analysis(
    agents: &[Agent],
    comm_range: f64,
    disorder_strength: f64,
) -> CoordinationResult {
    let n = agents.len();
    let adj = interaction_graph(agents, comm_range);
    let lap = graph_laplacian(&adj, n);
    let h = disordered_laplacian(&lap, n, agents, disorder_strength);

    let decomp = eigh_householder_qr(&h, n);
    let mut eigenvalues = decomp.eigenvalues.clone();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_ipr_val = mean_ipr(&decomp.eigenvectors, n);
    let lsr = crate::weight_spectral::level_spacing_ratio(&eigenvalues);

    let algebraic_connectivity = if eigenvalues.len() >= 2 {
        eigenvalues[1]
    } else {
        0.0
    };

    CoordinationResult {
        mean_ipr: mean_ipr_val,
        level_spacing_ratio: lsr,
        algebraic_connectivity,
        eigenvalues,
    }
}

/// Result of coordination spectral analysis.
#[derive(Debug, Clone)]
pub struct CoordinationResult {
    /// Mean IPR (high = localized/fragmented, low = delocalized/coordinated).
    pub mean_ipr: f64,
    /// Level spacing ratio (GOE ≈ 0.531 = coordinated, Poisson ≈ 0.386 = fragmented).
    pub level_spacing_ratio: f64,
    /// Second-smallest eigenvalue of the Laplacian (Fiedler value).
    /// > 0 means the graph is connected; larger = better coordination.
    pub algebraic_connectivity: f64,
    /// Full sorted eigenvalue spectrum.
    pub eigenvalues: Vec<f64>,
}

/// QS signaling step: each agent emits signal proportional to capability.
///
/// Agents detect signal from neighbors within `detection_radius`.
/// If local signal exceeds `threshold`, the agent cooperates.
pub fn qs_signaling_step(
    agents: &mut [Agent],
    detection_radius: f64,
    signal_decay: f64,
    threshold: f64,
) {
    let n = agents.len();
    let signals: Vec<f64> = agents.iter().map(|a| a.signal_level).collect();
    let positions: Vec<Vec<f64>> = agents.iter().map(|a| a.position.clone()).collect();

    for i in 0..n {
        let mut local_signal = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = euclidean_distance(&positions[i], &positions[j]);
            if dist < detection_radius && dist > 1e-300 {
                local_signal += signals[j] * (-signal_decay * dist).exp();
            }
        }
        agents[i].cooperating = local_signal > threshold;
        agents[i].signal_level = if agents[i].cooperating {
            agents[i].capability
        } else {
            agents[i].capability * 0.1
        };
    }
}

/// Measure coordination: fraction of agents cooperating.
#[must_use]
pub fn coordination_fraction(agents: &[Agent]) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    let cooperating = agents.iter().filter(|a| a.cooperating).count();
    cooperating as f64 / agents.len() as f64
}

/// Generate agents on a lattice with given dimensionality.
///
/// `dim` = 1 (chain), 2 (grid), or 3 (cube).
/// Agents placed at integer lattice sites with random capabilities.
#[must_use]
pub fn generate_lattice_agents(
    n_per_side: usize,
    dim: usize,
    capability_variance: f64,
    rng: &mut Rng,
) -> Vec<Agent> {
    let mut agents = Vec::new();
    let sides: Vec<usize> = (0..dim.max(1)).map(|_| n_per_side).collect();

    let total: usize = sides.iter().product();
    for idx in 0..total {
        let mut position = Vec::with_capacity(dim);
        let mut remainder = idx;
        for &side in &sides {
            position.push((remainder % side) as f64);
            remainder /= side;
        }
        let capability = capability_variance.mul_add(rng.normal(), 1.0);
        agents.push(Agent {
            position,
            capability: capability.max(0.01),
            signal_level: capability.max(0.01),
            cooperating: false,
        });
    }
    agents
}

/// Run dimensional coordination experiment.
///
/// Generates agents on 1D, 2D, and 3D lattices and measures
/// coordination efficiency for each. Returns coordination fractions
/// for each dimension after `n_steps` QS signaling steps.
#[must_use]
pub fn dimensional_coordination_sweep(
    n_per_side: usize,
    capability_variance: f64,
    comm_range: f64,
    detection_radius: f64,
    signal_decay: f64,
    threshold: f64,
    n_steps: usize,
    seed: u64,
) -> DimensionalResult {
    let mut results = Vec::with_capacity(3);

    for dim in 1..=3 {
        let mut rng = Rng::new(seed);
        let mut agents = generate_lattice_agents(n_per_side, dim, capability_variance, &mut rng);
        for _ in 0..n_steps {
            qs_signaling_step(&mut agents, detection_radius, signal_decay, threshold);
        }
        let frac = coordination_fraction(&agents);
        let spectral = coordination_spectral_analysis(&agents, comm_range, capability_variance);
        results.push((dim, frac, spectral));
    }

    DimensionalResult {
        dim1_coordination: results[0].1,
        dim2_coordination: results[1].1,
        dim3_coordination: results[2].1,
        dim1_ipr: results[0].2.mean_ipr,
        dim2_ipr: results[1].2.mean_ipr,
        dim3_ipr: results[2].2.mean_ipr,
    }
}

/// Result of dimensional coordination sweep.
#[derive(Debug, Clone)]
pub struct DimensionalResult {
    /// Coordination fraction in 1D.
    pub dim1_coordination: f64,
    /// Coordination fraction in 2D.
    pub dim2_coordination: f64,
    /// Coordination fraction in 3D.
    pub dim3_coordination: f64,
    /// Mean IPR in 1D (high = localized/fragmented).
    pub dim1_ipr: f64,
    /// Mean IPR in 2D.
    pub dim2_ipr: f64,
    /// Mean IPR in 3D (low = delocalized/coordinated).
    pub dim3_ipr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agents(n: usize) -> Vec<Agent> {
        let mut rng = Rng::new(42);
        generate_lattice_agents(n, 2, 0.1, &mut rng)
    }

    #[test]
    fn interaction_graph_symmetric() {
        let agents = test_agents(4);
        let adj = interaction_graph(&agents, 2.0);
        let n = agents.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (adj[i * n + j] - adj[j * n + i]).abs() < 1e-14,
                    "adjacency not symmetric"
                );
            }
        }
    }

    #[test]
    fn laplacian_row_sums_zero() {
        let agents = test_agents(4);
        let adj = interaction_graph(&agents, 2.0);
        let n = agents.len();
        let lap = graph_laplacian(&adj, n);
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| lap[i * n + j]).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Laplacian row {i} sums to {row_sum}, not 0"
            );
        }
    }

    #[test]
    fn coordination_fraction_bounds() {
        let agents = test_agents(4);
        let frac = coordination_fraction(&agents);
        assert!((0.0..=1.0).contains(&frac));
    }

    #[test]
    fn lattice_agents_correct_count() {
        let mut rng = Rng::new(42);
        let agents_1d = generate_lattice_agents(5, 1, 0.1, &mut rng);
        assert_eq!(agents_1d.len(), 5);

        let agents_2d = generate_lattice_agents(4, 2, 0.1, &mut rng);
        assert_eq!(agents_2d.len(), 16);

        let agents_3d = generate_lattice_agents(3, 3, 0.1, &mut rng);
        assert_eq!(agents_3d.len(), 27);
    }

    #[test]
    fn qs_step_changes_state() {
        let mut agents = test_agents(4);
        for a in &mut agents {
            a.signal_level = 1.0;
        }
        qs_signaling_step(&mut agents, 3.0, 0.1, 0.5);
        let any_cooperating = agents.iter().any(|a| a.cooperating);
        assert!(any_cooperating, "at least some agents should cooperate");
    }

    #[test]
    fn spectral_analysis_finite() {
        let agents = test_agents(4);
        let result = coordination_spectral_analysis(&agents, 2.0, 0.1);
        assert!(result.mean_ipr.is_finite());
        assert!(result.level_spacing_ratio.is_finite());
        assert!(result.algebraic_connectivity.is_finite());
    }

    #[test]
    fn determinism() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let a1 = generate_lattice_agents(4, 2, 0.1, &mut rng1);
        let a2 = generate_lattice_agents(4, 2, 0.1, &mut rng2);
        for (ag1, ag2) in a1.iter().zip(a2.iter()) {
            assert_eq!(ag1.capability, ag2.capability);
            assert_eq!(ag1.position, ag2.position);
        }
    }
}
