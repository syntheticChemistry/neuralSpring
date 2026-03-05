// SPDX-License-Identifier: AGPL-3.0-or-later

//! Heterogeneous controller representations for evolutionary swarm robotics.
//!
//! Port of `control/swarm_robotics/swarm_robotics.py`.
//! Foreback, Bohm, Dolson (2025) IEEE — heterogeneous controllers maintain diversity.
//!
//! ## `BarraCUDA` connection
//!
//! - Neural controller forward pass: `barracuda::ops::matmul` + sigmoid activation
//! - Behavior tree evaluation: not GPU-portable (branching control flow)
//! - Swarm fitness aggregation: `barracuda::ops::SumReduceF64` (team score)
//! - Heterogeneous population eval: `barracuda::ops::batch_gemm` (per-controller type)

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "domain-specific numeric patterns"
)]

/// WGSL shader: batch neural network forward pass for swarm controllers.
///
/// One thread per (controller, evaluation) pair. Architecture: 1 input →
/// 4 hidden (sigmoid) → 5 output (sigmoid) → argmax action.
/// Paper 015 (Swarm Robotics).
///
/// Absorption target: `barracuda::ops::batch_gemm`.
/// Validated: `validate_gpu_swarm` (9/9 PASS).
pub use neural_spring_forge::shaders::SWARM_NN_FORWARD as WGSL_SWARM_NN_FORWARD;

use crate::primitives;
use crate::rng::Rng;

/// Controller representation type for heterogeneous swarm evolution.
///
/// From Foreback, Bohm, Dolson (2025) IEEE — three representations
/// compete and co-evolve in a single population.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerType {
    /// 4-hidden-unit MLP with sigmoid activations (33 parameters).
    NeuralNet,
    /// Threshold-based decision tree (10 parameters).
    BehaviorTree,
    /// Ordered threshold rules (4 parameters).
    RuleBased,
}

impl ControllerType {
    #[must_use]
    pub const fn as_usize(self) -> usize {
        match self {
            Self::NeuralNet => 0,
            Self::BehaviorTree => 1,
            Self::RuleBased => 2,
        }
    }

    #[must_use]
    pub const fn param_len(self) -> usize {
        match self {
            Self::NeuralNet => 33,
            Self::BehaviorTree => 10,
            Self::RuleBased => 4,
        }
    }
}

/// An individual controller with its representation type and parameter vector.
///
/// The parameter vector length depends on [`ControllerType::param_len`].
#[derive(Debug, Clone)]
pub struct Controller {
    /// Which representation this controller uses.
    pub ctrl_type: ControllerType,
    /// Flat parameter vector (weights, thresholds, or rules).
    pub params: Vec<f64>,
}

impl Controller {
    #[must_use]
    pub const fn new(ctrl_type: ControllerType, params: Vec<f64>) -> Self {
        Self { ctrl_type, params }
    }
}

fn sigmoid(x: f64) -> f64 {
    primitives::sigmoid(x)
}

#[must_use]
pub fn neural_forward(params: &[f64], sense: f64) -> usize {
    let mut h = [0.0_f64; 4];
    for (i, (&w, &b)) in params[0..4].iter().zip(params[4..8].iter()).enumerate() {
        h[i] = sigmoid(sense.mul_add(w, b));
    }
    let mut out = [0.0_f64; 5];
    for j in 0..5 {
        let mut sum = params[28 + j];
        for (i, &hv) in h.iter().enumerate() {
            sum += hv * params[8 + i * 5 + j];
        }
        out[j] = sigmoid(sum);
    }
    out.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

/// Max output-layer activation (before argmax). For `mean_reduce` pipeline validation.
#[must_use]
pub fn neural_forward_max_score(params: &[f64], sense: f64) -> f64 {
    let mut h = [0.0_f64; 4];
    for (i, (&w, &b)) in params[0..4].iter().zip(params[4..8].iter()).enumerate() {
        h[i] = sigmoid(sense.mul_add(w, b));
    }
    let mut best = f64::NEG_INFINITY;
    for j in 0..5 {
        let mut sum = params[28 + j];
        for (i, &hv) in h.iter().enumerate() {
            sum += hv * params[8 + i * 5 + j];
        }
        let o_j = sigmoid(sum);
        if o_j > best {
            best = o_j;
        }
    }
    best
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "clamped f64 → usize for discrete action index"
)]
#[must_use]
pub fn behavior_forward(params: &[f64], sense: f64) -> usize {
    for i in (0..10).step_by(2) {
        if sense < params[i] {
            return (params[i + 1] * 5.0).clamp(0.0, 4.99) as usize;
        }
    }
    (params[9] * 5.0).clamp(0.0, 4.99) as usize
}

#[must_use]
pub fn rule_forward(params: &[f64], sense: f64) -> usize {
    let mut t = [params[0], params[1], params[2], params[3]];
    t.sort_by(f64::total_cmp);
    t.iter().filter(|&&x| sense > x).count().min(4)
}

#[must_use]
pub fn controller_forward(ctrl: &Controller, sense: f64) -> usize {
    match ctrl.ctrl_type {
        ControllerType::NeuralNet => neural_forward(&ctrl.params, sense),
        ControllerType::BehaviorTree => behavior_forward(&ctrl.params, sense),
        ControllerType::RuleBased => rule_forward(&ctrl.params, sense),
    }
}

/// Swarm simulation grid and population parameters.
/// From Foreback, Bohm, Dolson (2025) IEEE — chosen to balance
/// computational cost against statistical convergence:
/// - 12×12 grid: small enough for rapid evaluation, large enough
///   for meaningful spatial dynamics.
/// - 6 agents, 4 food: matches the paper's "small swarm" experiments.
/// - 30 steps: sufficient for agents to traverse the grid and
///   demonstrate foraging behaviour.
const GRID_SIZE: i32 = 12;
const N_AGENTS: usize = 6;
const N_FOOD: usize = 4;
const N_STEPS: usize = 30;

/// Grid-world foraging simulation for evaluating controller fitness.
///
/// Agents navigate the grid collecting food; total collected is the fitness.
#[derive(Debug)]
pub struct SwarmSimulation {
    grid: Vec<Vec<i32>>,
    food_pos: Vec<(i32, i32)>,
    agent_pos: Vec<(i32, i32)>,
}

impl SwarmSimulation {
    #[must_use]
    pub fn new(rng: &mut Rng) -> Self {
        let g = GRID_SIZE as usize;
        let mut grid = vec![vec![0; g]; g];
        let food_pos: Vec<_> = (0..N_FOOD)
            .map(|_| {
                let (x, y) = (rng.usize(g) as i32, rng.usize(g) as i32);
                grid[x as usize][y as usize] = 1;
                (x, y)
            })
            .collect();
        let agent_pos: Vec<_> = (0..N_AGENTS)
            .map(|_| (rng.usize(g) as i32, rng.usize(g) as i32))
            .collect();
        Self {
            grid,
            food_pos,
            agent_pos,
        }
    }

    #[must_use]
    pub fn run(&mut self, ctrl: &Controller) -> f64 {
        static MOVES: [(i32, i32); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
        let mut collected = 0.0_f64;
        for _ in 0..N_STEPS {
            for a in 0..N_AGENTS {
                let (x, y) = self.agent_pos[a];
                let min_d = self
                    .food_pos
                    .iter()
                    .map(|(fx, fy)| {
                        let dx = f64::from(fx - x);
                        let dy = f64::from(fy - y);
                        dx.hypot(dy)
                    })
                    .fold(f64::INFINITY, f64::min);
                let sense = 1.0 / (1.0 + min_d / f64::from(GRID_SIZE));
                let act = controller_forward(ctrl, sense);
                let (dx, dy) = MOVES[act];
                let nx = (x + dx).clamp(0, GRID_SIZE - 1);
                let ny = (y + dy).clamp(0, GRID_SIZE - 1);
                self.agent_pos[a] = (nx, ny);
                if self.grid[nx as usize][ny as usize] == 1 {
                    self.grid[nx as usize][ny as usize] = 0;
                    collected += 1.0;
                }
            }
        }
        collected
    }
}

#[must_use]
pub fn shannon_diversity(types: &[ControllerType]) -> f64 {
    if types.is_empty() {
        return 0.0;
    }
    let n = types.len() as f64;
    let mut counts = [0usize; 3];
    for &t in types {
        counts[t.as_usize()] += 1;
    }
    let freqs: Vec<f64> = counts.iter().map(|&c| c as f64 / n).collect();
    primitives::shannon_entropy(&freqs)
}

#[must_use]
pub fn create_controller(ctrl_type: ControllerType, rng: &mut Rng) -> Controller {
    let params = (0..ctrl_type.param_len()).map(|_| rng.uniform()).collect();
    Controller::new(ctrl_type, params)
}

#[must_use]
pub fn mutate(c: &Controller, rng: &mut Rng, mutation_rate: f64) -> Controller {
    let params = c
        .params
        .iter()
        .map(|&p| (p + rng.normal_params(0.0, mutation_rate)).clamp(0.0, 1.0))
        .collect();
    Controller::new(c.ctrl_type, params)
}

/// Evolutionary algorithm parameters from Foreback et al. (2025).
/// `POP_SIZE=48`: divisible by 3 controller types (16 each for heterogeneous).
/// `N_GEN=40`: sufficient for fitness plateau in small-swarm domain.
/// `TOURNAMENT_SIZE=5`: standard tournament pressure for pop≈50.
/// `MUTATION_RATE=0.08`: Gaussian σ — calibrated for [0,1] parameter space.
const POP_SIZE: usize = 48;
const N_GEN: usize = 40;
const TOURNAMENT_SIZE: usize = 5;
const MUTATION_RATE: f64 = 0.08;

/// Per-generation metrics from an evolutionary swarm experiment.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Mean fitness across the population at each generation.
    pub mean_fitness: Vec<f64>,
    /// Shannon diversity of controller types at each generation.
    pub diversity: Vec<f64>,
}

#[must_use]
pub fn run_evolution_homogeneous(seed: u64) -> EvolutionResult {
    run_evolution_inner(seed, true)
}

#[must_use]
pub fn run_evolution_heterogeneous(seed: u64) -> EvolutionResult {
    run_evolution_inner(seed, false)
}

fn run_evolution_inner(seed: u64, homogeneous: bool) -> EvolutionResult {
    let mut rng = Rng::new(seed);
    let t = |i: usize| {
        if homogeneous {
            ControllerType::NeuralNet
        } else {
            match i % 3 {
                0 => ControllerType::NeuralNet,
                1 => ControllerType::BehaviorTree,
                _ => ControllerType::RuleBased,
            }
        }
    };
    let mut population: Vec<Controller> = (0..POP_SIZE)
        .map(|i| create_controller(t(i), &mut rng))
        .collect();

    let mut mean_fitness = Vec::with_capacity(N_GEN);
    let mut diversity = Vec::with_capacity(N_GEN);
    let k = TOURNAMENT_SIZE.min(population.len());

    for _ in 0..N_GEN {
        let fitnesses: Vec<f64> = population
            .iter()
            .map(|c| SwarmSimulation::new(&mut rng).run(c))
            .collect();
        mean_fitness.push(fitnesses.iter().sum::<f64>() / POP_SIZE as f64);
        diversity.push(shannon_diversity(
            &population.iter().map(|c| c.ctrl_type).collect::<Vec<_>>(),
        ));

        population = (0..POP_SIZE)
            .map(|_| {
                let idx = rng.choose_distinct(population.len(), k);
                let w = idx
                    .iter()
                    .max_by(|a, b| f64::total_cmp(&fitnesses[**a], &fitnesses[**b]))
                    .copied()
                    .unwrap_or(0);
                mutate(&population[w], &mut rng, MUTATION_RATE)
            })
            .collect();

        if !homogeneous {
            for (i, &t) in [
                ControllerType::NeuralNet,
                ControllerType::BehaviorTree,
                ControllerType::RuleBased,
            ]
            .iter()
            .enumerate()
            {
                population[(i * (POP_SIZE / 3)) % POP_SIZE] = create_controller(t, &mut rng);
            }
        }
    }

    EvolutionResult {
        mean_fitness,
        diversity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn neural_forward_valid_action() {
        let params: Vec<f64> = (0..33).map(|i| f64::from(i) / 33.0).collect();
        assert!(neural_forward(&params, 0.5) < 5);
    }

    #[test]
    fn shannon_homogeneous_zero() {
        let types = vec![ControllerType::NeuralNet; 10];
        assert!((shannon_diversity(&types) - 0.0).abs() < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn mutate_preserves_type() {
        let mut rng = Rng::new(42);
        let m = mutate(
            &Controller::new(ControllerType::BehaviorTree, vec![0.5; 10]),
            &mut rng,
            0.1,
        );
        assert_eq!(m.ctrl_type, ControllerType::BehaviorTree);
    }

    #[test]
    fn create_controller_has_correct_params() {
        let mut rng = Rng::new(42);
        for ct in [
            ControllerType::NeuralNet,
            ControllerType::BehaviorTree,
            ControllerType::RuleBased,
        ] {
            let c = create_controller(ct, &mut rng);
            assert_eq!(c.ctrl_type, ct);
            assert_eq!(c.params.len(), ct.param_len());
        }
    }

    #[test]
    fn controller_forward_all_types() {
        let mut rng = Rng::new(42);
        for ct in [
            ControllerType::NeuralNet,
            ControllerType::BehaviorTree,
            ControllerType::RuleBased,
        ] {
            let c = create_controller(ct, &mut rng);
            let action = controller_forward(&c, 0.5);
            assert!(action < 5, "action must be valid (< 5)");
        }
    }

    #[test]
    fn behavior_forward_returns_valid() {
        let params: Vec<f64> = (0..10).map(|i| f64::from(i) / 10.0).collect();
        assert!(behavior_forward(&params, 0.5) < 5);
    }

    #[test]
    fn rule_forward_returns_valid() {
        assert!(rule_forward(&[0.3, 0.6, 0.1, 0.8], 0.5) < 5);
    }

    #[test]
    fn controller_type_as_usize_roundtrip() {
        assert_eq!(ControllerType::NeuralNet.as_usize(), 0);
        assert_eq!(ControllerType::BehaviorTree.as_usize(), 1);
        assert_eq!(ControllerType::RuleBased.as_usize(), 2);
    }

    #[test]
    fn swarm_simulation_runs() {
        let mut rng = Rng::new(42);
        let ctrl = create_controller(ControllerType::NeuralNet, &mut rng);
        let mut sim = SwarmSimulation::new(&mut rng);
        let fitness = sim.run(&ctrl);
        assert!((0.0..=1.0).contains(&fitness), "fitness should be in [0,1]");
    }

    #[test]
    fn run_evolution_homogeneous_produces_results() {
        let result = run_evolution_homogeneous(42);
        assert!(!result.mean_fitness.is_empty());
        assert!(!result.diversity.is_empty());
        assert!(result.mean_fitness.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn run_evolution_heterogeneous_produces_results() {
        let result = run_evolution_heterogeneous(42);
        assert!(!result.mean_fitness.is_empty());
        assert!(!result.diversity.is_empty());
        assert!(result.diversity.iter().any(|&d| d > 0.0));
    }
}
