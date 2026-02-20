// SPDX-License-Identifier: AGPL-3.0-only

//! Heterogeneous controller representations for evolutionary swarm robotics.
//!
//! Port of `control/swarm_robotics/swarm_robotics.py`.
//! Foreback, Bohm, Dolson (2025) IEEE — heterogeneous controllers maintain diversity.

#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::imprecise_flops,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate
)]

use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerType {
    NeuralNet,
    BehaviorTree,
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

#[derive(Debug, Clone)]
pub struct Controller {
    pub ctrl_type: ControllerType,
    pub params: Vec<f64>,
}

impl Controller {
    #[must_use]
    pub fn new(ctrl_type: ControllerType, params: Vec<f64>) -> Self {
        Self { ctrl_type, params }
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
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
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

#[allow(clippy::cast_possible_truncation)]
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

const GRID_SIZE: i32 = 12;
const N_AGENTS: usize = 6;
const N_FOOD: usize = 4;
const N_STEPS: usize = 30;

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
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.ln();
        }
    }
    h
}

pub fn create_controller(ctrl_type: ControllerType, rng: &mut Rng) -> Controller {
    let params = (0..ctrl_type.param_len()).map(|_| rng.uniform()).collect();
    Controller::new(ctrl_type, params)
}

pub fn mutate(c: &Controller, rng: &mut Rng, mutation_rate: f64) -> Controller {
    let params = c
        .params
        .iter()
        .map(|&p| (p + rng.normal_params(0.0, mutation_rate)).clamp(0.0, 1.0))
        .collect();
    Controller::new(c.ctrl_type, params)
}

const POP_SIZE: usize = 48;
const N_GEN: usize = 40;
const TOURNAMENT_SIZE: usize = 5;
const MUTATION_RATE: f64 = 0.08;

#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub mean_fitness: Vec<f64>,
    pub diversity: Vec<f64>,
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn run_evolution_homogeneous(seed: u64) -> EvolutionResult {
    run_evolution_inner(seed, true)
}

#[allow(clippy::cast_precision_loss)]
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
                    .max_by(|a, b| {
                        fitnesses[**a]
                            .partial_cmp(&fitnesses[**b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
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

    #[test]
    fn neural_forward_valid_action() {
        let params: Vec<f64> = (0..33).map(|i| f64::from(i) / 33.0).collect();
        assert!(neural_forward(&params, 0.5) < 5);
    }

    #[test]
    fn shannon_homogeneous_zero() {
        let types = vec![ControllerType::NeuralNet; 10];
        assert!((shannon_diversity(&types) - 0.0).abs() < 1e-10);
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
}
