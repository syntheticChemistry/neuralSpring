// SPDX-License-Identifier: AGPL-3.0-or-later

//! ML workload dispatch routing — GPU vs CPU crossover logic.
//!
//! Codifies the empirical crossover points documented in
//! `metalForge/CROSS_SYSTEM_DISPATCH.md`. `ToadStool` can absorb this into
//! `barracuda::dispatch` to extend the routing with ML-specific heuristics.
//!
//! ## Key finding
//!
//! GPU dispatch overhead on Vulkan is ~1.5 ms fixed cost (`queue.submit()` +
//! readback). GPU compute is negligible at all tested scales (RTX 4070, 5888
//! CUDA cores). The crossover is therefore:
//!
//! - **CPU wins** when total CPU work < 1.5 ms
//! - **GPU wins** when total CPU work > 1.5 ms OR fused dispatch amortizes cost
//!
//! `StatefulPipeline` and `TensorSession` eliminate the fixed cost for
//! multi-pass workloads by batching into a single `queue.submit()`.

/// Substrate recommendation for a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Substrate {
    /// Use GPU (Vulkan/Metal/DX12) — large workloads or fused pipelines.
    Gpu,
    /// Use CPU (llvmpipe / native) — small workloads or CI.
    Cpu,
}

/// Empirical GPU dispatch overhead in microseconds (RTX 4070, Vulkan).
///
/// This is the fixed cost of `wgpu::Queue::submit()` + buffer readback.
/// GPU compute time is negligible at all tested scales.
pub const GPU_DISPATCH_OVERHEAD_US: u64 = 1500;

/// Recommend GPU vs CPU for a pairwise distance computation.
///
/// Based on empirical data from `bench_gpu_kernels`:
/// - Hamming: GPU wins at 200×1000 (CPU ~7 ms), loses at 20×500 (CPU ~34 µs)
/// - Jaccard: GPU wins at 100×2000 (CPU ~8 ms), loses at 30×500 (CPU ~142 µs)
/// - L2: follows same pattern
#[must_use]
pub const fn pairwise_substrate(n_items: usize, item_dim: usize) -> Substrate {
    let n_pairs = n_items * (n_items.saturating_sub(1)) / 2;
    let estimated_work = n_pairs * item_dim;
    if estimated_work > 500_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

/// Recommend GPU vs CPU for batch fitness evaluation.
///
/// - GPU wins when `pop_size * genome_len` exceeds ~50k elements
/// - Below that, CPU loop overhead is less than GPU dispatch overhead
#[must_use]
pub const fn batch_fitness_substrate(pop_size: usize, genome_len: usize) -> Substrate {
    let total_work = pop_size * genome_len;
    if total_work > 50_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

/// Recommend GPU vs CPU for parallel ODE integration.
///
/// - GPU wins when many independent systems are integrated
/// - Below ~100 systems, CPU overhead beats GPU dispatch
#[must_use]
pub const fn ode_substrate(n_systems: usize, n_steps: usize) -> Substrate {
    let total_work = n_systems * n_steps;
    if total_work > 10_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

/// Recommend GPU vs CPU for HMM forward pass.
///
/// - HMM forward is sequential in time steps but parallel across states
/// - GPU wins when `n_states * n_observations` is large
/// - `StatefulPipeline` eliminates dispatch overhead for multi-step chains
#[must_use]
pub const fn hmm_substrate(n_states: usize, n_observations: usize) -> Substrate {
    let total_work = n_states * n_observations;
    if total_work > 5_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

/// Recommend GPU vs CPU for spatial payoff (game theory grid).
///
/// - GPU wins when grid is large (each cell checks 8 Moore neighbors)
/// - Below ~64×64 grids, CPU is faster
#[must_use]
pub const fn spatial_substrate(grid_cells: usize) -> Substrate {
    if grid_cells > 4_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

/// Recommend GPU vs CPU for batch IPR computation.
///
/// - Each eigenvector requires O(n) work for sum of 4th powers
/// - GPU wins when `n_vectors * dim` is large
#[must_use]
pub const fn batch_ipr_substrate(n_vectors: usize, dim: usize) -> Substrate {
    let total_work = n_vectors * dim;
    if total_work > 50_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

/// Recommend GPU vs CPU for batched logsumexp reduction.
///
/// - GPU wins when `batch * width` exceeds ~20k elements
/// - Below that, the sequential per-row scan is faster on CPU
#[must_use]
pub const fn logsumexp_substrate(batch: usize, width: usize) -> Substrate {
    let total_work = batch * width;
    if total_work > 20_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

/// Recommend GPU vs CPU for stochastic population genetics simulation.
///
/// Wright-Fisher drift requires `2N` random draws per (population, locus).
/// GPU wins when `n_pops * n_loci * two_n` is large.
#[must_use]
pub const fn stochastic_substrate(n_pops: usize, n_loci: usize, two_n: usize) -> Substrate {
    let total_work = n_pops * n_loci * two_n;
    if total_work > 100_000 {
        Substrate::Gpu
    } else {
        Substrate::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_pairwise_uses_cpu() {
        assert_eq!(pairwise_substrate(20, 500), Substrate::Cpu);
    }

    #[test]
    fn large_pairwise_uses_gpu() {
        assert_eq!(pairwise_substrate(200, 1000), Substrate::Gpu);
    }

    #[test]
    fn small_fitness_uses_cpu() {
        assert_eq!(batch_fitness_substrate(100, 10), Substrate::Cpu);
    }

    #[test]
    fn large_fitness_uses_gpu() {
        assert_eq!(batch_fitness_substrate(50_000, 64), Substrate::Gpu);
    }

    #[test]
    fn small_ode_uses_cpu() {
        assert_eq!(ode_substrate(10, 100), Substrate::Cpu);
    }

    #[test]
    fn large_ode_uses_gpu() {
        assert_eq!(ode_substrate(1000, 2000), Substrate::Gpu);
    }

    #[test]
    fn small_hmm_uses_cpu() {
        assert_eq!(hmm_substrate(3, 100), Substrate::Cpu);
    }

    #[test]
    fn large_hmm_uses_gpu() {
        assert_eq!(hmm_substrate(3, 5000), Substrate::Gpu);
    }

    #[test]
    fn small_grid_uses_cpu() {
        assert_eq!(spatial_substrate(100), Substrate::Cpu);
    }

    #[test]
    fn large_grid_uses_gpu() {
        assert_eq!(spatial_substrate(10_000), Substrate::Gpu);
    }
}
