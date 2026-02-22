// SPDX-License-Identifier: AGPL-3.0-or-later

//! Binding layout and dispatch geometry for each WGSL shader.
//!
//! `ToadStool` can copy these structs directly when absorbing a shader into
//! `barracuda::ops::*`. Each [`ShaderLayout`] documents:
//!
//! - Buffer bindings (group 0) with their role and element type
//! - Workgroup size
//! - Dispatch geometry (how to compute workgroup counts from problem dimensions)
//! - Entry point name

/// Describes one buffer binding in a shader.
#[derive(Debug, Clone, Copy)]
pub struct BufferBinding {
    /// Binding index within group 0.
    pub binding: u32,
    /// Human-readable role (e.g. "population", "fitness").
    pub role: &'static str,
    /// WGSL element type (`f32`, `u32`, `array<vec4<f32>>`).
    pub element_type: &'static str,
    /// Whether this is a uniform buffer (params) or storage buffer.
    pub is_uniform: bool,
}

/// Complete layout for one shader, sufficient for `ToadStool` absorption.
#[derive(Debug, Clone)]
pub struct ShaderLayout {
    /// Shader name matching [`super::shaders`] constant name.
    pub name: &'static str,
    /// WGSL entry point function name.
    pub entry_point: &'static str,
    /// Workgroup size (x, y, z).
    pub workgroup_size: [u32; 3],
    /// Buffer bindings in group 0.
    pub bindings: &'static [BufferBinding],
    /// How to compute dispatch workgroup count from problem size N.
    /// For most shaders: `ceil(N / workgroup_size.x)`.
    pub dispatch_note: &'static str,
}

const fn storage(binding: u32, role: &'static str, element_type: &'static str) -> BufferBinding {
    BufferBinding {
        binding,
        role,
        element_type,
        is_uniform: false,
    }
}

const fn uniform(binding: u32, role: &'static str) -> BufferBinding {
    BufferBinding {
        binding,
        role,
        element_type: "struct",
        is_uniform: true,
    }
}

/// HMM forward pass (log-domain).
pub const HMM_FORWARD_LOG: ShaderLayout = ShaderLayout {
    name: "HMM_FORWARD_LOG",
    entry_point: "hmm_forward_log",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "alpha_prev", "array<f32>"),
        storage(1, "log_trans", "array<f32>"),
        storage(2, "log_emit", "array<f32>"),
        storage(3, "alpha_curr", "array<f32>"),
        uniform(4, "HmmParams {n_states}"),
    ],
    dispatch_note: "ceil(n_states / 256)",
};

/// Batch linear fitness evaluation.
pub const BATCH_FITNESS_EVAL: ShaderLayout = ShaderLayout {
    name: "BATCH_FITNESS_EVAL",
    entry_point: "batch_fitness_linear",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "population", "array<f32>"),
        storage(1, "weights", "array<f32>"),
        storage(2, "fitness", "array<f32>"),
        uniform(3, "FitnessParams {pop_size, genome_len}"),
    ],
    dispatch_note: "ceil(pop_size / 256)",
};

/// Parallel RK4 ODE integration.
pub const RK4_PARALLEL: ShaderLayout = ShaderLayout {
    name: "RK4_PARALLEL",
    entry_point: "rk4_step",
    workgroup_size: [64, 1, 1],
    bindings: &[
        storage(0, "state", "array<f32>"),
        storage(1, "coeffs", "array<f32>"),
        storage(2, "state_out", "array<f32>"),
        uniform(3, "OdeParams {n_systems, n_dims, dt, n_steps}"),
        storage(4, "scratch", "array<f32>"),
    ],
    dispatch_note: "ceil(n_systems / 64)",
};

/// Scalar mean reduction.
pub const MEAN_REDUCE: ShaderLayout = ShaderLayout {
    name: "MEAN_REDUCE",
    entry_point: "mean_reduce",
    workgroup_size: [1, 1, 1],
    bindings: &[
        storage(0, "values", "array<f32>"),
        storage(1, "result", "array<f32>"),
        uniform(2, "ReduceParams {count}"),
    ],
    dispatch_note: "1 (single workgroup)",
};

/// Pairwise Jaccard distance.
pub const PAIRWISE_JACCARD: ShaderLayout = ShaderLayout {
    name: "PAIRWISE_JACCARD",
    entry_point: "pairwise_jaccard",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "pa", "array<u32>"),
        storage(1, "distances", "array<f32>"),
        uniform(2, "JaccardParams {n_genomes, n_genes}"),
    ],
    dispatch_note: "ceil(n_pairs / 256) where n_pairs = n_genomes*(n_genomes-1)/2",
};

/// Per-locus allele frequency variance.
pub const LOCUS_VARIANCE: ShaderLayout = ShaderLayout {
    name: "LOCUS_VARIANCE",
    entry_point: "locus_variance",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "allele_freqs", "array<f32>"),
        storage(1, "per_locus_var", "array<f32>"),
        uniform(2, "VarianceParams {n_loci, n_pops}"),
    ],
    dispatch_note: "ceil(n_loci / 256)",
};

/// Spatial payoff (Moore neighborhood).
pub const SPATIAL_PAYOFF: ShaderLayout = ShaderLayout {
    name: "SPATIAL_PAYOFF",
    entry_point: "spatial_payoff",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "grid", "array<u32>"),
        storage(1, "fitness", "array<f32>"),
        uniform(2, "Params {width, height, R, S, T, P}"),
    ],
    dispatch_note: "ceil(width * height / 256)",
};

/// Batch inverse participation ratio.
pub const BATCH_IPR: ShaderLayout = ShaderLayout {
    name: "BATCH_IPR",
    entry_point: "batch_ipr",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "eigenvectors", "array<f32>"),
        storage(1, "ipr_out", "array<f32>"),
        uniform(2, "Params {n_vectors, dim}"),
    ],
    dispatch_note: "ceil(n_vectors / 256)",
};

/// Pairwise Hamming distance.
pub const PAIRWISE_HAMMING: ShaderLayout = ShaderLayout {
    name: "PAIRWISE_HAMMING",
    entry_point: "pairwise_hamming",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "sequences", "array<u32>"),
        storage(1, "distances", "array<f32>"),
        uniform(2, "Params {n_seqs, seq_len}"),
    ],
    dispatch_note: "ceil(n_pairs / 256) where n_pairs = n_seqs*(n_seqs-1)/2",
};

/// Pairwise L2 (Euclidean) distance.
pub const PAIRWISE_L2: ShaderLayout = ShaderLayout {
    name: "PAIRWISE_L2",
    entry_point: "pairwise_l2",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "features", "array<f32>"),
        storage(1, "distances", "array<f32>"),
        uniform(2, "Params {n_items, dim}"),
    ],
    dispatch_note: "ceil(n_pairs / 256) where n_pairs = n_items*(n_items-1)/2",
};

/// Multi-objective fitness evaluation.
pub const MULTI_OBJ_FITNESS: ShaderLayout = ShaderLayout {
    name: "MULTI_OBJ_FITNESS",
    entry_point: "multi_obj_fitness",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "genotypes", "array<f32>"),
        storage(1, "fitness", "array<f32>"),
        uniform(2, "Params {pop_size, genome_len, n_objectives, chunk_size}"),
    ],
    dispatch_note: "ceil(pop_size * n_objectives / 256)",
};

/// Batch swarm NN forward pass.
pub const SWARM_NN_FORWARD: ShaderLayout = ShaderLayout {
    name: "SWARM_NN_FORWARD",
    entry_point: "swarm_nn_forward",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "params", "array<f32>"),
        storage(1, "inputs", "array<f32>"),
        storage(2, "actions", "array<f32>"),
        uniform(
            3,
            "Config {n_controllers, n_evals, input_dim, hidden_dim, output_dim}",
        ),
    ],
    dispatch_note: "ceil(n_controllers * n_evals / 256)",
};

/// Two-input Hill function AND gate.
pub const HILL_GATE: ShaderLayout = ShaderLayout {
    name: "HILL_GATE",
    entry_point: "hill_gate",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "cdg_grid", "array<f32>"),
        storage(1, "ai_grid", "array<f32>"),
        storage(2, "output", "array<f32>"),
        uniform(3, "HillParams {n_cdg, n_ai, k_cdg, k_ai, n_hill, v_max}"),
    ],
    dispatch_note: "ceil(n_cdg * n_ai / 256)",
};

/// GPU head split for multi-head attention.
pub const HEAD_SPLIT: ShaderLayout = ShaderLayout {
    name: "HEAD_SPLIT",
    entry_point: "head_split",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "input", "array<f32>"),
        storage(1, "output", "array<f32>"),
        uniform(2, "Params {batch, seq_len, n_heads, d_head}"),
    ],
    dispatch_note: "ceil(batch * seq_len * n_heads * d_head / 256)",
};

/// GPU head concatenation for multi-head attention.
pub const HEAD_CONCAT: ShaderLayout = ShaderLayout {
    name: "HEAD_CONCAT",
    entry_point: "head_concat",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "input", "array<f32>"),
        storage(1, "output", "array<f32>"),
        uniform(2, "Params {batch, seq_len, n_heads, d_head}"),
    ],
    dispatch_note: "ceil(batch * seq_len * n_heads * d_head / 256)",
};

/// GPU-parallel PRNG (Xoshiro128**).
pub const XOSHIRO128SS: ShaderLayout = ShaderLayout {
    name: "XOSHIRO128SS",
    entry_point: "generate",
    workgroup_size: [256, 1, 1],
    bindings: &[
        storage(0, "state", "array<u32>"),
        storage(1, "output", "array<f32>"),
        uniform(2, "Params {n_threads}"),
    ],
    dispatch_note: "ceil(n_threads / 256)",
};

/// All shader layouts, for iteration.
pub const ALL: &[&ShaderLayout] = &[
    &HMM_FORWARD_LOG,
    &BATCH_FITNESS_EVAL,
    &RK4_PARALLEL,
    &MEAN_REDUCE,
    &PAIRWISE_JACCARD,
    &LOCUS_VARIANCE,
    &SPATIAL_PAYOFF,
    &BATCH_IPR,
    &PAIRWISE_HAMMING,
    &PAIRWISE_L2,
    &MULTI_OBJ_FITNESS,
    &SWARM_NN_FORWARD,
    &HILL_GATE,
    &HEAD_SPLIT,
    &HEAD_CONCAT,
    &XOSHIRO128SS,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_layouts_have_entry_point() {
        for layout in ALL {
            assert!(
                !layout.entry_point.is_empty(),
                "{} missing entry_point",
                layout.name
            );
        }
    }

    #[test]
    fn all_layouts_have_bindings() {
        for layout in ALL {
            assert!(
                !layout.bindings.is_empty(),
                "{} has no bindings",
                layout.name
            );
        }
    }

    #[test]
    fn layout_count_matches_shaders() {
        assert_eq!(ALL.len(), 16);
    }
}
