// SPDX-License-Identifier: AGPL-3.0-or-later

//! ML workloads with shader provenance tracking.
//!
//! Following wetSpring's `workloads.rs` pattern: each workload declares its
//! required capabilities and shader origin (local, absorbed, or CPU-only).
//! This enables:
//!
//! 1. **Dispatch decisions** — local shaders need `compile_shader_f64`;
//!    absorbed primitives use `ToadStool`'s pre-built pipelines.
//! 2. **Absorption planning** — `ToadStool` sees which domains still use
//!    local implementations and can prioritize absorption.
//! 3. **Validation routing** — local code needs CPU ↔ GPU parity checks;
//!    absorbed primitives are validated upstream.
//!
//! # Write → Absorb → Lean
//!
//! When `ToadStool` absorbs a local extension, update the origin from
//! [`ShaderOrigin::Local`] to [`ShaderOrigin::Absorbed`] and rewire the
//! dispatch to use the upstream primitive.

use crate::substrate::Capability;

/// Where the implementation for a workload lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderOrigin {
    /// Absorbed by `ToadStool` — uses `barracuda::ops::*` or `barracuda::dispatch::*`.
    Absorbed,
    /// Local implementation — pending absorption by `ToadStool`.
    Local,
    /// CPU-only domain — no GPU path exists or is planned.
    CpuOnly,
}

/// An ML workload with provenance tracking.
#[derive(Debug)]
pub struct MlWorkload {
    /// Human-readable workload name.
    pub name: &'static str,
    /// Where the implementation lives.
    pub origin: ShaderOrigin,
    /// `ToadStool` primitive name (if absorbed).
    pub primitive: Option<&'static str>,
    /// Cross-spring origin (which Spring contributed this).
    pub cross_spring_origin: &'static str,
    /// Required capabilities.
    pub required: &'static [Capability],
}

// ── Absorbed: Dispatcher methods (upstream domain_ops) ──────────────

/// matmul dispatch — hotSpring precision (4-tier `KernelRouter`).
pub const MATMUL: MlWorkload = MlWorkload {
    name: "matmul",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("matmul_dispatch"),
    cross_spring_origin: "hotSpring precision",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// softmax dispatch — hotSpring numerics.
pub const SOFTMAX: MlWorkload = MlWorkload {
    name: "softmax",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("softmax_dispatch"),
    cross_spring_origin: "hotSpring numerics",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// gelu dispatch — neuralSpring ML, absorbed S52.
pub const GELU: MlWorkload = MlWorkload {
    name: "gelu",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("gelu_dispatch"),
    cross_spring_origin: "neuralSpring ML → S52",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// mean dispatch — hotSpring reduce.
pub const MEAN: MlWorkload = MlWorkload {
    name: "mean",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("mean_dispatch"),
    cross_spring_origin: "hotSpring reduce",
    required: &[Capability::F64Compute, Capability::ScalarReduce],
};

/// variance — hotSpring Welford algorithm (3.49× faster than f32 Tensor).
pub const VARIANCE: MlWorkload = MlWorkload {
    name: "variance",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("VarianceReduceF64"),
    cross_spring_origin: "hotSpring Welford",
    required: &[Capability::F64Compute, Capability::FusedMapReduce],
};

/// Pearson correlation — wetSpring + hotSpring (f64 precision).
pub const PEARSON: MlWorkload = MlWorkload {
    name: "pearson_correlation",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("CorrelationF64"),
    cross_spring_origin: "wetSpring + hotSpring",
    required: &[Capability::F64Compute, Capability::FusedMapReduce],
};

/// Shannon entropy — wetSpring fused map-reduce (2.56× faster).
pub const ENTROPY: MlWorkload = MlWorkload {
    name: "shannon_entropy",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("FusedMapReduceF64"),
    cross_spring_origin: "wetSpring fused",
    required: &[Capability::F64Compute, Capability::FusedMapReduce],
};

/// HMM forward — wetSpring bio, absorbed S52.
pub const HMM_FORWARD: MlWorkload = MlWorkload {
    name: "hmm_forward",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("hmm_forward_dispatch"),
    cross_spring_origin: "wetSpring bio → S52",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// frobenius norm — hotSpring reduction.
pub const FROBENIUS: MlWorkload = MlWorkload {
    name: "frobenius_norm",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("frobenius_norm_dispatch"),
    cross_spring_origin: "hotSpring reduction",
    required: &[Capability::F64Compute, Capability::ScalarReduce],
};

/// transpose — hotSpring precision.
pub const TRANSPOSE: MlWorkload = MlWorkload {
    name: "transpose",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("transpose_dispatch"),
    cross_spring_origin: "hotSpring precision",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// L2 distance — neuralSpring MODES.
pub const L2_DISTANCE: MlWorkload = MlWorkload {
    name: "l2_distance",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("l2_distance_dispatch"),
    cross_spring_origin: "neuralSpring MODES",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// MHA — neuralSpring evolved, S-03b resolved upstream.
pub const MHA: MlWorkload = MlWorkload {
    name: "multi_head_attention",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("MultiHeadAttention"),
    cross_spring_origin: "neuralSpring → ToadStool S-03b",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

// ── Absorbed: Typed GPU ops ────────────────────────────────────────

/// Batch fitness evaluation — neuralSpring EA (S-25).
pub const BATCH_FITNESS: MlWorkload = MlWorkload {
    name: "batch_fitness",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("BatchFitnessGpu"),
    cross_spring_origin: "neuralSpring EA S-25",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Pairwise L2 — neuralSpring MODES (S-42).
pub const PAIRWISE_L2: MlWorkload = MlWorkload {
    name: "pairwise_l2",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("PairwiseL2Gpu"),
    cross_spring_origin: "neuralSpring MODES S-42",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Pairwise Hamming — neuralSpring `SATé` (S-25).
pub const PAIRWISE_HAMMING: MlWorkload = MlWorkload {
    name: "pairwise_hamming",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("PairwiseHammingGpu"),
    cross_spring_origin: "neuralSpring SATé S-25",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Pairwise Jaccard — neuralSpring pangenome (S-25).
pub const PAIRWISE_JACCARD: MlWorkload = MlWorkload {
    name: "pairwise_jaccard",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("PairwiseJaccardGpu"),
    cross_spring_origin: "neuralSpring pangenome S-25",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Spatial payoff — neuralSpring game theory (S-25).
pub const SPATIAL_PAYOFF: MlWorkload = MlWorkload {
    name: "spatial_payoff",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("SpatialPayoffGpu"),
    cross_spring_origin: "neuralSpring game theory S-25",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Batch IPR — neuralSpring Anderson localization (S-25).
pub const BATCH_IPR: MlWorkload = MlWorkload {
    name: "batch_ipr",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("BatchIprGpu"),
    cross_spring_origin: "neuralSpring Anderson S-25",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Eigensolve — hotSpring NAK-optimized (S-39).
pub const EIGENSOLVE: MlWorkload = MlWorkload {
    name: "eigensolve",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("BatchedEighGpu"),
    cross_spring_origin: "hotSpring nuclear S-39",
    required: &[Capability::F64Compute, Capability::Eigensolve],
};

/// HMM batch forward f64 — wetSpring phylo (S-39).
pub const HMM_BATCH_F64: MlWorkload = MlWorkload {
    name: "hmm_batch_forward_f64",
    origin: ShaderOrigin::Absorbed,
    primitive: Some("HmmBatchForwardF64"),
    cross_spring_origin: "wetSpring phylo S-39",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

// ── Local: Write-phase extensions for ToadStool absorption ──────────

/// Chi-squared GPU — local elementwise `(o-e)²/e` + sum reduction.
///
/// Currently CPU loop in `gpu_ops::reduction::chi_squared_gpu`. Should be
/// a fused GPU op similar to `FusedMapReduceF64`.
pub const CHI_SQUARED_GPU: MlWorkload = MlWorkload {
    name: "chi_squared_gpu",
    origin: ShaderOrigin::Local,
    primitive: None,
    cross_spring_origin: "neuralSpring validation S-64",
    required: &[Capability::F64Compute, Capability::FusedMapReduce],
};

/// KL divergence GPU — local `p*ln(p/q)` sum reduction.
///
/// Currently CPU loop in `gpu_ops::reduction::kl_divergence_gpu`. Should be
/// a fused GPU op.
pub const KL_DIVERGENCE_GPU: MlWorkload = MlWorkload {
    name: "kl_divergence_gpu",
    origin: ShaderOrigin::Local,
    primitive: None,
    cross_spring_origin: "neuralSpring validation S-64",
    required: &[Capability::F64Compute, Capability::FusedMapReduce],
};

/// HMM backward step — CPU fallback loop in `gpu_dispatch`.
///
/// Needs an upstream `hmm_backward_dispatch` in `domain_ops`.
pub const HMM_BACKWARD: MlWorkload = MlWorkload {
    name: "hmm_backward",
    origin: ShaderOrigin::Local,
    primitive: None,
    cross_spring_origin: "neuralSpring HMM S-46",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// HMM Viterbi step — CPU fallback loop + argmax.
///
/// Needs an upstream `hmm_viterbi_dispatch` in `domain_ops`.
pub const HMM_VITERBI: MlWorkload = MlWorkload {
    name: "hmm_viterbi",
    origin: ShaderOrigin::Local,
    primitive: None,
    cross_spring_origin: "neuralSpring HMM S-46",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Pairwise L2 matrix — CPU O(n²) loop over `l2_distance_gpu`.
///
/// Should use `PairwiseL2Gpu` directly for the full matrix.
pub const PAIRWISE_L2_MATRIX: MlWorkload = MlWorkload {
    name: "pairwise_l2_matrix",
    origin: ShaderOrigin::Local,
    primitive: None,
    cross_spring_origin: "neuralSpring MODES S-42",
    required: &[Capability::F64Compute, Capability::ShaderDispatch],
};

/// Replicator dynamics step — small GEMV + update.
pub const REPLICATOR_STEP: MlWorkload = MlWorkload {
    name: "replicator_step",
    origin: ShaderOrigin::Local,
    primitive: None,
    cross_spring_origin: "neuralSpring game theory S-46",
    required: &[Capability::F64Compute],
};

// ── CPU-only domains ───────────────────────────────────────────────

/// Pareto front counting — inherently sequential O(n²) dominance check.
pub const PARETO_FRONT: MlWorkload = MlWorkload {
    name: "pareto_front",
    origin: ShaderOrigin::CpuOnly,
    primitive: None,
    cross_spring_origin: "neuralSpring directed evolution",
    required: &[Capability::CpuCompute],
};

/// Mantel test — permutation + correlation (sequential).
pub const MANTEL_TEST: MlWorkload = MlWorkload {
    name: "mantel_test",
    origin: ShaderOrigin::CpuOnly,
    primitive: None,
    cross_spring_origin: "neuralSpring meta-population",
    required: &[Capability::CpuCompute],
};

// ── Inventory ──────────────────────────────────────────────────────

/// All known ML domain workloads.
///
/// Returns the full catalog for dispatch planning and absorption tracking.
#[must_use]
pub fn all_workloads() -> Vec<&'static MlWorkload> {
    vec![
        // Absorbed: Dispatcher methods (9 upstream domain_ops)
        &MATMUL,
        &SOFTMAX,
        &GELU,
        &MEAN,
        &VARIANCE,
        &PEARSON,
        &ENTROPY,
        &HMM_FORWARD,
        &FROBENIUS,
        &TRANSPOSE,
        &L2_DISTANCE,
        &MHA,
        // Absorbed: Typed GPU ops
        &BATCH_FITNESS,
        &PAIRWISE_L2,
        &PAIRWISE_HAMMING,
        &PAIRWISE_JACCARD,
        &SPATIAL_PAYOFF,
        &BATCH_IPR,
        &EIGENSOLVE,
        &HMM_BATCH_F64,
        // Write phase: local extensions
        &CHI_SQUARED_GPU,
        &KL_DIVERGENCE_GPU,
        &HMM_BACKWARD,
        &HMM_VITERBI,
        &PAIRWISE_L2_MATRIX,
        &REPLICATOR_STEP,
        // CPU-only
        &PARETO_FRONT,
        &MANTEL_TEST,
    ]
}

/// Count workloads by shader origin.
#[must_use]
pub fn origin_summary() -> (usize, usize, usize) {
    let all = all_workloads();
    let absorbed = all.iter().filter(|w| w.origin == ShaderOrigin::Absorbed).count();
    let local = all.iter().filter(|w| w.origin == ShaderOrigin::Local).count();
    let cpu_only = all.iter().filter(|w| w.origin == ShaderOrigin::CpuOnly).count();
    (absorbed, local, cpu_only)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_workloads_has_entries() {
        let all = all_workloads();
        assert!(all.len() >= 28, "expected at least 28 workloads");
    }

    #[test]
    fn origin_counts_match() {
        let (absorbed, local, cpu_only) = origin_summary();
        assert_eq!(absorbed, 20, "20 absorbed workloads");
        assert_eq!(local, 6, "6 local write-phase extensions");
        assert_eq!(cpu_only, 2, "2 CPU-only domains");
    }

    #[test]
    fn absorbed_workloads_have_primitive() {
        for w in all_workloads() {
            if w.origin != ShaderOrigin::Absorbed {
                continue;
            }
            assert!(
                w.primitive.is_some(),
                "{} should have primitive name",
                w.name
            );
        }
    }

    #[test]
    fn local_workloads_have_no_primitive() {
        for w in all_workloads() {
            if w.origin != ShaderOrigin::Local {
                continue;
            }
            assert!(
                w.primitive.is_none(),
                "{} should not have primitive name (not yet absorbed)",
                w.name
            );
        }
    }

    #[test]
    fn all_workloads_no_duplicate_names() {
        let all = all_workloads();
        let mut names: Vec<&str> = all.iter().map(|w| w.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate workload names found");
    }
}
