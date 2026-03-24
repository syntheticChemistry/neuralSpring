// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tower phase: map forge capability strings to local neuralSpring computations (CPU and GPU paths).

#![expect(
    clippy::cast_precision_loss,
    reason = "timing and dimension values fit in f64"
)]

use neural_spring_forge::graph::StageOutput;

use crate::gpu_dispatch::Dispatcher;
use crate::tolerances;

/// Tower: resolve a capability string to a local computation function (CPU path).
///
/// Returns `(success, output)`. Each capability maps to a real neuralSpring
/// module function. Unknown capabilities return `(false, Empty)`.
pub(super) fn dispatch_capability(capability: &str) -> (bool, StageOutput) {
    match capability {
        "science.eigensolve" => stage_eigensolve(),
        "science.digester_anderson_coupling" => stage_digester_anderson(),
        "science.isomorphic_reservoir" => stage_isomorphic_reservoir(),
        "science.wdm_ensemble_qs" => stage_wdm_ensemble_qs(),
        "science.introgression_nn" => stage_introgression_nn(),
        "science.attention_anderson" => stage_attention_anderson(),
        _ => (false, StageOutput::Empty),
    }
}

/// Tower: resolve a capability to GPU-accelerated dispatch.
///
/// GPU stages use the `Dispatcher` for eigensolve and spectral ops.
/// Non-GPU stages fall through to the CPU path.
pub(super) fn dispatch_capability_gpu(
    capability: &str,
    dispatcher: &Dispatcher,
) -> (bool, StageOutput) {
    match capability {
        "science.eigensolve" => stage_eigensolve_gpu(dispatcher),
        "science.attention_anderson" => stage_attention_anderson_gpu(dispatcher),
        "science.digester_anderson_coupling" => stage_digester_anderson(),
        "science.isomorphic_reservoir" => stage_isomorphic_reservoir(),
        "science.wdm_ensemble_qs" => stage_wdm_ensemble_qs(),
        "science.introgression_nn" => stage_introgression_nn(),
        _ => (false, StageOutput::Empty),
    }
}

fn stage_eigensolve() -> (bool, StageOutput) {
    let n = 16;
    let mut matrix = vec![0.0; n * n];
    for i in 0..n {
        matrix[i * n + i] = 1.0;
    }
    let result = crate::eigh::eigh_householder_qr(&matrix, n);
    let sum: f64 = result.eigenvalues.iter().sum();
    (
        (sum - n as f64).abs() < tolerances::SPECIAL_FUNCTION_F64,
        StageOutput::Vector(result.eigenvalues),
    )
}

fn stage_eigensolve_gpu(dispatcher: &Dispatcher) -> (bool, StageOutput) {
    let n = 16;
    let mut matrix = vec![0.0; n * n];
    for i in 0..n {
        matrix[i * n + i] = 1.0;
    }
    let (eigenvalues, _eigenvectors) = dispatcher.eigh(&matrix, n);
    let sum: f64 = eigenvalues.iter().sum();
    (
        (sum - n as f64).abs() < tolerances::SPECIAL_FUNCTION_F64,
        StageOutput::Vector(eigenvalues),
    )
}

fn stage_digester_anderson() -> (bool, StageOutput) {
    let mut rng = crate::rng::Rng::new(42);
    let n_species = 10;
    let (h, evenness, w, ipr, xi) =
        crate::digester_anderson::community_anderson(n_species, 1.0, 20, &mut rng);

    let mut map = std::collections::HashMap::new();
    map.insert("shannon_h".to_string(), h);
    map.insert("evenness".to_string(), evenness);
    map.insert("disorder_w".to_string(), w);
    map.insert("mean_ipr".to_string(), ipr);
    map.insert("xi".to_string(), xi);

    let valid = h > 0.0 && (0.0..=1.0).contains(&ipr);
    (valid, StageOutput::Map(map))
}

fn stage_isomorphic_reservoir() -> (bool, StageOutput) {
    let n = 16;
    let mut rng = crate::rng::Rng::new(42);
    let mut matrices = Vec::new();

    for gain in [0.9, 0.85, 0.95] {
        let mut m = vec![0.0; n * n];
        for val in &mut m {
            *val = rng.uniform().mul_add(2.0, -1.0) * gain / (n as f64).sqrt();
        }
        let sym: Vec<f64> = (0..n * n)
            .map(|idx| {
                let r = idx / n;
                let c = idx % n;
                (m[r * n + c] + m[c * n + r]) * 0.5
            })
            .collect();
        matrices.push(sym);
    }

    let profiles: Vec<_> = matrices
        .iter()
        .zip(["esn", "glucose", "weather"])
        .map(|(m, name)| crate::isomorphic_reservoir::spectral_properties(m, n, name))
        .collect();

    let cdm = crate::isomorphic_reservoir::cross_domain_metrics(&profiles);

    let mut map = std::collections::HashMap::new();
    map.insert("eff_ratio_cv".to_string(), cdm.eff_ratio_cv);
    map.insert("ipr_cv".to_string(), cdm.ipr_cv);
    map.insert("spacing_ratio_mean".to_string(), cdm.spacing_ratio_mean);

    let valid = cdm.eff_ratio_cv < 0.5 && cdm.ipr_cv < 0.5;
    (valid, StageOutput::Map(map))
}

/// WDM ensemble QS stage domain parameters.
const WDM_DISAGREEMENT_INPUT: f64 = 0.5;
const WDM_DISAGREEMENT_MIN: f64 = 0.01;
const WDM_DISAGREEMENT_MAX: f64 = 1.0;
const WDM_W_SCALE: f64 = 16.0;
const WDM_DISORDER_SAMPLES: usize = 20;
const WDM_REPLICATOR_STEPS: usize = 500;

fn stage_wdm_ensemble_qs() -> (bool, StageOutput) {
    let mut rng = crate::rng::Rng::new(42);
    let w = crate::wdm_ensemble_qs::disagreement_to_disorder(
        WDM_DISAGREEMENT_INPUT,
        WDM_DISAGREEMENT_MIN,
        WDM_DISAGREEMENT_MAX,
        WDM_W_SCALE,
    );

    let disorder_vec: Vec<f64> = (0..WDM_DISORDER_SAMPLES)
        .map(|_| rng.uniform() * w)
        .collect();
    let (ipr, xi) = crate::wdm_ensemble_qs::anderson_from_disorder(&disorder_vec);

    let w_frac = (w / WDM_W_SCALE).clamp(0.0, 1.0);
    let payoff = crate::wdm_ensemble_qs::snowdrift_payoff(w_frac);
    let coop = crate::wdm_ensemble_qs::replicator_final_coop(&payoff, WDM_REPLICATOR_STEPS);

    let mut map = std::collections::HashMap::new();
    map.insert("disorder".to_string(), w);
    map.insert("mean_ipr".to_string(), ipr);
    map.insert("xi".to_string(), xi);
    map.insert("cooperation".to_string(), coop);

    let valid = ipr >= 0.0 && (0.0..=1.0).contains(&coop);
    (valid, StageOutput::Map(map))
}

fn stage_introgression_nn() -> (bool, StageOutput) {
    let hmm = crate::introgression_nn::build_nn_hmm();
    let null_hmm = crate::introgression_nn::build_null_hmm();
    let n_layers = 50;

    let mut truth = vec![0_usize; n_layers];
    for t in &mut truth[15..30] {
        *t = 1;
    }

    let mut rng = crate::rng::Rng::new(42);
    let obs: Vec<usize> = truth
        .iter()
        .map(|&s| {
            if s == 1 {
                2
            } else {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "rng in [0,2) → usize"
                )]
                let v = (rng.uniform() * 2.0) as usize;
                v
            }
        })
        .collect();

    let (path, _) = hmm.viterbi(&obs);
    let (tpr, fpr, accuracy) = crate::introgression_nn::detection_metrics(&path, &truth);

    let (_, log_lik_model) = hmm.forward(&obs);
    let (_, log_lik_null) = null_hmm.forward(&obs);

    let mut map = std::collections::HashMap::new();
    map.insert("tpr".to_string(), tpr);
    map.insert("fpr".to_string(), fpr);
    map.insert("accuracy".to_string(), accuracy);
    map.insert("llr".to_string(), log_lik_model - log_lik_null);

    let valid = tpr > 0.5 && accuracy > 0.5;
    (valid, StageOutput::Map(map))
}

fn build_attention_matrix(n: usize) -> Vec<f64> {
    let mut rng = crate::rng::Rng::new(42);
    let quality = 0.8;
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        let mut row_vals = Vec::with_capacity(n);
        for col in 0..n {
            let base = if row == col { quality } else { 1.0 - quality };
            row_vals.push(rng.uniform().mul_add(0.1, base));
        }
        let sum: f64 = row_vals.iter().sum();
        for (col, val) in row_vals.into_iter().enumerate() {
            matrix[row * n + col] = val / sum;
        }
    }
    (0..n * n)
        .map(|idx| {
            let r = idx / n;
            let c = idx % n;
            (matrix[r * n + c] + matrix[c * n + r]) * 0.5
        })
        .collect()
}

fn stage_attention_anderson() -> (bool, StageOutput) {
    let n = 16;
    let sym = build_attention_matrix(n);
    let result = crate::attention_anderson::attention_spectral(&sym, n);

    let mut map = std::collections::HashMap::new();
    map.insert("quality".to_string(), result.quality);
    map.insert("entropy".to_string(), result.entropy);
    map.insert("mean_ipr".to_string(), result.mean_ipr);
    map.insert("spectral_radius".to_string(), result.spectral_radius);
    map.insert("participation".to_string(), result.participation);

    let valid = result.spectral_radius > 0.0 && result.participation > 0.0;
    (valid, StageOutput::Map(map))
}

fn stage_attention_anderson_gpu(dispatcher: &Dispatcher) -> (bool, StageOutput) {
    let n = 16;
    let sym = build_attention_matrix(n);

    let spectral = dispatcher.attention_spectral_analysis(&sym, n);

    let mean_ipr = spectral.mean_ipr;
    let lsr = spectral.level_spacing_ratio;
    let spectral_radius = spectral
        .eigenvalues
        .iter()
        .map(|e| e.abs())
        .fold(0.0_f64, f64::max);
    let participation = if mean_ipr > 0.0 { 1.0 / mean_ipr } else { 0.0 };

    let mut map = std::collections::HashMap::new();
    map.insert("mean_ipr".to_string(), mean_ipr);
    map.insert("spectral_radius".to_string(), spectral_radius);
    map.insert("participation".to_string(), participation);
    map.insert("level_spacing_ratio".to_string(), lsr);

    let valid = spectral_radius > 0.0 && participation > 0.0;
    (valid, StageOutput::Map(map))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions use expect for clear messages"
)]
mod tests {
    use super::*;
    use neural_spring_forge::graph::StageOutput;

    use crate::gpu_dispatch::Dispatcher;

    #[test]
    fn eigensolve_stage_produces_correct_eigenvalues() {
        let (success, output) = dispatch_capability("science.eigensolve");
        assert!(success);
        if let StageOutput::Vector(evals) = output {
            assert_eq!(evals.len(), 16);
            for &e in &evals {
                assert!(
                    (e - 1.0).abs() < crate::tolerances::GELU_LARGE_INPUT,
                    "identity matrix eigenvalue should be 1.0"
                );
            }
        } else {
            panic!("eigensolve should produce Vector output");
        }
    }

    #[test]
    fn unknown_capability_fails() {
        let (success, output) = dispatch_capability("science.nonexistent");
        assert!(!success);
        assert!(matches!(output, StageOutput::Empty));
    }

    #[test]
    fn digester_anderson_produces_valid_metrics() {
        let (success, output) = dispatch_capability("science.digester_anderson_coupling");
        assert!(success);
        if let StageOutput::Map(m) = output {
            assert!(m.contains_key("shannon_h"));
            assert!(m.contains_key("mean_ipr"));
            assert!(*m.get("mean_ipr").expect("mean_ipr key missing") >= 0.0);
            assert!(*m.get("mean_ipr").expect("mean_ipr key missing") <= 1.0);
        } else {
            panic!("expected Map output");
        }
    }

    #[test]
    fn introgression_detects_anomalous_layers() {
        let (success, output) = dispatch_capability("science.introgression_nn");
        assert!(success);
        if let StageOutput::Map(m) = output {
            assert!(
                *m.get("tpr").expect("tpr key missing") > 0.5,
                "TPR should be > 0.5"
            );
            assert!(
                *m.get("accuracy").expect("accuracy key missing") > 0.5,
                "accuracy should be > 0.5"
            );
        } else {
            panic!("expected Map output");
        }
    }

    #[test]
    fn gpu_pipeline_eigensolve_via_dispatcher() {
        let dispatcher = Dispatcher::cpu_only();
        let (success, output) = dispatch_capability_gpu("science.eigensolve", &dispatcher);
        assert!(success);
        if let StageOutput::Vector(evals) = output {
            assert_eq!(evals.len(), 16);
            for &e in &evals {
                assert!((e - 1.0).abs() < crate::tolerances::GELU_LARGE_INPUT);
            }
        } else {
            panic!("expected Vector output from GPU eigensolve");
        }
    }

    #[test]
    fn gpu_pipeline_attention_via_dispatcher() {
        let dispatcher = Dispatcher::cpu_only();
        let (success, output) = dispatch_capability_gpu("science.attention_anderson", &dispatcher);
        assert!(success);
        if let StageOutput::Map(m) = output {
            assert!(m.contains_key("spectral_radius"));
            assert!(m.contains_key("mean_ipr"));
            assert!(*m.get("spectral_radius").expect("has spectral_radius") > 0.0);
        } else {
            panic!("expected Map output from GPU attention");
        }
    }

    #[test]
    fn dispatch_gpu_unknown_capability_returns_empty() {
        let dispatcher = Dispatcher::cpu_only();
        let (success, output) =
            dispatch_capability_gpu("science.unknown_capability_xyz", &dispatcher);
        assert!(!success);
        assert!(matches!(output, StageOutput::Empty));
    }

    #[test]
    fn wdm_ensemble_qs_stage_runs() {
        let (success, output) = dispatch_capability("science.wdm_ensemble_qs");
        assert!(success);
        let StageOutput::Map(m) = output else {
            panic!("expected Map");
        };
        assert!(m.contains_key("disorder"));
        assert!(m.contains_key("cooperation"));
    }

    #[test]
    fn isomorphic_reservoir_stage_runs() {
        let (success, output) = dispatch_capability("science.isomorphic_reservoir");
        assert!(success);
        let StageOutput::Map(m) = output else {
            panic!("expected Map");
        };
        assert!(m.contains_key("eff_ratio_cv"));
    }
}
