// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 099: HMM Introgression Detection on NN Weight Layers.
//!
//! Novel composition of nS-04 (introgression HMM) applied to neural network
//! layer statistics. Adjacent layers with similar weight distributions are
//! "concordant"; layers with abrupt shifts are "introgressed."
//!
//! Composes:
//! - [`crate::hmm`] — forward, Viterbi algorithms
//! - [`crate::introgression`] — HMM parameterization concept

use crate::hmm::Hmm;

/// Baseline loaded from Python JSON.
#[derive(Debug, Clone)]
pub struct IntrogressionNnBaseline {
    pub n_layers: usize,
    pub n_introgressed: usize,
    pub hmm: Hmm,
    pub observations: Vec<usize>,
    pub true_states: Vec<usize>,
    pub viterbi_path: Vec<usize>,
    pub tpr: f64,
    pub fpr: f64,
    pub accuracy: f64,
    pub introgression_fraction: f64,
    pub log_likelihood_full: f64,
    pub log_likelihood_null: f64,
    pub llr: f64,
}

/// Build the NN-introgression HMM (matching Python baseline).
#[must_use]
pub fn build_nn_hmm() -> Hmm {
    let transition = vec![vec![0.92, 0.08], vec![0.08, 0.92]];
    let emission = vec![vec![0.80, 0.15, 0.05], vec![0.05, 0.10, 0.85]];
    let initial = vec![0.80, 0.20];
    Hmm::new(transition, emission, initial)
}

/// Build null model (single normal state).
#[must_use]
pub fn build_null_hmm() -> Hmm {
    let transition = vec![vec![1.0]];
    let emission = vec![vec![0.80, 0.15, 0.05]];
    let initial = vec![1.0];
    Hmm::new(transition, emission, initial)
}

/// Compute introgression fraction from Viterbi path.
#[must_use]
pub fn introgression_fraction(path: &[usize]) -> f64 {
    if path.is_empty() {
        return 0.0;
    }
    let n_introg = path.iter().filter(|&&s| s == 1).count();
    n_introg as f64 / path.len() as f64
}

/// Compute detection metrics: (TPR, FPR, accuracy).
#[must_use]
pub fn detection_metrics(path: &[usize], truth: &[usize]) -> (f64, f64, f64) {
    let n = path.len();
    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..n {
        match (path[i], truth[i]) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }
    let n_pos = (tp + fn_).max(1);
    let n_neg = (tn + fp).max(1);
    (
        tp as f64 / n_pos as f64,
        fp as f64 / n_neg as f64,
        (tp + tn) as f64 / n as f64,
    )
}

/// Load baseline from Python JSON.
///
/// # Errors
///
/// Returns `Err` if JSON structure is unexpected.
pub fn load_introgression_nn_from_json(
    json_str: &str,
) -> Result<IntrogressionNnBaseline, String> {
    let v: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("parse: {e}"))?;

    let hmm_v = &v["hmm"];
    let transition = parse_2d(hmm_v, "transition")?;
    let emission = parse_2d(hmm_v, "emission")?;
    let initial = parse_1d(hmm_v, "initial")?;
    let hmm = Hmm::new(transition, emission, initial);

    let observations = parse_usize_array(&v, "observations")?;
    let true_states = parse_usize_array(&v, "true_states")?;
    let viterbi_path = parse_usize_array(&v, "viterbi_path")?;

    let m = &v["metrics"];

    Ok(IntrogressionNnBaseline {
        n_layers: v["n_layers"].as_u64().ok_or("n_layers")? as usize,
        n_introgressed: v["n_introgressed"].as_u64().ok_or("n_introg")? as usize,
        hmm,
        observations,
        true_states,
        viterbi_path,
        tpr: m["tpr"].as_f64().ok_or("tpr")?,
        fpr: m["fpr"].as_f64().ok_or("fpr")?,
        accuracy: m["accuracy"].as_f64().ok_or("accuracy")?,
        introgression_fraction: m["introgression_fraction"].as_f64().ok_or("frac")?,
        log_likelihood_full: m["log_likelihood_full"].as_f64().ok_or("ll_full")?,
        log_likelihood_null: m["log_likelihood_null"].as_f64().ok_or("ll_null")?,
        llr: m["llr"].as_f64().ok_or("llr")?,
    })
}

fn parse_2d(v: &serde_json::Value, key: &str) -> Result<Vec<Vec<f64>>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("missing {key}"))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| format!("{key} not 2D"))
                .and_then(|r| {
                    r.iter()
                        .map(|x| x.as_f64().ok_or_else(|| "not f64".to_string()))
                        .collect()
                })
        })
        .collect()
}

fn parse_1d(v: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("missing {key}"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| "not f64".to_string()))
        .collect()
}

fn parse_usize_array(v: &serde_json::Value, key: &str) -> Result<Vec<usize>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("missing {key}"))?
        .iter()
        .map(|x| x.as_u64().ok_or_else(|| "not u64".to_string()).map(|n| n as usize))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_hmm_states() {
        let hmm = build_nn_hmm();
        let obs = vec![0, 0, 0, 2, 2, 2, 0, 0];
        let (path, log_prob) = hmm.viterbi(&obs);
        assert_eq!(path.len(), obs.len());
        assert!(path[3] == 1 || path[4] == 1, "should detect introgression at obs=2 block");
        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_detection_metrics() {
        let path = vec![0, 0, 1, 1, 0];
        let truth = vec![0, 0, 1, 0, 0];
        let (tpr, fpr, acc) = detection_metrics(&path, &truth);
        assert!((tpr - 1.0).abs() < 1e-10);
        assert!((fpr - 0.25).abs() < 1e-10);
        assert!((acc - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_introgression_fraction() {
        assert!((introgression_fraction(&[0, 0, 1, 0]) - 0.25).abs() < 1e-10);
        assert!((introgression_fraction(&[0, 0, 0, 0]) - 0.0).abs() < 1e-10);
    }
}
