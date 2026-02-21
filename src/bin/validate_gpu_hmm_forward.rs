// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: HMM forward pass via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/hmm_forward_log.wgsl` against the CPU
//! reference in `src/hmm.rs`.  The GPU shader uses f32 log-domain
//! arithmetic; the CPU reference uses f64 scaled forward.  Both should
//! produce equivalent log-likelihood values within documented tolerance.
//!
//! Evolution path:
//! ```text
//! Python (hmmlearn) → Rust CPU (hmm.rs) → BarraCUDA CPU (stats/linalg)
//!   → GPU WGSL shader (hmm_forward_log.wgsl) → ToadStool absorption
//! ```
//!
//! ## Papers validated
//!
//! - Paper 016: HMM Forward/Backward/Viterbi (Liu et al., 2014)
//! - Paper 017: `SATé` Alignment (Liu et al., 2009)
//! - Paper 018: Introgression Detection (Liu et al., 2015)
//!
//! ## Backend selection
//!
//! Set `NEURALSPRING_BACKEND=cpu|gpu|auto`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names,
    clippy::too_many_lines
)]

use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            eprintln!("  SKIP: {e} — no GPU/CPU adapter available");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };

    let mut h = ValidationHarness::new("gpu_hmm_forward");

    validate_2state_weather(&mut h, &gpu);
    validate_3state_genomic(&mut h, &gpu);
    validate_log_likelihood_sign(&mut h, &gpu);
    validate_alpha_sum_property(&mut h, &gpu);
    validate_longer_sequence(&mut h, &gpu);

    h.finish();
}

/// 2-state weather HMM (same as `validate_barracuda_hmm.rs`).
fn weather_hmm() -> Hmm {
    Hmm::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]],
        vec![0.6, 0.4],
    )
}

/// 3-state genomic HMM (simplified PhyloNet-HMM from Paper 016).
fn genomic_hmm() -> Hmm {
    Hmm::new(
        vec![
            vec![0.8, 0.1, 0.1],
            vec![0.05, 0.9, 0.05],
            vec![0.1, 0.1, 0.8],
        ],
        vec![
            vec![0.4, 0.3, 0.2, 0.1],
            vec![0.1, 0.4, 0.3, 0.2],
            vec![0.2, 0.1, 0.4, 0.3],
        ],
        vec![0.33, 0.34, 0.33],
    )
}

/// Convert Hmm parameters to f32 log-domain for GPU shader.
fn hmm_to_log_f32(hmm: &Hmm) -> (Vec<f32>, Vec<f32>) {
    let n = hmm.num_states();
    let log_initial: Vec<f32> = hmm.initial.iter().map(|&p| (p as f32).ln()).collect();
    let log_trans: Vec<f32> = hmm
        .transition
        .iter()
        .flat_map(|row| row.iter().map(|&p| (p as f32).ln()))
        .collect();
    let _ = n;
    (log_initial, log_trans)
}

/// Convert observation sequence to f32 log-emission matrix (T × N).
fn obs_to_log_emissions(hmm: &Hmm, obs: &[usize]) -> Vec<f32> {
    let n = hmm.num_states();
    let m = hmm.emission[0].len();
    obs.iter()
        .flat_map(|&o| {
            let oi = o.min(m - 1);
            (0..n).map(move |j| (hmm.emission[j][oi] as f32).ln())
        })
        .collect()
}

fn validate_2state_weather(h: &mut ValidationHarness, gpu: &Gpu) {
    let hmm = weather_hmm();
    let mut rng = Rng::new(42);
    let (_, obs) = hmm.generate_sequence(20, &mut rng);

    let (cpu_alpha, cpu_ll) = hmm.forward(&obs);
    let (log_initial, log_trans) = hmm_to_log_f32(&hmm);
    let log_emissions = obs_to_log_emissions(&hmm, &obs);

    match neural_spring::evolved::hmm_forward_gpu::hmm_forward_gpu(
        gpu,
        &log_initial,
        &log_trans,
        &log_emissions,
    ) {
        Ok(output) => {
            match output.readback(gpu) {
                Ok(gpu_alpha) => {
                    // GPU alpha is in log-domain; CPU alpha is in probability domain (scaled).
                    // Compare: GPU log-alpha vs CPU log(alpha * product(scales))
                    // For a simpler check: verify GPU alpha values are finite and ordered correctly.
                    let all_finite = gpu_alpha.iter().all(|v| v.is_finite());
                    h.check_bool("2-state weather: GPU alpha all finite", all_finite);

                    // GPU log-likelihood = logsumexp(final alpha)
                    let max_a = gpu_alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let gpu_ll: f32 = max_a
                        + gpu_alpha
                            .iter()
                            .map(|&a| (a - max_a).exp())
                            .sum::<f32>()
                            .ln();

                    h.check_bool(
                        &format!("2-state weather: GPU LL finite ({gpu_ll:.4})"),
                        gpu_ll.is_finite(),
                    );
                    h.check_bool(
                        "2-state weather: GPU LL negative (probability < 1)",
                        gpu_ll < 0.0,
                    );

                    #[allow(clippy::cast_possible_truncation)]
                    let cpu_ll_f32 = cpu_ll as f32;
                    h.check_abs(
                        &format!(
                            "2-state weather: GPU LL ≈ CPU LL ({gpu_ll:.4} vs {cpu_ll_f32:.4})"
                        ),
                        f64::from(gpu_ll),
                        f64::from(cpu_ll_f32),
                        tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
                    );

                    // Check CPU reference is sane
                    h.check_bool(
                        &format!("2-state weather: CPU LL finite ({cpu_ll:.4})"),
                        cpu_ll.is_finite(),
                    );

                    let _ = cpu_alpha;
                }
                Err(e) => {
                    h.check_bool(&format!("2-state weather: readback failed — {e}"), false);
                }
            }
        }
        Err(e) => {
            h.check_bool(
                &format!("2-state weather: GPU dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_3state_genomic(h: &mut ValidationHarness, gpu: &Gpu) {
    let hmm = genomic_hmm();
    let mut rng = Rng::new(123);
    let (_, obs) = hmm.generate_sequence(30, &mut rng);

    let (_, cpu_ll) = hmm.forward(&obs);
    let (log_initial, log_trans) = hmm_to_log_f32(&hmm);
    let log_emissions = obs_to_log_emissions(&hmm, &obs);

    match neural_spring::evolved::hmm_forward_gpu::hmm_forward_gpu(
        gpu,
        &log_initial,
        &log_trans,
        &log_emissions,
    ) {
        Ok(output) => match output.readback(gpu) {
            Ok(gpu_alpha) => {
                let all_finite = gpu_alpha.iter().all(|v| v.is_finite());
                h.check_bool("3-state genomic: GPU alpha all finite", all_finite);

                let max_a = gpu_alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let gpu_ll: f32 = max_a
                    + gpu_alpha
                        .iter()
                        .map(|&a| (a - max_a).exp())
                        .sum::<f32>()
                        .ln();

                h.check_bool(
                    &format!("3-state genomic: GPU LL finite ({gpu_ll:.4})"),
                    gpu_ll.is_finite(),
                );

                #[allow(clippy::cast_possible_truncation)]
                let cpu_ll_f32 = cpu_ll as f32;
                h.check_abs(
                    &format!("3-state genomic: GPU LL ≈ CPU LL ({gpu_ll:.4} vs {cpu_ll_f32:.4})"),
                    f64::from(gpu_ll),
                    f64::from(cpu_ll_f32),
                    tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
                );
            }
            Err(e) => {
                h.check_bool(&format!("3-state genomic: readback failed — {e}"), false);
            }
        },
        Err(e) => {
            h.check_bool(
                &format!("3-state genomic: GPU dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_log_likelihood_sign(h: &mut ValidationHarness, gpu: &Gpu) {
    let hmm = weather_hmm();
    let obs_short: Vec<usize> = vec![0, 1, 2, 0, 1];
    let obs_long: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];

    let (log_initial, log_trans) = hmm_to_log_f32(&hmm);

    let log_emit_short = obs_to_log_emissions(&hmm, &obs_short);
    let log_emit_long = obs_to_log_emissions(&hmm, &obs_long);

    let ll_short = gpu_log_likelihood(gpu, &log_initial, &log_trans, &log_emit_short);
    let ll_long = gpu_log_likelihood(gpu, &log_initial, &log_trans, &log_emit_long);

    if let (Some(s), Some(l)) = (ll_short, ll_long) {
        h.check_bool(&format!("LL sign: short > long ({s:.4} > {l:.4})"), s > l);
    } else {
        h.check_bool("LL sign: dispatch failed", false);
    }
}

fn validate_alpha_sum_property(h: &mut ValidationHarness, gpu: &Gpu) {
    let hmm = weather_hmm();
    let obs: Vec<usize> = vec![0, 1, 0, 2, 1];

    let (log_initial, log_trans) = hmm_to_log_f32(&hmm);
    let log_emissions = obs_to_log_emissions(&hmm, &obs);

    match neural_spring::evolved::hmm_forward_gpu::hmm_forward_gpu(
        gpu,
        &log_initial,
        &log_trans,
        &log_emissions,
    ) {
        Ok(output) => match output.readback(gpu) {
            Ok(gpu_alpha) => {
                // In log-domain, "sum" = logsumexp.  The total should be finite.
                let max_a = gpu_alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = max_a
                    + gpu_alpha
                        .iter()
                        .map(|&a| (a - max_a).exp())
                        .sum::<f32>()
                        .ln();
                h.check_bool(
                    &format!("alpha sum property: logsumexp finite ({lse:.4})"),
                    lse.is_finite(),
                );
                h.check_bool(
                    "alpha sum property: logsumexp negative (prob < 1)",
                    lse < 0.0,
                );
            }
            Err(e) => {
                h.check_bool(&format!("alpha sum: readback failed — {e}"), false);
            }
        },
        Err(e) => {
            h.check_bool(&format!("alpha sum: dispatch failed — {e}"), false);
        }
    }
}

fn validate_longer_sequence(h: &mut ValidationHarness, gpu: &Gpu) {
    let hmm = genomic_hmm();
    let mut rng = Rng::new(999);
    let (_, obs) = hmm.generate_sequence(100, &mut rng);

    let (_, cpu_ll) = hmm.forward(&obs);
    let (log_initial, log_trans) = hmm_to_log_f32(&hmm);
    let log_emissions = obs_to_log_emissions(&hmm, &obs);

    match neural_spring::evolved::hmm_forward_gpu::hmm_forward_gpu(
        gpu,
        &log_initial,
        &log_trans,
        &log_emissions,
    ) {
        Ok(output) => match output.readback(gpu) {
            Ok(gpu_alpha) => {
                let max_a = gpu_alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let gpu_ll: f32 = max_a
                    + gpu_alpha
                        .iter()
                        .map(|&a| (a - max_a).exp())
                        .sum::<f32>()
                        .ln();

                #[allow(clippy::cast_possible_truncation)]
                let cpu_ll_f32 = cpu_ll as f32;
                h.check_abs(
                    &format!("100-obs genomic: GPU LL ≈ CPU LL ({gpu_ll:.4} vs {cpu_ll_f32:.4})"),
                    f64::from(gpu_ll),
                    f64::from(cpu_ll_f32),
                    tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
                );

                h.check_bool(
                    &format!("100-obs genomic: GPU LL finite ({gpu_ll:.4})"),
                    gpu_ll.is_finite(),
                );
            }
            Err(e) => {
                h.check_bool(&format!("100-obs: readback failed — {e}"), false);
            }
        },
        Err(e) => {
            h.check_bool(&format!("100-obs: dispatch failed — {e}"), false);
        }
    }
}

fn gpu_log_likelihood(
    gpu: &Gpu,
    log_initial: &[f32],
    log_trans: &[f32],
    log_emissions: &[f32],
) -> Option<f32> {
    let output = neural_spring::evolved::hmm_forward_gpu::hmm_forward_gpu(
        gpu,
        log_initial,
        log_trans,
        log_emissions,
    )
    .ok()?;
    let alpha = output.readback(gpu).ok()?;
    let max_a = alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Some(max_a + alpha.iter().map(|&a| (a - max_a).exp()).sum::<f32>().ln())
}
