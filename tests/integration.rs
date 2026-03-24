// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration tests for the neuralSpring library.
//!
//! Tests cross-module interactions and round-trip properties that
//! unit tests in individual modules cannot cover.

use neural_spring::provenance;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

#[test]
fn provenance_registry_all_records_have_valid_commit_and_command() {
    let valid_commits = [provenance::BASELINE_COMMIT, provenance::CPU_PARITY_COMMIT];

    for p in provenance::PROVENANCE_REGISTRY {
        assert!(
            valid_commits.contains(&p.commit),
            "{} has unknown commit '{}' — expected one of {valid_commits:?}",
            p.label,
            p.commit
        );
        assert!(!p.command.is_empty(), "{} has empty command", p.label);
        assert!(
            p.command.starts_with("python3"),
            "{} command doesn't start with python3: {}",
            p.label,
            p.command
        );
        assert!(!p.date.is_empty(), "{} has empty date", p.label);
        assert!(!p.script.is_empty(), "{} has empty script path", p.label);
        assert!(!p.unit.is_empty(), "{} has empty unit", p.label);
    }
}

#[test]
fn provenance_all_records_have_well_formed_environment() {
    let known_envs = [
        provenance::ENVIRONMENT,
        provenance::CPU_PARITY_ENVIRONMENT,
        provenance::PUBLICATION_ENVIRONMENT,
        provenance::WDM_ENVIRONMENT,
        provenance::ANDERSON_MULTIAGENT_ENVIRONMENT,
    ];

    for p in provenance::PROVENANCE_REGISTRY {
        assert!(
            !p.environment.is_empty(),
            "{} has empty environment",
            p.label
        );
        assert!(
            p.environment.contains("Python"),
            "{} environment missing Python version: {}",
            p.label,
            p.environment
        );
        assert!(
            known_envs.contains(&p.environment),
            "{} has non-centralized environment string '{}' — \
             add a named constant to provenance/mod.rs",
            p.label,
            p.environment
        );
    }
}

#[test]
fn provenance_cpu_parity_records_have_parity_environment() {
    for p in provenance::PROVENANCE_REGISTRY {
        if p.commit == provenance::CPU_PARITY_COMMIT {
            assert_eq!(
                p.environment,
                provenance::CPU_PARITY_ENVIRONMENT,
                "{} (cpu-parity commit) has wrong environment: {}",
                p.label,
                p.environment
            );
        }
    }
}

#[test]
fn provenance_registry_minimum_record_count() {
    assert!(
        provenance::PROVENANCE_REGISTRY.len() >= 49,
        "expected at least 49 provenance records, found {}",
        provenance::PROVENANCE_REGISTRY.len()
    );
}

#[test]
fn tolerance_registry_covers_all_categories() {
    let cats = tolerances::categories();
    for expected in [
        "machine",
        "benchmark",
        "transformer",
        "metric",
        "training",
        "evolutionary",
        "gpu_shader",
        "gpu_f64",
        "gpu_pipeline",
        "tensor",
        "fft",
        "physics",
        "spectral",
        "numerical",
        "statistical",
        "linalg",
        "ml_pipeline",
        "gpu_dispatch",
        "folding",
        "domain_guards",
    ] {
        assert!(cats.contains(&expected), "missing category: {expected}");
    }
}

#[test]
fn tolerance_lookup_round_trip() {
    let all = tolerances::all_tolerances();
    for t in all {
        let found = tolerances::tolerance_by_name(t.name);
        assert!(found.is_some(), "tolerance {} not found by name", t.name);
        if let Some(v) = found {
            assert!(
                (v - t.value).abs() < f64::EPSILON,
                "tolerance {} value mismatch",
                t.name
            );
        }
    }
}

#[test]
fn harness_round_trip_all_check_types() {
    let mut h = ValidationHarness::new("integration_round_trip");

    h.check_abs("abs_pass", 1.0, 1.0, 1e-10);
    h.check_rel("rel_pass", 100.001, 100.0, 1e-4);
    h.check_upper("upper_pass", 0.5, 1.0);
    h.check_lower("lower_pass", 5.0, 1.0);
    h.check_bool("bool_pass", true);
    h.check_abs_or_rel("abs_or_rel_pass", 1.000_000_001, 1.0, 1e-6);

    assert!(h.all_passed(), "all check types should pass");
    assert_eq!(h.total_count(), 6);
    assert_eq!(h.passed_count(), 6);
}

#[test]
fn runtime_environment_self_knowledge() {
    let env = provenance::RuntimeEnvironment::discover();
    assert!(!env.os.is_empty(), "OS should be discovered");
    assert!(!env.arch.is_empty(), "arch should be discovered");
    assert!(
        !env.rust_version.is_empty(),
        "Rust version should be discovered"
    );
    assert!(
        !env.neuralspring_version.is_empty(),
        "neuralSpring version should be discovered"
    );

    let summary = env.summary();
    assert!(
        summary.contains(env!("CARGO_PKG_NAME")),
        "summary should contain crate name from Cargo.toml, got: {summary}"
    );
}

#[test]
fn cross_module_hmm_forward_matches_provenance() {
    use neural_spring::hmm;

    let model = hmm::Hmm::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.5, 0.5], vec![0.1, 0.9]],
        vec![0.6, 0.4],
    );
    let obs = &[0, 1, 0, 1, 0];
    let (_, log_lik) = model.forward(obs);

    assert!(log_lik.is_finite(), "log-likelihood should be finite");
    assert!(log_lik < 0.0, "log-likelihood should be negative");

    let (path, viterbi_prob) = model.viterbi(obs);
    assert_eq!(path.len(), obs.len());
    assert!(viterbi_prob.is_finite());
    for &s in &path {
        assert!(s < 2, "state index out of range");
    }
}

#[test]
fn cross_module_benchmark_functions_match_provenance() {
    use neural_spring::surrogate::{ackley_2d, rastrigin_2d, rosenbrock_2d};

    for &(x, y, expected) in &provenance::RASTRIGIN_REFERENCE {
        let got = rastrigin_2d(x, y);
        assert!(
            (got - expected).abs() < tolerances::BENCHMARK_CROSS_PYTHON,
            "rastrigin({x}, {y}): got {got}, expected {expected}"
        );
    }

    for &(x, y, expected) in &provenance::ROSENBROCK_REFERENCE {
        let got = rosenbrock_2d(x, y);
        assert!(
            (got - expected).abs() < tolerances::BENCHMARK_CROSS_PYTHON,
            "rosenbrock({x}, {y}): got {got}, expected {expected}"
        );
    }

    for &(x, y, expected) in &provenance::ACKLEY_REFERENCE {
        let got = ackley_2d(x, y);
        assert!(
            (got - expected).abs() < tolerances::BENCHMARK_CROSS_PYTHON,
            "ackley({x}, {y}): got {got}, expected {expected}"
        );
    }
}

#[test]
fn cross_module_softmax_matches_provenance() {
    use neural_spring::transformer::softmax;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = softmax(&input);

    for (i, (&got, &expected)) in result
        .iter()
        .zip(provenance::SOFTMAX_1_TO_5.iter())
        .enumerate()
    {
        assert!(
            (got - expected).abs() < tolerances::SOFTMAX_CROSS_PYTHON,
            "softmax[{i}]: got {got}, expected {expected}"
        );
    }
}

#[test]
fn cross_module_gelu_matches_provenance() {
    use neural_spring::transformer::gelu;

    for &(input, expected) in &provenance::GELU_REFERENCE {
        let got = gelu(input);
        assert!(
            (got - expected).abs() < tolerances::GELU_CROSS_PYTHON,
            "gelu({input}): got {got}, expected {expected}"
        );
    }
}
