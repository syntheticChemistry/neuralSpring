// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring provenance registry tests.

#[test]
fn provenance_registry_has_neuralspring_shaders() {
    use barracuda::shaders::provenance::{shaders_from, SpringDomain};
    let ns = shaders_from(SpringDomain::NEURAL_SPRING);
    assert!(
        !ns.is_empty(),
        "neuralSpring should have provenance entries"
    );
}

#[test]
fn provenance_registry_has_hotspring_math() {
    use barracuda::shaders::provenance::{shaders_from, SpringDomain};
    let hs = shaders_from(SpringDomain::HOT_SPRING);
    assert!(
        hs.len() >= 5,
        "hotSpring should have ≥5 provenance entries (precision, spectral, md)"
    );
    assert!(
        hs.iter().any(|s| s.path.contains("df64_core")),
        "hotSpring should include df64_core.wgsl"
    );
}

#[test]
fn provenance_cross_spring_matrix_non_empty() {
    use barracuda::shaders::provenance::cross_spring_matrix;
    let matrix = cross_spring_matrix();
    assert!(
        !matrix.is_empty(),
        "cross-spring matrix should be non-empty"
    );
}

#[test]
fn provenance_evolution_report_has_sections() {
    use barracuda::shaders::provenance::evolution_report;
    let report = evolution_report();
    assert!(report.contains("Timeline"));
    assert!(report.contains("Dependency Matrix"));
    assert!(report.contains("hotSpring"));
    assert!(report.contains("neuralSpring"));
    assert!(report.contains("wetSpring"));
}

#[test]
fn provenance_neuralspring_consumed_by_others() {
    use barracuda::shaders::provenance::{shaders_from, SpringDomain};
    let ns = shaders_from(SpringDomain::NEURAL_SPRING);
    assert!(
        ns.iter().any(|s| s
            .consumers
            .iter()
            .any(|c| *c != SpringDomain::NEURAL_SPRING)),
        "neuralSpring shaders should be consumed by other springs"
    );
}

#[test]
fn provenance_hotspring_df64_consumed_by_neuralspring() {
    use barracuda::shaders::provenance::{shaders_from, SpringDomain};
    let hs = shaders_from(SpringDomain::HOT_SPRING);
    assert!(
        hs.iter()
            .any(|s| s.consumers.contains(&SpringDomain::NEURAL_SPRING)),
        "hotSpring shaders should be consumed by neuralSpring (DF64, precision)"
    );
}

#[test]
fn provenance_wetspring_bio_shaders_exist() {
    use barracuda::shaders::provenance::{shaders_from, SpringDomain};
    let ws = shaders_from(SpringDomain::WET_SPRING);
    assert!(
        !ws.is_empty(),
        "wetSpring should have provenance entries (bio shaders)"
    );
}
