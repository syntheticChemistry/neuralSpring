// SPDX-License-Identifier: AGPL-3.0-or-later

//! Meta-population and pangenome scenario builder (Papers 024-025).
//!
//! Produces 2 nodes: FST population structure with heatmap and
//! pangenome gene frequency analysis with diversity metrics.

#![expect(
    clippy::cast_precision_loss,
    reason = "index-to-f64 conversions for labels and coordinates"
)]

use crate::meta_population::{fst::fst_matrix, generate_population, nucleotide_diversity};
use crate::pangenome_selection::{
    frequency_spectrum, gene_frequencies, gene_repertoire_diversity, generate_pa_matrix,
    neutral_spectrum, partition_pangenome,
};
use crate::rng::Rng;
use crate::visualization::types::{NeuralScenario, ScenarioEdge};

use super::{bar, edge, gauge, heatmap, node, scaffold, scatter3d};

/// Build the population genetics scenario.
///
/// Nodes:
/// - `meta_pop`: pairwise FST heatmap + diversity metrics
/// - `pangenome`: gene frequency spectrum + core/accessory/singleton partition
#[expect(
    clippy::too_many_lines,
    reason = "scenario builder — single cohesive builder"
)]
#[must_use]
pub fn population_study() -> (NeuralScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Population Genetics & Pangenome",
        "FST population structure, nucleotide diversity, pangenome gene frequency analysis",
    );

    let n_pops = 4;
    let n_ind = 20;
    let n_loci = 50;
    let mut rng = Rng::new(42);

    let mut populations = Vec::with_capacity(n_pops);
    let fst_targets = [0.05, 0.10, 0.15, 0.20];
    let ancestral: Vec<f64> = vec![0.3; n_loci];
    for &fst in &fst_targets {
        let pop = generate_population(
            n_ind, n_loci, &ancestral, fst, 25.0, 15.0, 35.0, 5, &mut rng,
        );
        populations.push(pop);
    }

    let n_inds: Vec<usize> = vec![n_ind; n_pops];
    let fst_mat = fst_matrix(&populations, &n_inds, n_loci);
    let pop_labels: Vec<String> = (0..n_pops).map(|i| format!("Pop{i}")).collect();

    let diversities: Vec<f64> = populations
        .iter()
        .map(|p| nucleotide_diversity(p, n_ind, n_loci))
        .collect();

    let coords: Vec<(f64, f64)> = fst_targets
        .iter()
        .enumerate()
        .map(|(i, &fst)| (i as f64 * 100.0, fst * 1000.0))
        .collect();
    let geo_x: Vec<f64> = coords.iter().map(|(x, _)| *x).collect();
    let geo_y: Vec<f64> = coords.iter().map(|(_, y)| *y).collect();
    let geo_z = diversities.clone();

    s.ecosystem.primals.push(node(
        "meta_pop",
        "Meta-Population Structure",
        "compute",
        0.0,
        0.0,
        &["science.meta_population", "science.fst"],
        vec![
            heatmap(
                "fst-matrix",
                "Pairwise FST Matrix",
                pop_labels.clone(),
                pop_labels,
                fst_mat,
                "FST",
            ),
            bar(
                "nucleotide-diversity",
                "Nucleotide Diversity (π)",
                (0..n_pops).map(|i| format!("Pop{i}")).collect(),
                diversities,
                "π",
            ),
            scatter3d(
                "pop-geography",
                "Population Geography (x, FST*1000, π)",
                "mixed",
                geo_x,
                geo_y,
                geo_z,
                (0..n_pops).map(|i| format!("Pop{i}")).collect(),
            ),
        ],
        vec![],
    ));

    let n_genomes = 20;
    let n_genes = 100;
    let env_labels: Vec<usize> = (0..n_genomes).map(|i| usize::from(i >= 10)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.15, 0.25, &mut rng, &env_labels);
    let freqs = gene_frequencies(&pa, n_genes, n_genomes);
    let (core, accessory, singleton) = partition_pangenome(&freqs, n_genomes, 0.95);
    let obs_spectrum = frequency_spectrum(&freqs, 10);
    let _neut_spectrum = neutral_spectrum(10);

    let bin_labels: Vec<String> = (1..=10).map(|b| format!("Bin{b}")).collect();
    let diversity = gene_repertoire_diversity(&pa, n_genes, n_genomes);

    s.ecosystem.primals.push(node(
        "pangenome",
        "Pangenome Analysis",
        "compute",
        400.0,
        0.0,
        &["science.pangenome_selection"],
        vec![
            bar(
                "gene-partition",
                "Core / Accessory / Singleton",
                vec!["Core".into(), "Accessory".into(), "Singleton".into()],
                vec![core as f64, accessory as f64, singleton as f64],
                "genes",
            ),
            bar(
                "frequency-spectrum",
                "Gene Frequency Spectrum (observed vs neutral)",
                bin_labels,
                obs_spectrum,
                "fraction",
            ),
            gauge(
                "repertoire-diversity",
                "Gene Repertoire Diversity",
                diversity,
                0.0,
                1.0,
                "Shannon H'",
                [0.5, 1.0],
                [0.2, 0.5],
            ),
        ],
        vec![],
    ));

    let edges = vec![edge(
        "meta_pop",
        "pangenome",
        "population structure → gene flow",
    )];
    (s, edges)
}
