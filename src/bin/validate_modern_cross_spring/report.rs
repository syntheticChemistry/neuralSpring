// SPDX-License-Identifier: AGPL-3.0-or-later

// Short provenance summary for stdout.

pub fn report_provenance_summary() {
    println!("\n═══ Cross-Spring Provenance: 5 springs → BarraCUDA → sovereign GPU ═══\n");
    println!(
        "  hotSpring    → DF64/Fp64Strategy/lattice QCD/nautilus → eigh, eigensolve, precision"
    );
    println!(
        "  wetSpring    → diversity/HMM/NMF/ODE bio/chao1       → alpha_diversity, FST chains"
    );
    println!("  airSpring    → regression/hydrology/metrics           → ET₀ (5 methods), fit_*");
    println!("  groundSpring → bootstrap/multinomial/jackknife        → bootstrap_ci, norm_*");
    println!("  neuralSpring → batch_fitness/pairwise/eigh/swarm_nn  → Dispatcher (47 ops)");
    println!("  BarraCUDA v0.3.5: 719+ WGSL, 144 ComputeDispatch ops, nautilus absorbed S80");
    println!();
}
