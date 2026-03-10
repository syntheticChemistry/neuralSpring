// SPDX-License-Identifier: AGPL-3.0-or-later

//! Live spectral training dashboard for neuralSpring.
//!
//! Discovers petalTongue via IPC, pushes the training scenario as the
//! initial render, then runs a simulated training loop that streams
//! epoch-by-epoch spectral diagnostics via [`TrainingVisualizer`].
//!
//! This is the "anyone with a GPU and some curiosity" entry point:
//! watch eigenvalues move as a neural network trains.
//!
//! ## Usage
//!
//! ```text
//! # Start petalTongue first
//! petaltongue ui
//!
//! # Then run the dashboard
//! cargo run --bin neuralspring_live_dashboard
//!
//! # Or with custom epoch count / interval
//! EPOCHS=200 INTERVAL_MS=100 cargo run --bin neuralspring_live_dashboard
//! ```
//!
//! ## Environment variables
//!
//! - `EPOCHS`: number of simulated training epochs (default: 100)
//! - `INTERVAL_MS`: milliseconds between epoch pushes (default: 50)

#![expect(
    clippy::cast_precision_loss,
    reason = "epoch/grid indices as f64 for visualization axes"
)]

use std::thread;
use std::time::Duration;

use neural_spring::rng::Rng;
use neural_spring::training_monitor::{AttentionState, TrainingMonitor, TrainingVisualizer};
use neural_spring::visualization::ipc_push::PetalTonguePushClient;
use neural_spring::visualization::stream::StreamSession;
use neural_spring::visualization::training_study;
use neural_spring::weight_spectral::{SpectralPhase, WeightSpectralResult};

fn main() {
    let n_epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let interval_ms: u64 = std::env::var("INTERVAL_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    println!("neuralSpring Live Dashboard");
    println!("  epochs:      {n_epochs}");
    println!("  interval:    {interval_ms}ms");
    println!();

    let client = match PetalTonguePushClient::discover() {
        Ok(c) => {
            println!("Discovered petalTongue via IPC");
            c
        }
        Err(e) => {
            eprintln!("petalTongue not found ({e}) — running in headless mode (stats only)");
            PetalTonguePushClient::headless()
        }
    };

    let (scenario, _edges) = training_study();
    let session = match StreamSession::start(
        client,
        "live-training",
        "Live Training Dashboard",
        &scenario,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Initial render failed ({e}), continuing in headless mode");
            StreamSession::resume(PetalTonguePushClient::headless(), "live-training")
        }
    };
    let viz = TrainingVisualizer::new(session);
    let mut monitor = TrainingMonitor::new();
    let mut rng = Rng::new(42);

    println!("Streaming {n_epochs} epochs...\n");

    for epoch in 0..n_epochs {
        let (loss, spectral) = simulate_epoch(epoch, &mut rng);
        monitor.observe_epoch(epoch, loss, &spectral);

        if monitor.should_check(epoch) {
            let interrupt = monitor.check_interrupt();
            let state = monitor.attention();

            let _ = viz.on_epoch(epoch, &spectral, state);

            let state_str = match state {
                AttentionState::Green => "GREEN",
                AttentionState::Yellow => "YELLOW",
                AttentionState::Red => "RED",
            };
            print!(
                "\repoch {epoch:>4} | loss {loss:.6} | BW {bw:.4} | IPR {ipr:.4} | [{state_str}]",
                bw = spectral.bandwidth,
                ipr = spectral.mean_ipr,
            );

            if interrupt != neural_spring::training_monitor::TrainingInterrupt::Continue {
                println!("\n  interrupt: {interrupt:?}");
            }
        }

        thread::sleep(Duration::from_millis(interval_ms));
    }

    let stats = viz.session().stats();
    println!("\n\nSession complete:");
    println!("  messages: {}", stats.messages_sent);
    println!("  bytes:    {}", stats.bytes_sent);
    println!("  errors:   {}", stats.errors);
    println!("  uptime:   {}ms", stats.uptime_ms);
    println!("  throughput: {:.1} msg/s", stats.messages_per_second());
}

fn simulate_epoch(epoch: usize, rng: &mut Rng) -> (f64, WeightSpectralResult) {
    let t = epoch as f64;

    let base_loss = 2.0f64.mul_add((-0.03 * t).exp(), 0.1);
    let noise = (rng.uniform() - 0.5) * 0.02;
    let loss = (base_loss + noise).max(0.0);

    let bandwidth = rng
        .uniform()
        .mul_add(0.1, 0.5f64.mul_add(1.0 - (-0.02 * t).exp(), 1.0));
    let ipr = rng
        .uniform()
        .mul_add(0.02, 0.5f64.mul_add((-0.005 * t).exp(), 0.1));
    let entropy = rng
        .uniform()
        .mul_add(0.05, 0.3f64.mul_add((t / 50.0).min(1.0), 2.0));
    let lsr = rng.uniform().mul_add(0.02, 0.53);
    let cond = rng.uniform().mul_add(2.0, t.mul_add(0.5, 10.0));

    let n_evals: i32 = 20;
    let eigenvalues: Vec<f64> = (0..n_evals)
        .map(|i| {
            let base = f64::from(i).mul_add(4.0 / f64::from(n_evals), -2.0);
            base.mul_add(bandwidth, rng.uniform() * 0.1)
        })
        .collect();

    let spectral = WeightSpectralResult {
        eigenvalues,
        mean_ipr: ipr,
        level_spacing_ratio: lsr,
        spectral_entropy: entropy,
        mp_departure: rng.uniform().mul_add(0.05, 0.1),
        bandwidth,
        condition_number: cond,
        phase: if ipr < 0.15 {
            SpectralPhase::Localized
        } else if bandwidth > 1.8 {
            SpectralPhase::Critical
        } else {
            SpectralPhase::Extended
        },
    };

    (loss, spectral)
}
