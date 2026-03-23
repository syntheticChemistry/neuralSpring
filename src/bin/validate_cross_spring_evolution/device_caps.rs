// SPDX-License-Identifier: AGPL-3.0-or-later

//! `DeviceCapabilities` / fp64 strategy validation.

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::validation::ValidationHarness;

pub fn validate_device_capabilities(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    if let Some(caps) = dispatcher.device_caps() {
        println!("[caps] Device: {}", caps.device_name);
        println!("[caps] Type: {:?}", caps.device_type);
        println!("[caps] Backend: {:?}", caps.backend);
        println!("[caps] FP64 strategy: {:?}", caps.fp64_strategy());
        println!("[caps] exp workaround: {}", caps.needs_exp_f64_workaround());
        println!(
            "[caps] Eigensolve strategy: {:?}",
            caps.optimal_eigensolve_strategy()
        );

        h.check_bool("device capabilities detected", true);

        let strategy = dispatcher.fp64_strategy();
        let strategy_valid = matches!(
            strategy,
            barracuda::device::capabilities::Fp64Strategy::Native
                | barracuda::device::capabilities::Fp64Strategy::Hybrid
                | barracuda::device::capabilities::Fp64Strategy::Concurrent
        );
        h.check_bool("fp64 strategy valid", strategy_valid);
    } else {
        println!("[caps] No GPU — skipping device capabilities checks");
        h.check_bool("device capabilities (no GPU, skip)", true);
    }
}
