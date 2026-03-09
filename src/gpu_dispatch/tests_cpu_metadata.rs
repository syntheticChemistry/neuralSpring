// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path metadata, mixed dispatch, and precision routing tests.

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

// ── Metadata ────────────────────────────────────────────────

#[test]
fn cpu_only_no_gpu() {
    let d = cpu();
    assert!(!d.has_gpu());
    assert_eq!(d.backend(), Backend::Cpu);
    assert!(d.capabilities().is_none());
    assert_eq!(d.adapter_name(), "(none)");
    assert!(d.wgpu_device().is_none());
    assert!(d.gpu().is_none());
}

#[test]
fn backend_display() {
    assert_eq!(format!("{}", Backend::Gpu), "GPU");
    assert_eq!(format!("{}", Backend::Cpu), "CPU");
}

#[test]
fn cpu_fp64_strategy_native() {
    let d = cpu();
    assert_eq!(
        d.fp64_strategy(),
        barracuda::device::driver_profile::Fp64Strategy::Native
    );
}

#[test]
fn cpu_needs_pow_workaround_false() {
    let d = cpu();
    assert!(!d.needs_pow_workaround());
}

#[test]
fn cpu_bandwidth_tier_unknown() {
    let d = cpu();
    assert_eq!(
        d.bandwidth_tier(),
        barracuda::unified_hardware::BandwidthTier::Unknown
    );
}

#[test]
fn cpu_check_allocation_safe_ok() {
    let d = cpu();
    assert!(d.check_allocation_safe(1_000_000).is_ok());
}

#[test]
fn cpu_driver_profile_none() {
    let d = cpu();
    assert!(d.driver_profile().is_none());
}

// ── mixed_dispatch (CPU-only path) ─────────────────────────

#[test]
fn mixed_dispatch_cpu_only_small() {
    let d = cpu();
    let (result, substrate) = d.mixed_dispatch(
        &MixedWorkload {
            op: "test_add",
            compute_us: 1.0,
            data_bytes: 32,
            npu_available: false,
            needs_realtime: false,
        },
        |_dev| Ok(42.0_f64),
        || 42.0_f64,
    );
    assert!((result - 42.0).abs() < tolerances::ZERO_DETECTION);
    assert_eq!(
        substrate,
        neural_spring_forge::mixed::MixedSubstrate::CpuOnly
    );
}

#[test]
fn mixed_dispatch_cpu_only_large() {
    let d = cpu();
    let (result, _substrate) = d.mixed_dispatch(
        &MixedWorkload {
            op: "test_matmul",
            compute_us: 1000.0,
            data_bytes: 8_000_000,
            npu_available: false,
            needs_realtime: false,
        },
        |_dev| Ok(99.0_f64),
        || 99.0_f64,
    );
    assert!((result - 99.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_mixed_dispatch_routes_cpu() {
    let d = cpu();
    let workload = MixedWorkload {
        op: "test_op",
        compute_us: 100.0,
        data_bytes: 1024,
        npu_available: false,
        needs_realtime: false,
    };
    let (result, _substrate) = d.mixed_dispatch(&workload, |_dev| Ok(42.0_f64), || 99.0);
    assert!((result - 99.0).abs() < tolerances::ZERO_DETECTION);
}

// ── Precision routing ──────────────────────────────────────

#[test]
fn cpu_precision_routing_default() {
    let d = cpu();
    assert_eq!(
        d.precision_routing(),
        barracuda::device::driver_profile::PrecisionRoutingAdvice::F64Native
    );
    assert!(d.shared_memory_f64_safe());
}

#[test]
fn cpu_fp64_strategy_default() {
    let d = cpu();
    assert_eq!(
        d.fp64_strategy(),
        barracuda::device::driver_profile::Fp64Strategy::Native
    );
}

#[test]
fn cpu_only_driver_profile_none() {
    let d = cpu();
    assert!(d.driver_profile().is_none());
    assert!(!d.needs_pow_workaround());
    assert!(d.check_allocation_safe(1_000_000).is_ok());
}

#[test]
fn cpu_only_bandwidth_tier_unknown() {
    let d = cpu();
    assert_eq!(
        format!("{:?}", d.bandwidth_tier()),
        format!("{:?}", barracuda::unified_hardware::BandwidthTier::Unknown)
    );
}

#[test]
fn cpu_fp64_strategy_defaults_native() {
    let d = cpu();
    assert_eq!(
        format!("{:?}", d.fp64_strategy()),
        format!(
            "{:?}",
            barracuda::device::driver_profile::Fp64Strategy::Native
        )
    );
}
