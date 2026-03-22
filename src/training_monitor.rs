// SPDX-License-Identifier: AGPL-3.0-or-later

//! Training intelligence: adaptive monitoring with brain-inspired interrupts.
//!
//! Adapts hotSpring's 4-layer brain architecture interrupt pattern for
//! neural network training. Monitors spectral properties of weight matrices
//! during training to detect phase transitions, divergence, and drift.
//!
//! ## Cross-Spring Provenance
//!
//! ```text
//! hotSpring BrainInterrupt (GREEN/YELLOW/RED) → neuralSpring TrainingMonitor
//! hotSpring DriftMonitor (N_e*s detection)     → training drift detection
//! hotSpring HeadGroupDisagreement              → phase boundary signal
//! ```
//!
//! ## Attention State Machine
//!
//! Adapted from `hotSpring/specs/BIOMEGATE_BRAIN_ARCHITECTURE.md`:
//!
//! | State  | Check interval | Trigger |
//! |--------|---------------|---------|
//! | Green  | Every 10 epochs | Normal training |
//! | Yellow | Every 3 epochs  | Bandwidth increasing or loss stalling |
//! | Red    | Every epoch     | Bandwidth exploding, loss diverging, IPR collapsing |

use crate::tolerances;
use crate::weight_spectral::WeightSpectralResult;
use barracuda::nautilus::{DriftMonitor, GenerationRecord, InstanceId};

/// Training interrupt signal (adapted from hotSpring `BrainInterrupt`).
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingInterrupt {
    /// Training proceeding normally.
    Continue,
    /// Reduce learning rate by the given factor (Yellow → corrective action).
    ReduceLearningRate {
        /// Multiplicative factor applied to the learning rate (< 1.0).
        factor: f64,
    },
    /// Stop training immediately (Red → irrecoverable state).
    EarlyStop {
        /// Human-readable explanation shown when training halts.
        reason: String,
    },
}

/// Attention state for the training monitor (hotSpring 3-state FSM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionState {
    /// Normal operation. Check every `GREEN_CHECK_INTERVAL` epochs.
    Green,
    /// Elevated monitoring. Check every `YELLOW_CHECK_INTERVAL` epochs.
    Yellow,
    /// Critical alert. Check every epoch.
    Red,
}

/// Epoch-level snapshot for the training monitor.
#[derive(Debug, Clone)]
struct EpochSnapshot {
    loss: f64,
    bandwidth: f64,
    ipr: f64,
}

/// Adaptive training monitor with brain-inspired interrupt logic.
///
/// Combines spectral weight analysis with drift detection to provide
/// real-time training health assessment.
pub struct TrainingMonitor {
    drift: DriftMonitor,
    history: Vec<EpochSnapshot>,
    attention: AttentionState,
}

const GREEN_CHECK_INTERVAL: usize = 10;
const YELLOW_CHECK_INTERVAL: usize = 3;

const BANDWIDTH_GROWTH_THRESHOLD: f64 = tolerances::TRAINING_BANDWIDTH_GROWTH;
const LOSS_STALL_THRESHOLD: f64 = tolerances::TRAINING_LOSS_STALL;
const LOSS_STALL_WINDOW: usize = 5;

const BANDWIDTH_EXPLOSION_THRESHOLD: f64 = tolerances::TRAINING_BANDWIDTH_EXPLOSION;
const IPR_COLLAPSE_THRESHOLD: f64 = tolerances::TRAINING_IPR_COLLAPSE;
const LOSS_DIVERGENCE_THRESHOLD: f64 = tolerances::TRAINING_LOSS_DIVERGENCE;

const LR_REDUCTION_FACTOR: f64 = tolerances::TRAINING_LR_REDUCTION;

impl TrainingMonitor {
    /// Create a new training monitor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            drift: DriftMonitor::default(),
            history: Vec::new(),
            attention: AttentionState::Green,
        }
    }

    /// Record a training epoch observation.
    ///
    /// Feed the current epoch's loss and weight spectral analysis result.
    /// The monitor updates its internal state and transitions the attention
    /// FSM as needed.
    pub fn observe_epoch(&mut self, epoch: usize, loss: f64, spectral: &WeightSpectralResult) {
        let snapshot = EpochSnapshot {
            loss,
            bandwidth: spectral.bandwidth,
            ipr: spectral.mean_ipr,
        };

        self.history.push(snapshot);

        let pop_size = self.history.len();
        let mean_fitness = -loss;
        let best_fitness = self
            .history
            .iter()
            .map(|s| -s.loss)
            .fold(f64::NEG_INFINITY, f64::max);
        let gen_record = GenerationRecord {
            generation: epoch,
            mean_fitness,
            best_fitness,
            pop_size,
            origin: InstanceId("neuralSpring-training-monitor".to_string()),
            training_size: self.history.len(),
        };
        self.drift.record(&gen_record, pop_size);

        self.transition_attention();
    }

    /// Check if a training interrupt should be raised.
    ///
    /// Call this after `observe_epoch()`. Returns `Continue` if training
    /// should proceed, or an interrupt signal if corrective action is needed.
    #[must_use]
    pub fn check_interrupt(&self) -> TrainingInterrupt {
        if self.history.len() < 2 {
            return TrainingInterrupt::Continue;
        }

        match self.attention {
            AttentionState::Red => self.check_red(),
            AttentionState::Yellow => self.check_yellow(),
            AttentionState::Green => TrainingInterrupt::Continue,
        }
    }

    /// Current attention state.
    #[must_use]
    pub const fn attention(&self) -> AttentionState {
        self.attention
    }

    /// Whether the training population is drifting.
    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.drift.is_drifting()
    }

    /// Access the underlying drift monitor.
    #[must_use]
    pub const fn drift_monitor(&self) -> &DriftMonitor {
        &self.drift
    }

    /// Number of epochs observed.
    #[must_use]
    pub const fn epoch_count(&self) -> usize {
        self.history.len()
    }

    /// Whether the current epoch should be checked (respects check interval).
    #[must_use]
    pub const fn should_check(&self, epoch: usize) -> bool {
        let interval = match self.attention {
            AttentionState::Green => GREEN_CHECK_INTERVAL,
            AttentionState::Yellow => YELLOW_CHECK_INTERVAL,
            AttentionState::Red => 1,
        };
        epoch.is_multiple_of(interval)
    }

    fn transition_attention(&mut self) {
        if self.history.len() < 2 {
            return;
        }

        let len = self.history.len();
        let curr = &self.history[len - 1];
        let prev = &self.history[len - 2];

        let bw_ratio = if prev.bandwidth.abs() > tolerances::LOG_ZERO_GUARD {
            curr.bandwidth / prev.bandwidth
        } else {
            1.0
        };

        if bw_ratio > BANDWIDTH_EXPLOSION_THRESHOLD
            || curr.ipr < IPR_COLLAPSE_THRESHOLD
            || (curr.loss > prev.loss * LOSS_DIVERGENCE_THRESHOLD
                && prev.loss > tolerances::LOG_ZERO_GUARD)
        {
            self.attention = AttentionState::Red;
        } else if bw_ratio > BANDWIDTH_GROWTH_THRESHOLD || self.is_loss_stalling() {
            self.attention = match self.attention {
                AttentionState::Green => AttentionState::Yellow,
                other => other,
            };
        } else if self.attention == AttentionState::Yellow
            && bw_ratio < 1.1
            && !self.is_loss_stalling()
        {
            self.attention = AttentionState::Green;
        }
    }

    fn is_loss_stalling(&self) -> bool {
        if self.history.len() < LOSS_STALL_WINDOW {
            return false;
        }
        let recent = self.history[self.history.len() - LOSS_STALL_WINDOW..]
            .iter()
            .map(|s| s.loss);
        let (min_val, max_val) = recent.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
        (max_val - min_val).abs() < LOSS_STALL_THRESHOLD
    }

    fn check_red(&self) -> TrainingInterrupt {
        let len = self.history.len();
        let curr = &self.history[len - 1];
        let prev = &self.history[len - 2];

        if curr.loss > prev.loss * LOSS_DIVERGENCE_THRESHOLD
            && prev.loss > tolerances::LOG_ZERO_GUARD
        {
            return TrainingInterrupt::EarlyStop {
                reason: format!(
                    "loss diverging: {:.6} → {:.6} ({}× increase)",
                    prev.loss,
                    curr.loss,
                    curr.loss / prev.loss
                ),
            };
        }

        if curr.ipr < IPR_COLLAPSE_THRESHOLD {
            return TrainingInterrupt::EarlyStop {
                reason: format!(
                    "IPR collapsed to {:.6} (threshold {IPR_COLLAPSE_THRESHOLD}): \
                     eigenstates fully localized, network memorizing",
                    curr.ipr
                ),
            };
        }

        TrainingInterrupt::ReduceLearningRate {
            factor: LR_REDUCTION_FACTOR,
        }
    }

    fn check_yellow(&self) -> TrainingInterrupt {
        if self.is_loss_stalling() {
            return TrainingInterrupt::ReduceLearningRate {
                factor: LR_REDUCTION_FACTOR,
            };
        }
        TrainingInterrupt::Continue
    }
}

impl Default for TrainingMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TrainingVisualizer: live petalTongue streaming wrapper
// ---------------------------------------------------------------------------

use crate::visualization::ipc_push::PushResult;
use crate::visualization::stream::StreamSession;

/// Live visualization wrapper for [`TrainingMonitor`].
///
/// Bridges the training FSM to a [`StreamSession`], pushing spectral
/// diagnostics to petalTongue on each epoch. Scientists see IPR
/// collapse, bandwidth explosion, and attention state changes in
/// real time.
pub struct TrainingVisualizer {
    session: StreamSession,
}

impl TrainingVisualizer {
    /// Wrap an existing [`StreamSession`] as the visualization sink.
    #[must_use]
    pub const fn new(session: StreamSession) -> Self {
        Self { session }
    }

    /// Push epoch-level spectral metrics to petalTongue.
    ///
    /// Call this immediately after [`TrainingMonitor::observe_epoch`]
    /// with the same spectral result.
    ///
    /// # Errors
    ///
    /// Returns a [`PushError`] if any stream operation fails.
    /// The caller can check [`StreamSession::backpressure_active`]
    /// to decide whether to skip visualization updates.
    ///
    /// [`PushError`]: crate::visualization::ipc_push::PushError
    #[expect(
        clippy::cast_precision_loss,
        reason = "epoch count will never exceed 2^53"
    )]
    pub fn on_epoch(
        &self,
        epoch: usize,
        spectral: &WeightSpectralResult,
        state: AttentionState,
    ) -> PushResult<()> {
        let x = &[epoch as f64];

        self.session
            .append("epoch-vs-ipr", x, &[spectral.mean_ipr])?;
        self.session
            .append("epoch-vs-bandwidth", x, &[spectral.bandwidth])?;
        self.session
            .append("epoch-vs-entropy", x, &[spectral.spectral_entropy])?;
        self.session
            .append("epoch-vs-lsr", x, &[spectral.level_spacing_ratio])?;
        self.session
            .set_gauge("attention-state", state_to_f64(state))?;
        self.session.set_gauge("current-ipr", spectral.mean_ipr)?;
        self.session
            .set_gauge("current-bandwidth", spectral.bandwidth)?;
        self.session
            .set_gauge("condition-number", spectral.condition_number)?;

        Ok(())
    }

    /// Access the underlying session for stats / backpressure checks.
    #[must_use]
    pub const fn session(&self) -> &StreamSession {
        &self.session
    }
}

/// Map attention state to a numeric gauge value (0=Green, 1=Yellow, 2=Red).
const fn state_to_f64(state: AttentionState) -> f64 {
    match state {
        AttentionState::Green => 0.0,
        AttentionState::Yellow => 1.0,
        AttentionState::Red => 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_spectral::{SpectralPhase, WeightSpectralResult};

    fn mock_spectral(bandwidth: f64, ipr: f64, lsr: f64) -> WeightSpectralResult {
        WeightSpectralResult {
            eigenvalues: vec![0.0; 10],
            mean_ipr: ipr,
            level_spacing_ratio: lsr,
            spectral_entropy: 2.0,
            mp_departure: 0.1,
            bandwidth,
            condition_number: 10.0,
            phase: SpectralPhase::Extended,
        }
    }

    #[test]
    fn new_monitor_starts_green() {
        let m = TrainingMonitor::new();
        assert_eq!(m.attention(), AttentionState::Green);
        assert_eq!(m.epoch_count(), 0);
        assert!(!m.is_drifting());
    }

    #[test]
    fn stable_training_stays_green() {
        let mut m = TrainingMonitor::new();
        for i in 0_u32..20 {
            let loss = 1.0 / (f64::from(i) + 1.0);
            m.observe_epoch(i as usize, loss, &mock_spectral(1.0, 0.5, 0.53));
        }
        assert_eq!(m.attention(), AttentionState::Green);
        assert_eq!(m.check_interrupt(), TrainingInterrupt::Continue);
    }

    #[test]
    fn bandwidth_growth_triggers_yellow() {
        let mut m = TrainingMonitor::new();
        m.observe_epoch(0, 1.0, &mock_spectral(1.0, 0.5, 0.53));
        m.observe_epoch(1, 0.9, &mock_spectral(2.5, 0.5, 0.53));
        assert_eq!(m.attention(), AttentionState::Yellow);
    }

    #[test]
    fn bandwidth_explosion_triggers_red() {
        let mut m = TrainingMonitor::new();
        m.observe_epoch(0, 1.0, &mock_spectral(1.0, 0.5, 0.53));
        m.observe_epoch(1, 0.9, &mock_spectral(6.0, 0.5, 0.53));
        assert_eq!(m.attention(), AttentionState::Red);
    }

    #[test]
    fn ipr_collapse_triggers_red_and_early_stop() {
        let mut m = TrainingMonitor::new();
        m.observe_epoch(0, 1.0, &mock_spectral(1.0, 0.5, 0.53));
        m.observe_epoch(1, 0.9, &mock_spectral(1.1, 0.005, 0.53));
        assert_eq!(m.attention(), AttentionState::Red);
        let interrupt = m.check_interrupt();
        assert!(
            matches!(&interrupt, TrainingInterrupt::EarlyStop { reason } if reason.contains("IPR collapsed")),
            "expected EarlyStop(IPR collapsed), got {interrupt:?}"
        );
    }

    #[test]
    fn loss_divergence_triggers_early_stop() {
        let mut m = TrainingMonitor::new();
        m.observe_epoch(0, 1.0, &mock_spectral(1.0, 0.5, 0.53));
        m.observe_epoch(1, 15.0, &mock_spectral(1.0, 0.5, 0.53));
        assert_eq!(m.attention(), AttentionState::Red);
        let interrupt = m.check_interrupt();
        assert!(
            matches!(&interrupt, TrainingInterrupt::EarlyStop { reason } if reason.contains("diverging")),
            "expected EarlyStop(diverging), got {interrupt:?}"
        );
    }

    #[test]
    fn loss_stall_triggers_lr_reduction() {
        let mut m = TrainingMonitor::new();
        for i in 0..10 {
            m.observe_epoch(i, 0.5, &mock_spectral(1.0, 0.5, 0.53));
        }
        m.observe_epoch(10, 0.5, &mock_spectral(2.5, 0.5, 0.53));
        assert_eq!(m.attention(), AttentionState::Yellow);
        assert_eq!(
            m.check_interrupt(),
            TrainingInterrupt::ReduceLearningRate { factor: 0.5 }
        );
    }

    #[test]
    fn check_interval_respects_state() {
        let m_green = TrainingMonitor::new();
        assert!(m_green.should_check(0));
        assert!(!m_green.should_check(1));
        assert!(m_green.should_check(10));
    }

    #[test]
    fn recovery_from_yellow_to_green() {
        let mut m = TrainingMonitor::new();
        m.observe_epoch(0, 1.0, &mock_spectral(1.0, 0.5, 0.53));
        m.observe_epoch(1, 0.9, &mock_spectral(2.5, 0.5, 0.53));
        assert_eq!(m.attention(), AttentionState::Yellow);

        m.observe_epoch(2, 0.8, &mock_spectral(2.6, 0.5, 0.53));
        assert_eq!(m.attention(), AttentionState::Green);
    }
}
