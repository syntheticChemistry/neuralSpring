// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable IPC resilience primitives (groundSpring V113 / airSpring V0.8.8 pattern).
//!
//! Provides [`RetryPolicy`] (configurable exponential backoff) and
//! [`CircuitBreaker`] (Closed/Open/HalfOpen with configurable threshold
//! and cooldown).  Both are transport-agnostic: they wrap any fallible
//! operation, not just Unix socket calls.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

/// Configurable exponential-backoff retry policy.
///
/// Delays grow as `initial_delay * multiplier^attempt`, capped at
/// `max_delay`.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Delay before the first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Backoff multiplier applied per attempt.
    pub multiplier: f64,
}

impl RetryPolicy {
    /// Compute the delay for a given attempt index (0-based).
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.initial_delay.as_secs_f64() * self.multiplier.powf(f64::from(attempt));
        let capped = base.min(self.max_delay.as_secs_f64());
        Duration::from_secs_f64(capped)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(2),
            multiplier: 2.0,
        }
    }
}

/// Three-state circuit breaker preventing cascading failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — requests pass through.
    Closed,
    /// Failures exceeded threshold — requests are short-circuited.
    Open,
    /// Cooldown elapsed — one probe request is allowed.
    HalfOpen,
}

/// Thread-safe circuit breaker with configurable failure threshold and
/// cooldown.
///
/// State transitions:
/// - `Closed` → `Open`: after `threshold` consecutive failures.
/// - `Open` → `HalfOpen`: after `cooldown` elapses.
/// - `HalfOpen` → `Closed`: on success.
/// - `HalfOpen` → `Open`: on failure.
pub struct CircuitBreaker {
    threshold: u32,
    cooldown: Duration,
    consecutive_failures: AtomicU32,
    last_failure_epoch_ms: AtomicU64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// - `threshold`: number of consecutive failures to open the circuit.
    /// - `cooldown`: duration to wait before transitioning to `HalfOpen`.
    #[must_use]
    pub const fn new(threshold: u32, cooldown: Duration) -> Self {
        Self {
            threshold,
            cooldown,
            consecutive_failures: AtomicU32::new(0),
            last_failure_epoch_ms: AtomicU64::new(0),
        }
    }

    /// Current state of the breaker.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        if failures < self.threshold {
            return CircuitState::Closed;
        }
        let last = self.last_failure_epoch_ms.load(Ordering::Relaxed);
        if last == 0 {
            return CircuitState::Closed;
        }
        let now = epoch_ms_now();
        let cooldown_ms = u64::try_from(self.cooldown.as_millis()).unwrap_or(u64::MAX);
        if now.saturating_sub(last) >= cooldown_ms {
            CircuitState::HalfOpen
        } else {
            CircuitState::Open
        }
    }

    /// Whether a request should be attempted (`Closed` or `HalfOpen`).
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.state() != CircuitState::Open
    }

    /// Record a successful call — resets the failure counter.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.last_failure_epoch_ms.store(0, Ordering::Relaxed);
    }

    /// Record a failed call — increments the failure counter.
    pub fn record_failure(&self) {
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        self.last_failure_epoch_ms
            .store(epoch_ms_now(), Ordering::Relaxed);
    }
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("state", &self.state())
            .field("threshold", &self.threshold)
            .field("cooldown", &self.cooldown)
            .field(
                "consecutive_failures",
                &self.consecutive_failures.load(Ordering::Relaxed),
            )
            .field(
                "last_failure_epoch_ms",
                &self.last_failure_epoch_ms.load(Ordering::Relaxed),
            )
            .finish()
    }
}

fn epoch_ms_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "epoch millis fits u64 for ~584 million years"
            )]
            let ms = d.as_millis() as u64;
            ms
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_policy_default() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 2);
        assert_eq!(p.initial_delay, Duration::from_millis(50));
    }

    #[test]
    fn retry_delay_exponential_backoff() {
        let p = RetryPolicy {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        };
        assert_eq!(p.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(p.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(p.delay_for_attempt(2), Duration::from_millis(400));
    }

    #[test]
    fn retry_delay_capped_at_max() {
        let p = RetryPolicy {
            max_retries: 10,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            multiplier: 10.0,
        };
        assert_eq!(p.delay_for_attempt(3), Duration::from_secs(5));
    }

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_allowed());
    }

    #[test]
    fn circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.is_allowed());
    }

    #[test]
    fn circuit_breaker_resets_on_success() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_allowed());
    }

    #[test]
    fn circuit_breaker_half_open_after_cooldown() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(0));
        cb.record_failure();
        // cooldown is 0ms, so should immediately transition to HalfOpen
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.is_allowed());
    }

    #[test]
    fn circuit_breaker_debug_shows_state() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        let debug = format!("{cb:?}");
        assert!(debug.contains("Closed"));
    }
}
