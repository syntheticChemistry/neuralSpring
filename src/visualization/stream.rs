// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::cast_precision_loss,
    reason = "session counters won't realistically exceed 2^53"
)]

//! Streaming session for live petalTongue visualization.
//!
//! Wraps [`PetalTonguePushClient`] with session lifecycle, backpressure
//! tracking, and cumulative statistics. Follows the `StreamSession`
//! pattern from healthSpring.
//!
//! ## Usage
//!
//! ```text
//! let session = StreamSession::start(client, "live-spectral", scenario)?;
//! session.append("ipr_series", &[1.0], &[0.25])?;
//! session.set_gauge("entropy_gauge", 2.47)?;
//! session.replace("heatmap_1", &json!({...}))?;
//! let stats = session.stats();
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::ipc_push::{PetalTonguePushClient, PushResult};
use super::types::NeuralScenario;

/// Live streaming session for petalTongue visualization.
///
/// Tracks message count, byte volume, and error rate for backpressure
/// awareness. The session renders an initial scenario on start and
/// then streams updates via `append`/`set_value`/`replace` operations.
pub struct StreamSession {
    client: PetalTonguePushClient,
    session_id: String,
    started_at: Instant,
    messages_sent: AtomicU64,
    bytes_sent: AtomicU64,
    errors: AtomicU64,
}

/// Session statistics snapshot.
#[derive(Debug, Clone, Copy)]
pub struct SessionStats {
    /// Number of messages sent (including initial render).
    pub messages_sent: u64,
    /// Approximate total bytes pushed.
    pub bytes_sent: u64,
    /// Number of failed push operations.
    pub errors: u64,
    /// Session uptime in milliseconds.
    pub uptime_ms: u64,
}

impl SessionStats {
    /// Messages per second throughput.
    #[must_use]
    pub fn messages_per_second(&self) -> f64 {
        if self.uptime_ms == 0 {
            return 0.0;
        }
        (self.messages_sent as f64) / (self.uptime_ms as f64 / 1000.0)
    }

    /// Error rate as a fraction [0.0, 1.0].
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.messages_sent + self.errors;
        if total == 0 {
            return 0.0;
        }
        self.errors as f64 / total as f64
    }
}

impl StreamSession {
    /// Start a new streaming session by rendering the initial scenario.
    ///
    /// # Errors
    ///
    /// Returns a [`PushError`] if the initial render fails.
    ///
    /// [`PushError`]: super::ipc_push::PushError
    pub fn start(
        client: PetalTonguePushClient,
        session_id: &str,
        title: &str,
        scenario: &NeuralScenario,
    ) -> PushResult<Self> {
        client.push_render(session_id, title, scenario)?;

        Ok(Self {
            client,
            session_id: session_id.to_string(),
            started_at: Instant::now(),
            messages_sent: AtomicU64::new(1),
            bytes_sent: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        })
    }

    /// Create a session without sending an initial render (for resumed sessions).
    #[must_use]
    pub fn resume(client: PetalTonguePushClient, session_id: &str) -> Self {
        Self {
            client,
            session_id: session_id.to_string(),
            started_at: Instant::now(),
            messages_sent: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }

    /// Append data points to a `TimeSeries` binding.
    ///
    /// # Errors
    ///
    /// Returns a [`PushError`] if the stream update fails.
    ///
    /// [`PushError`]: super::ipc_push::PushError
    pub fn append(&self, binding_id: &str, x_values: &[f64], y_values: &[f64]) -> PushResult<()> {
        match self
            .client
            .push_append(&self.session_id, binding_id, x_values, y_values)
        {
            Ok(()) => {
                self.messages_sent.fetch_add(1, Ordering::Relaxed);
                let approx_bytes = (x_values.len() + y_values.len()) * 8;
                self.bytes_sent
                    .fetch_add(approx_bytes as u64, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Update a gauge binding value.
    ///
    /// # Errors
    ///
    /// Returns a [`PushError`] if the gauge update fails.
    ///
    /// [`PushError`]: super::ipc_push::PushError
    pub fn set_gauge(&self, binding_id: &str, value: f64) -> PushResult<()> {
        match self
            .client
            .push_gauge_update(&self.session_id, binding_id, value)
        {
            Ok(()) => {
                self.messages_sent.fetch_add(1, Ordering::Relaxed);
                self.bytes_sent.fetch_add(8, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Replace the entire data payload for a binding.
    ///
    /// # Errors
    ///
    /// Returns a [`PushError`] if the replace operation fails.
    ///
    /// [`PushError`]: super::ipc_push::PushError
    pub fn replace(&self, binding_id: &str, data: &serde_json::Value) -> PushResult<()> {
        match self.client.push_replace(&self.session_id, binding_id, data) {
            Ok(()) => {
                self.messages_sent.fetch_add(1, Ordering::Relaxed);
                let approx_bytes = serde_json::to_vec(data).map_or(0, |v| v.len());
                self.bytes_sent
                    .fetch_add(approx_bytes as u64, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Snapshot current session statistics.
    #[must_use]
    pub fn stats(&self) -> SessionStats {
        SessionStats {
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            uptime_ms: u64::try_from(self.started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
        }
    }

    /// The session ID.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Whether the error rate exceeds a backpressure threshold.
    ///
    /// If more than 10% of messages have failed, the caller should
    /// slow down or stop sending.
    #[must_use]
    pub fn backpressure_active(&self) -> bool {
        self.stats().error_rate() > 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_stats_zero_uptime() {
        let stats = SessionStats {
            messages_sent: 0,
            bytes_sent: 0,
            errors: 0,
            uptime_ms: 0,
        };
        assert!(stats.messages_per_second().abs() < f64::EPSILON);
        assert!(stats.error_rate().abs() < f64::EPSILON);
    }

    #[test]
    fn session_stats_throughput() {
        let stats = SessionStats {
            messages_sent: 100,
            bytes_sent: 8000,
            errors: 0,
            uptime_ms: 1000,
        };
        assert!((stats.messages_per_second() - 100.0).abs() < f64::EPSILON);
        assert!(stats.error_rate().abs() < f64::EPSILON);
    }

    #[test]
    fn session_stats_error_rate() {
        let stats = SessionStats {
            messages_sent: 90,
            bytes_sent: 0,
            errors: 10,
            uptime_ms: 1000,
        };
        assert!((stats.error_rate() - 0.1).abs() < crate::tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn resume_starts_with_zero_messages() {
        let client =
            PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent_ns_stream.sock"));
        let session = StreamSession::resume(client, "test-session");
        let stats = session.stats();
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(session.session_id(), "test-session");
    }

    #[test]
    fn backpressure_inactive_at_zero() {
        let client =
            PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent_ns_stream.sock"));
        let session = StreamSession::resume(client, "test-bp");
        assert!(!session.backpressure_active());
    }
}
