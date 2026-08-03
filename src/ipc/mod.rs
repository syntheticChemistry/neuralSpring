// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-primal IPC modules — eukaryotic cell membrane.
//!
//! Graduated from the monolithic `ipc_dispatch` module during the
//! interstadial eukaryotic evolution (May 2026). Each submodule owns
//! the JSON-RPC surface for a single primal:
//!
//! | Module       | Primal       | Capabilities |
//! |--------------|--------------|--------------|
//! | [`barracuda`] | barraCuda   | `stats.*`, `tensor.*`, `barracuda.precision.route`, `ml.mlp_infer` |
//! | [`toadstool`] | toadStool   | `compute.dispatch`, `toadstool.validate`, `toadstool.list_workloads` |
//! | [`beardog`]   | `BearDog`   | `crypto.hash`, `crypto.btsp_handshake` |
//! | [`squirrel`]  | Squirrel    | `inference.*` (incl. `register_provider`, `unregister_provider`) |
//! | [`coralreef`] | coralReef   | `shader.compile.*` |
//! | [`skunkbat`]  | skunkBat    | `security.audit_log` |
//! | [`nestgate`]  | `NestGate`  | `content.put`, `content.get` |
//!
//! ## Discovery Model
//!
//! [`IpcMathClient`] discovers primals via a **hint-then-probe** model:
//!
//! 1. **Hint**: [`router::CAPABILITY_HINTS`] maps each capability to its expected
//!    primal (e.g. `stats.mean` → `barracuda`). The primal name is used
//!    to locate sockets via biomeOS directory scanning — no socket paths
//!    are hardcoded.
//! 2. **Probe**: Once a socket is found, the primal binary's async
//!    discovery layer (`neuralspring_primal/discovery.rs`) can verify
//!    the primal actually advertises the capability via `capability.list`.
//!
//! This follows the ecoPrimals self-knowledge principle: a spring only
//! knows *what* it needs (a capability), not *where* to find it. The
//! hint table is a compile-time optimization for fast startup; runtime
//! capability probing via [`crate::validation::composition::probe_capabilities`]
//! provides full dynamic verification.

pub mod barracuda;
pub mod beardog;
pub mod client;
pub mod coralreef;
pub mod health;
pub mod nestgate;
pub mod router;
pub mod skunkbat;
pub mod squirrel;
pub mod toadstool;

use std::time::Duration;

pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

pub use client::IpcMathClient;
pub use health::{IpcLivenessReport, PrimalSlot};
pub use router::CapabilityRouter;

pub(crate) use router::{extract_f64, extract_f64_array};
