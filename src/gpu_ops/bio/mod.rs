// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated bio operations: HMM forward/backward/Viterbi,
//! pairwise distance, Hill activation.

#![expect(
    clippy::cast_possible_truncation,
    reason = "domain-specific numeric patterns"
)]

mod activation;
mod evolution;
mod hmm;

pub use activation::*;
pub use evolution::*;
pub use hmm::*;
