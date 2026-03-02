// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated bio operations: HMM forward/backward/Viterbi,
//! pairwise distance, Hill activation.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

mod activation;
mod evolution;
mod hmm;

pub use activation::*;
pub use evolution::*;
pub use hmm::*;
