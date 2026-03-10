// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sequence search pipelines — BLAST-like seed-and-extend over barraCuda GPU.
//!
//! Composes barraCuda primitives (`SmithWatermanGpu`, `KmerHistogramGpu`) into
//! multi-stage pipelines that replace external tools (BLAST, HMMER).
//!
//! ## Architecture
//!
//! ```text
//! Phase 1: SEED        Phase 2: EXTEND            Phase 3: SCORE
//! ─────────────────    ─────────────────────────   ────────────────
//! k-mer index of DB    Smith-Waterman on hits      E-value, bit score
//! exact word match     banded + affine gaps        Karlin-Altschul
//! O(n) scan            GPU batch dispatch          per-hit filter
//! ```
//!
//! See `specs/BLAST_LIKE_SEARCH_SCOPE.md` for the full design.

pub mod kmer_index;
pub mod seed_extend;
