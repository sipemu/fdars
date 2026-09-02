//! Shapelet transform & classification.
//!
//! Discovery-based shapelets (Ye & Keogh 2009; Hills–Lines 2014): discriminative
//! subsequences whose z-normalized distance to a curve becomes a feature for
//! downstream classification.
//!
//! This module is built up over four phases along a strict dependency chain:
//!
//! 1. **distance** (this phase) — per-window z-normalization and the
//!    sliding-window minimum z-normalized Euclidean distance (`sdist`), plus the
//!    [`Shapelet`] type. The atomic primitive every later step consumes.
//! 2. *discovery* — candidate generation, quality scoring, top-K selection.
//! 3. *transform* — apply a fitted shapelet set to produce an n×K feature matrix.
//! 4. *classifier* — the bundled end-to-end shapelet-transform classifier.
//!
//! Crate-root flat re-exports are intentionally deferred; reach the public items
//! via `fdars_core::shapelet::...` for now.

pub mod distance;

pub use distance::{shapelet_distance, z_normalize_into, z_normalize_window, Shapelet};
