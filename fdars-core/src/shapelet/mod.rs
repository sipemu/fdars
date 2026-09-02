//! Shapelet transform & classification.
//!
//! Discovery-based shapelets (Ye & Keogh 2009; Hills–Lines 2014): discriminative
//! subsequences whose z-normalized distance to a curve becomes a feature for
//! downstream classification.
//!
//! This module is built up over four phases along a strict dependency chain:
//!
//! 1. **distance** — per-window z-normalization and the sliding-window minimum
//!    z-normalized Euclidean distance (`sdist`), plus the [`Shapelet`] type. The
//!    atomic primitive every later step consumes.
//! 2. **discovery** — candidate generation, quality scoring (information gain /
//!    F-statistic), and top-K selection with self-similarity pruning
//!    ([`discover_shapelets`], [`ShapeletDiscoveryConfig`], [`ShapeletSet`]).
//! 3. **transform** — apply a fitted shapelet set to produce an n×K feature
//!    matrix, for training and out-of-sample curves ([`shapelet_transform`],
//!    [`shapelet_transform_fit`], [`ShapeletTransformFit`]).
//! 4. **classifier** — the bundled end-to-end shapelet-transform classifier
//!    ([`shapelet_classifier_fit`], [`ShapeletClassifier`],
//!    [`ShapeletClassifierConfig`], [`ShapeletClassifierFit`]).
//!
//! The full public surface is re-exported at the crate root (finalized in the
//! classifier phase); the flat items are also reachable via
//! `fdars_core::shapelet::...`.

pub mod classifier;
pub mod discovery;
pub mod distance;
pub mod transform;

pub use classifier::{
    shapelet_classifier_fit, ShapeletClassifier, ShapeletClassifierConfig, ShapeletClassifierFit,
};
pub use discovery::{discover_shapelets, QualityMeasure, ShapeletDiscoveryConfig, ShapeletSet};
pub use distance::{shapelet_distance, z_normalize_into, z_normalize_window, Shapelet};
pub use transform::{shapelet_transform, shapelet_transform_fit, ShapeletTransformFit};
