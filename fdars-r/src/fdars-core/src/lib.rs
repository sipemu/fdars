//! # fdars-core
//!
//! Core algorithms for Functional Data Analysis in Rust.
//!
//! This crate provides pure Rust implementations of various FDA methods including:
//! - Functional data operations (mean, derivatives, norms)
//! - Depth measures (Fraiman-Muniz, modal, band, random projection, etc.)
//! - Distance metrics (Lp, Hausdorff, DTW, Fourier, etc.)
//! - Basis representations (B-splines, P-splines, Fourier)
//! - Clustering (k-means, fuzzy c-means)
//! - Smoothing (Nadaraya-Watson, local linear/polynomial regression)
//! - Outlier detection
//! - Regression (PCA, PLS, ridge)
//! - Seasonal analysis (period estimation, peak detection, seasonal strength)
//! - Detrending and decomposition for non-stationary data
//!
//! ## Data Layout
//!
//! Functional data is represented as column-major matrices stored in flat vectors:
//! - For n observations with m evaluation points: `data[i + j * n]` gives observation i at point j
//! - 2D surfaces (n observations, m1 x m2 grid): stored as n x (m1*m2) matrices

#![allow(clippy::needless_range_loop)]

pub mod basis;
pub mod clustering;
pub mod depth;
pub mod detrend;
pub mod fdata;
pub mod helpers;
pub mod metric;
pub mod outliers;
pub mod regression;
pub mod seasonal;
pub mod smoothing;
pub mod utility;

// Re-export commonly used items
pub use helpers::{simpsons_weights, simpsons_weights_2d};

// Re-export seasonal analysis types
pub use seasonal::{
    ChangeDetectionResult, ChangePoint, ChangeType, DetectedPeriod, InstantaneousPeriod, Peak,
    PeakDetectionResult, PeriodEstimate, StrengthMethod,
};

// Re-export detrending types
pub use detrend::{DecomposeResult, TrendResult};
