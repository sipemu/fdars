//! Tolerance bands for functional data.
//!
//! This module provides methods for constructing regions expected to contain
//! a given fraction of individual curves in a population — the functional
//! analogue of classical tolerance intervals.
//!
//! # Methods
//!
//! - [`fpca_tolerance_band`] — FPCA + bootstrap tolerance band (pointwise or simultaneous)
//! - [`elastic_tolerance_band`] — Amplitude-only band after elastic alignment
//! - [`phase_tolerance_band`] — Phase band on warping functions via tangent-space FPCA
//! - [`elastic_tolerance_band_joint`] — Joint amplitude + phase bands (single alignment)
//! - [`conformal_prediction_band`] — Distribution-free conformal prediction band
//! - [`scb_mean_degras`] — Simultaneous confidence band for the mean (Degras method)
//! - [`exponential_family_tolerance_band`] — Tolerance band for exponential family data

mod conformal;
mod degras;
mod elastic;
mod equivalence;
mod exponential;
pub(crate) mod fpca;
pub(crate) mod helpers;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public items so lib.rs doesn't change
pub use conformal::conformal_prediction_band;
pub use degras::scb_mean_degras;
pub use elastic::{elastic_tolerance_band, elastic_tolerance_band_joint, phase_tolerance_band};
pub use equivalence::{equivalence_test, equivalence_test_one_sample};
pub use exponential::exponential_family_tolerance_band;
pub use fpca::fpca_tolerance_band;
pub use types::{
    BandType, ElasticToleranceBandResult, ElasticToleranceConfig, EquivalenceBootstrap,
    EquivalenceTestResult, ExponentialFamily, MultiplierDistribution, NonConformityScore,
    PhaseToleranceBand, ToleranceBand,
};
