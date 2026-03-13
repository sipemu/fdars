//! Basis representation functions for functional data.
//!
//! This module provides B-spline and Fourier basis expansions for representing
//! functional data in a finite-dimensional basis.

pub mod auto_select;
pub mod bspline;
pub mod fourier;
pub mod fourier_fit;
mod helpers;
pub mod projection;
pub mod pspline;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Re-exports — preserves the external API
// ---------------------------------------------------------------------------

pub use auto_select::{select_basis_auto_1d, BasisAutoSelectionResult, SingleCurveSelection};
pub use bspline::bspline_basis;
pub use fourier::{fourier_basis, fourier_basis_with_period};
pub use fourier_fit::{fourier_fit_1d, select_fourier_nbasis_gcv, FourierFitResult};
pub use projection::{
    basis_to_fdata, basis_to_fdata_1d, fdata_to_basis, fdata_to_basis_1d, BasisProjectionResult,
    ProjectionBasisType,
};
pub use pspline::{difference_matrix, pspline_fit_1d, PsplineFitResult};
