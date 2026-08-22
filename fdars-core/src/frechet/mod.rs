//! Fréchet / object-data regression and statistics (FRE-01).
//!
//! Metric-space (object-data) regression and statistics in the style of the R
//! `frechet` package: a [`MetricSpace`] abstraction (distance + weighted-Fréchet-
//! mean solver) with a 1D-Wasserstein (density-response) backend
//! ([`WassersteinDensitySpace`]) as the first concrete space, the sample Fréchet
//! [`frechet_mean`] / [`frechet_variance`], the 1D 2-Wasserstein distance
//! ([`wasserstein2_distance`]), global and local (kernel-weighted) Fréchet
//! regression over Euclidean predictors, density-response regression, and a
//! Fréchet ANOVA group-difference test.
//!
//! # R baselines
//!
//! * Global / local Fréchet regression — `frechet::GloWassReg` / `LocWassReg`
//!   (Petersen & Müller 2019, *Annals of Statistics* 47(2)).
//! * Fréchet ANOVA — `frechet::DenANOVA` (Dubey & Müller 2019, *Biometrika* 106(4)).
//!
//! # Reuse & conventions
//!
//! The density backend reuses DENS-01's quantile/Wasserstein machinery
//! ([`crate::density_fda`]) rather than re-deriving it. All public functions
//! return `Result<_, FdarError>` and validate inputs at entry (never panic).
//! Any permutation path uses per-thread seeded RNG
//! (`StdRng::seed_from_u64(seed + k)`) with a default of 999 replications. Result
//! structs derive `Debug, Clone, PartialEq` and are serde-gated.
//!
//! # Divergence
//!
//! Global/local Fréchet regression weights can be negative; where R uses an
//! `osqp` quadratic program to enforce a monotone predicted quantile, this crate
//! uses a zero-dependency sort-based isotonic projection (see the regression
//! submodule).

mod mean;
mod regression;
mod space;

pub use mean::{frechet_mean, frechet_variance};
pub use regression::{frechet_global_reg, frechet_local_reg};
pub use space::{wasserstein2_distance, MetricSpace, WassersteinDensitySpace};

use crate::matrix::FdMatrix;

/// Result of global Fréchet regression ([`frechet_global_reg`]).
///
/// Predicts a conditional Fréchet-mean density response at each `xout` row via the
/// Petersen–Müller global linear weight scheme.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FrechetGlobalRegResult {
    /// Predicted density responses, shape n_out × m (row i = prediction at `xout` row i).
    pub predicted: FdMatrix,
    /// The predictor values predictions were made at, shape n_out × p.
    pub xout: FdMatrix,
    /// Column means of the training predictors, length p.
    pub x_bar: Vec<f64>,
}

/// Result of local (local-linear, kernel-weighted) Fréchet regression
/// ([`frechet_local_reg`]).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FrechetLocalRegResult {
    /// Predicted density responses, shape n_out × m.
    pub predicted: FdMatrix,
    /// The predictor values predictions were made at, shape n_out × p.
    pub xout: FdMatrix,
    /// The kernel bandwidth used.
    pub bandwidth: f64,
}
