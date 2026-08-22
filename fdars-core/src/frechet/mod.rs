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
mod space;

pub use mean::{frechet_mean, frechet_variance};
pub use space::{wasserstein2_distance, MetricSpace, WassersteinDensitySpace};
