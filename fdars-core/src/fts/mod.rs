//! Functional time series serial-dependence diagnostics.
//!
//! # R baselines
//!
//! * [`functional_acf`] / [`functional_pacf`] — `fdaACF::facf`
//!   (Mestre et al. 2021, *Computational Statistics & Data Analysis*).
//! * [`stationarity_test`] — `ftsa::T_stationary`
//!   (Horváth, Kokoszka, Rice 2014, *Journal of Econometrics* 179:66–82).
//! * [`long_run_covariance`] — `ftsa::long_run_covariance_estimation`
//!   (Bartlett HAC kernel-sandwich estimator).
//! * [`functional_difference`] — `ftsa::diff.fts` (functional first-difference).
//!
//! # Conventions
//!
//! Entry points take an explicit deterministic `seed` (`StdRng::seed_from_u64(seed)`)
//! and default Monte-Carlo replications of 999. All public functions return
//! `Result<_, FdarError>` and validate inputs at entry. Result structs derive
//! `Debug, Clone, PartialEq` and are serde-gated.

mod acf;

pub use acf::{functional_acf, functional_pacf};

/// Result of functional ACF/PACF estimation.
///
/// Produced by [`functional_acf`] and [`functional_pacf`].
/// Lag values run from 1 to `max_lag`; `acf`, `pacf`, and `upper_band`
/// all have length `max_lag`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FacfResult {
    /// Lag values (1..=max_lag).
    pub lags: Vec<u32>,
    /// Functional autocorrelation ρ_h at each lag (L2-norm, fdaACF convention).
    pub acf: Vec<f64>,
    /// Functional partial autocorrelation (scalar Durbin-Levinson approximation).
    pub pacf: Vec<f64>,
    /// Upper confidence band under the strong-white-noise null (Monte-Carlo quantile).
    pub upper_band: Vec<f64>,
}

/// Result of the functional stationarity test.
///
/// Produced by `stationarity_test` (plan 34-02).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct StationarityResult {
    /// Test statistic T (KPSS-style partial-sum L2 norm).
    pub statistic: f64,
    /// Monte-Carlo permutation p-value.
    pub p_value: f64,
    /// Number of permutations used.
    pub n_perm: usize,
}

/// Result of the Bartlett kernel-sandwich long-run covariance estimator.
///
/// Produced by `long_run_covariance` (plan 34-02).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LongRunCovResult {
    /// Estimated m×m long-run covariance matrix (column-major flat Vec).
    pub cov_matrix: Vec<f64>,
    /// Grid dimension m (cov_matrix is m×m).
    pub m: usize,
    /// Bandwidth used.
    pub bandwidth: usize,
    /// Number of curves N.
    pub n_curves: usize,
}
