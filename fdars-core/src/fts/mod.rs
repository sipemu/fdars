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
mod forecast;
mod spectral;

pub use acf::{
    functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test,
};
pub use forecast::{fplsr, ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update};
pub use spectral::{dpca, dpca_reconstruct, spectral_density};

use crate::matrix::FdMatrix;

/// Result of the spectral density operator estimator.
///
/// Produced by [`spectral_density`] (plan 41-01, FTS-03-01). Holds the complex
/// m×m spectral density operator at each of the `n_curves` Fourier frequencies
/// `θ_j = 2πj/N`. The operator at frequency `k` is stored as separate real
/// (`re[k]`) and imaginary (`im[k]`) flat column-major m×m matrices — element
/// `(j1, j2)` lives at index `j1 + j2 * m`. The operator is Hermitian:
/// `im[k][j1 + j2*m] == -im[k][j2 + j1*m]`.
///
/// # Divergence from `freqdom`
///
/// The `1/2π` pre-factor is omitted (matching the crate's `long_run_covariance`,
/// which does not divide by `2π`), so eigenvalues are `2π` larger than `freqdom`
/// output. Filter shapes (eigenvectors) are unaffected.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SpectralDensityResult {
    /// Fourier frequencies `θ_j = 2πj/N`, length `n_curves`.
    pub freqs: Vec<f64>,
    /// Real part of the operator at each frequency; `re[k]` is a flat column-major m×m matrix.
    pub re: Vec<Vec<f64>>,
    /// Imaginary part of the operator at each frequency; `im[k]` is a flat column-major m×m matrix.
    pub im: Vec<Vec<f64>>,
    /// Grid dimension m (each `re[k]`/`im[k]` is m×m).
    pub m: usize,
    /// Number of curves N (= number of Fourier frequencies).
    pub n_curves: usize,
    /// Bartlett lag-window bandwidth used.
    pub bandwidth: usize,
}

/// Result of dynamic functional PCA (DPCA).
///
/// Produced by [`dpca`] (plan 41-01, FTS-03-02). Holds the time-domain dynamic
/// eigen-filters and the dynamic score series.
///
/// # Divergence from `freqdom`
///
/// Eigenvectors are computed from `Re(f̂(θ))` via [`nalgebra::SymmetricEigen`]
/// (nalgebra 0.33 has no stable complex Hermitian path without `faer`); this is
/// exact for the leading dynamic subspace of a Bartlett-windowed estimator.
/// Filters and scores use the Simpson-weighted L2 inner product consistently,
/// so the estimator/score/reconstruction metric matches.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DpcaResult {
    /// Dynamic eigen-filters, one per component. `filters[c]` is a `(2L+1) × m`
    /// [`FdMatrix`]: row `l_idx` holds the filter tap at lag `l_idx - L`, column `j`
    /// is the grid point.
    pub filters: Vec<FdMatrix>,
    /// Dynamic scores, shape `(N - 2L) × ncomp` (interior time points only).
    pub scores: FdMatrix,
    /// Per-component eigenvalue trajectory across frequencies: `eigenvalues[c]`
    /// has length `n_freqs` (negative finite-sample eigenvalues clipped to 0).
    pub eigenvalues: Vec<Vec<f64>>,
    /// Number of Fourier frequencies N.
    pub n_freqs: usize,
    /// Filter lag support L (window is `[-L, L]`).
    pub filter_lag: usize,
    /// Number of retained dynamic components.
    pub ncomp: usize,
    /// Inclusive interior time range `(L, N-1-L)` for which scores are defined.
    pub valid_range: (usize, usize),
}

/// Result of DPCA curve reconstruction from dynamic scores.
///
/// Produced by [`dpca_reconstruct`] (plan 41-01, FTS-03-03). The
/// `reconstruction_error` is monotone non-increasing in the number of retained
/// components (integrated-L2 error over the fully-defined interior).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DpcaReconstruction {
    /// Reconstructed curves over the interior, shape `(N - 2L) × m`.
    pub fitted: FdMatrix,
    /// Integrated-L2 reconstruction error using K = 1..=ncomp components
    /// (`reconstruction_error[K-1]`); monotone non-increasing in K.
    pub reconstruction_error: Vec<f64>,
    /// Inclusive interior time range `(L, N-1-L)` matching the source [`DpcaResult`].
    pub valid_range: (usize, usize),
}

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

/// Diagnostics for a single fitted FPC-score AR(p) model.
///
/// One per retained component in [`FtsmResult::ar_models`]. Produced by [`ftsm`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ArModelResult {
    /// Selected AR order p (0 = white noise), chosen by AIC.
    pub order: usize,
    /// AR coefficients phi_1..phi_p (0-indexed: `phi[0]` is the lag-1 coefficient).
    pub phi: Vec<f64>,
    /// Innovation (residual) variance from the Yule-Walker fit.
    pub sigma2: f64,
}

/// Result of fitting the FPCA-based functional time-series model.
///
/// Produced by [`ftsm`]. Carries the FPCA decomposition (mean, loadings,
/// score-time-series, reconstructed fitted curves, integration weights) plus the
/// per-component AR-model diagnostics used for forecasting.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FtsmResult {
    /// Mean curve μ(u), length m.
    pub mean: Vec<f64>,
    /// FPC loadings φ_k, shape m × ncomp.
    pub rotation: crate::matrix::FdMatrix,
    /// FPC score time-series β_{t,k}, shape n × ncomp.
    pub scores: crate::matrix::FdMatrix,
    /// Reconstructed fitted curves, shape n × m.
    pub fitted: crate::matrix::FdMatrix,
    /// Simpson integration weights, length m.
    pub weights: Vec<f64>,
    /// Effective number of retained components (clamped to min(ncomp, n, m)).
    pub ncomp: usize,
    /// Per-component fitted AR-model diagnostics (length = ncomp).
    pub ar_models: Vec<ArModelResult>,
}

/// Result of an FPC-score-AR curve forecast.
///
/// Produced by [`ftsm_forecast`]. `forecast` is an h × m matrix whose row `i`
/// holds the forecast curve for horizon `i + 1`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FtsmForecastResult {
    /// Forecast curves, shape h × m (row i = horizon i+1).
    pub forecast: crate::matrix::FdMatrix,
    /// Forecast horizon (number of steps ahead).
    pub h: usize,
}

/// Result of the functional PLS forecasting variant.
///
/// Produced by [`fplsr`]. A lag-1 PLS design (predictor = current curve,
/// response = next curve) yields a one-step-ahead forecast curve plus the
/// in-sample lag-1 fitted curves.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FplsrResult {
    /// One-step-ahead forecast of the next curve, shape 1 × m.
    pub forecast: crate::matrix::FdMatrix,
    /// In-sample lag-1 fitted curves, shape (n-1) × m.
    pub fitted: crate::matrix::FdMatrix,
    /// Number of PLS components used (clamped to min(ncomp, n-1, m)).
    pub ncomp: usize,
}
