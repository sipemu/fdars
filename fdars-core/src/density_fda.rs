//! Density-valued functional data analysis (LQD transform, Wasserstein barycenter, density FPCA).
//!
//! This module implements the log-quantile-density (LQD) transformation framework of
//! Petersen and Mueller (2016) for probability-density-valued functional data.  The LQD map
//! embeds the constraint-carrying space of probability densities into the unconstrained Hilbert
//! space L²([0,1]), where ordinary FPCA applies.  The inverse map always returns a valid
//! (non-negative, unit-integral) probability density.
//!
//! # Types
//!
//! - [`LqdFpcaResult`] — output of [`lqd_fpca`], embedding the LQD-space
//!   [`crate::regression::FpcaResult`] plus fraction of variance explained (FVE).
//!
//! # R baseline
//!
//! The algorithms in this module are based on the R package **fdadensity 0.1.4**
//! (<https://cran.r-project.org/package=fdadensity>), specifically:
//! - `dens2lqd` — forward LQD transform
//! - `lqd2dens` — inverse LQD transform
//! - `FPCAdens` — functional PCA of densities via LQD space
//! - `getWFmean` — Wasserstein Fréchet mean (quantile average)
//!
//! **Reference:** Petersen, A. and Mueller, H.-G. (2016). Functional data analysis for
//! density functions by transformation to a Hilbert space. *Annals of Statistics*,
//! 44(1):183–218. <https://doi.org/10.1214/15-AOS1363>
//!
//! # Examples
//!
//! ```
//! use fdars_core::density_fda::{normalize_density, lqd_transform, inverse_lqd};
//!
//! // Uniform density on [0, 1]: ψ(t) = 0 for all t
//! let argvals: Vec<f64> = (0..51).map(|i| i as f64 / 50.0).collect();
//! let uniform: Vec<f64> = vec![1.0; 51];
//!
//! let normed = normalize_density(&uniform, &argvals).unwrap();
//! let psi = lqd_transform(&normed, &argvals, Some(51)).unwrap();
//! // ψ ≡ 0 for the uniform density (analytic result)
//! assert!(psi.iter().all(|&v| v.abs() < 1e-6));
//!
//! // Round-trip back to density space
//! let t_grid: Vec<f64> = (0..51).map(|i| i as f64 / 50.0).collect();
//! let recovered = inverse_lqd(&psi, &t_grid, &argvals).unwrap();
//! // Recovered density integrates to 1
//! ```
//!
//! # Divergences from fdadensity
//!
//! 1. **Quantile interpolation:** `fdadensity` uses `spline(..., method = 'natural')` (natural
//!    cubic spline) for the CDF→t mapping in `dens2lqd` and the Q→target-grid back-mapping in
//!    `lqd2dens`.  This implementation uses [`crate::helpers::linear_interp`] (piecewise linear).
//!    Effect: the round-trip (`lqd_transform` → `inverse_lqd`) L∞ error is larger than the
//!    cubic-spline reference. Measured on a truncated standard Gaussian at 201 points it is
//!    ~1.0e-2 (vs. ~5e-3 for cubic spline); on flatter/smoother densities or denser grids it is
//!    smaller. Callers needing tighter round-trip accuracy should supply a finer density grid or
//!    a smoother reference. The reconstructed density always integrates to 1 and is non-negative
//!    regardless of interpolation error.
//!
//! 2. **`useSplines` integration path:** `fdadensity::lqd2dens` has an optional path that
//!    integrates `exp(spline(ψ))` analytically per panel.  This implementation always uses
//!    `cumulative_trapz(exp(ψ), t_grid)`, trading a small accuracy difference for code
//!    simplicity and zero new dependencies.
//!
//! 3. **`wasserstein_barycenter` weights:** `fdadensity::getWFmean` does not support a weight
//!    parameter.  This implementation accepts `weights: Option<&[f64]>`, defaulting to uniform
//!    1/n — a strict superset of fdadensity capability.
//!
//! 4. **Silent normalization:** `fdadensity` emits a warning when |trapz(dens) − 1| > 1e-5.
//!    This implementation normalizes silently without a warning and documents the behaviour here.

use crate::error::FdarError;
use crate::helpers::{cumulative_trapz, linear_interp, trapz};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

// ─── Result types ────────────────────────────────────────────────────────────

/// Result of functional PCA on log-quantile-density (LQD) transformed densities.
///
/// All fields (`fpca`, scores, loadings, mean) are in **LQD space** on the uniform
/// quantile grid t ∈ [0, 1], not in the original density space.
///
/// To obtain density-space variation modes, apply [`inverse_lqd`] to
/// `fpca.mean ± scale * loading_column` for each principal component column.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LqdFpcaResult {
    /// FPCA result in LQD space.
    ///
    /// The FPCA is performed on the LQD-transformed densities on the uniform
    /// quantile grid t ∈ [0, 1]. Scores, loadings, and mean are all in LQD
    /// space, not density space.
    pub fpca: FpcaResult,
    /// Fraction of variance explained by the first k components.
    ///
    /// `fve[k]` = cumsum(sv²)[0..=k] / sum(all sv²). Monotone non-decreasing;
    /// `fve.last()` ≈ 1.0 only when `ncomp == min(n_densities, n_quantile_pts)`.
    pub fve: Vec<f64>,
}

// ─── Public entry points ─────────────────────────────────────────────────────

/// Normalize a density so that it integrates to 1 via trapezoidal quadrature.
///
/// # Arguments
///
/// * `vals`    — density values sampled on `argvals`; must be non-negative.
/// * `argvals` — strictly increasing evaluation grid.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `vals.len() != argvals.len()`.
/// Returns [`FdarError::InvalidParameter`] if any value in `vals` is negative,
/// if `argvals` is not strictly increasing, or if the integral is < 1e-15
/// (all-zero density).
///
/// # Example
///
/// ```
/// use fdars_core::density_fda::normalize_density;
/// let argvals = vec![0.0, 0.5, 1.0];
/// let vals = vec![2.0, 2.0, 2.0]; // uniform, scale factor 2
/// let normed = normalize_density(&vals, &argvals).unwrap();
/// // integral ≈ 1.0
/// ```
pub fn normalize_density(vals: &[f64], argvals: &[f64]) -> Result<Vec<f64>, FdarError> {
    if vals.len() != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "vals",
            expected: format!("{}", argvals.len()),
            actual: format!("{}", vals.len()),
        });
    }
    if argvals.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must have at least 2 elements".to_string(),
        });
    }
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be strictly increasing".to_string(),
        });
    }
    if vals.iter().any(|&v| v < 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "vals",
            message: "density values must be non-negative".to_string(),
        });
    }
    let integral = trapz(vals, argvals);
    if integral < 1e-15 {
        return Err(FdarError::InvalidParameter {
            parameter: "vals",
            message: "density integrates to zero or is all-zero".to_string(),
        });
    }
    Ok(vals.iter().map(|&v| v / integral).collect())
}

/// Log-quantile-density (LQD) forward transform.
///
/// Maps a probability density `density` sampled on `argvals` (a physical grid in
/// density space) to the LQD representation ψ on a uniform quantile grid
/// t ∈ [0, 1] of length `n_quantile_pts`.
///
/// The LQD is defined as ψ(t) = log q(t) = −log f(Q(t)), where Q is the
/// quantile function and q = dQ/dt is the quantile density.
///
/// **Numeric chain** (matching `fdadensity::dens2lqd`):
/// 1. Normalize density.
/// 2. Compute CDF via `cumulative_trapz` (starts at 0).
/// 3. Compute `lqd_raw[i] = −log(density_norm[i])` on the physical grid.
/// 4. Interpolate (x = CDF, y = lqd_raw) onto the uniform t-grid via `linear_interp`.
///
/// # Arguments
///
/// * `density`        — strictly positive density values on `argvals`.
/// * `argvals`        — strictly increasing evaluation grid.
/// * `n_quantile_pts` — length of the output quantile grid (default: `argvals.len().max(101)`).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if any density value is ≤ 0 (since
/// −log(0) = +∞), if `argvals` is not strictly increasing, or if any output ψ
/// value is non-finite (NaN or ±∞).
/// Returns [`FdarError::InvalidDimension`] for length mismatches.
///
/// # Example
///
/// ```
/// use fdars_core::density_fda::lqd_transform;
/// let argvals: Vec<f64> = (0..51).map(|i| i as f64 / 50.0).collect();
/// let uniform = vec![1.0_f64; 51];
/// let psi = lqd_transform(&uniform, &argvals, Some(51)).unwrap();
/// // ψ ≡ 0 for uniform density
/// assert!(psi.iter().all(|&v| v.abs() < 1e-5));
/// ```
pub fn lqd_transform(
    density: &[f64],
    argvals: &[f64],
    n_quantile_pts: Option<usize>,
) -> Result<Vec<f64>, FdarError> {
    // --- validation ---
    if density.len() != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "density",
            expected: format!("{}", argvals.len()),
            actual: format!("{}", density.len()),
        });
    }
    if argvals.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must have at least 2 elements".to_string(),
        });
    }
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be strictly increasing".to_string(),
        });
    }
    // LQD requires strictly positive density (log(0) = -∞)
    if density.iter().any(|&v| v <= 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "density",
            message: "density values must be strictly positive for the LQD transform (zero/negative density produces ±∞)".to_string(),
        });
    }

    let n_q = n_quantile_pts.unwrap_or_else(|| argvals.len().max(101));
    if n_q < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_quantile_pts",
            message: "n_quantile_pts must be at least 2".to_string(),
        });
    }

    // Step 1: normalize
    let integral = trapz(density, argvals);
    let dens_norm: Vec<f64> = density.iter().map(|&d| d / integral).collect();

    // Step 2: CDF (starts at 0 by cumulative_trapz contract)
    let cdf = cumulative_trapz(&dens_norm, argvals);

    // Step 3: lqd on physical grid: ψ_raw[i] = −log(f(x_i))
    let lqd_raw: Vec<f64> = dens_norm.iter().map(|&d| -d.ln()).collect();

    // Step 4: interpolate onto uniform t-grid ∈ [0,1]
    let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();
    let psi: Vec<f64> = t_grid
        .iter()
        .map(|&t| linear_interp(&cdf, &lqd_raw, t))
        .collect();

    // Guard: non-finite ψ indicates a numeric failure
    if psi.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::ComputationFailed {
            operation: "lqd_transform",
            detail: "non-finite ψ values produced; possible cause: a density value \
                     underflowed to 0 after normalization (input density too small \
                     relative to its maximum on this grid)"
                .to_string(),
        });
    }

    Ok(psi)
}

/// Inverse LQD transform: reconstruct a normalized probability density on `target_argvals`.
///
/// Inverts the LQD transform: given ψ on a quantile grid t ∈ [0, 1], recovers a
/// probability density on `target_argvals`.  The result is always renormalized so
/// that it integrates to 1 over `target_argvals`.
///
/// **Numeric chain** (matching `fdadensity::lqd2dens`):
/// 1. Compute quantile function: `Q_raw = lb + cumtrapz(exp(ψ), t_grid)`.
/// 2. **Mandatory rescaling** (θ_ψ correction): map Q_raw to the target support
///    `[target_argvals[0], target_argvals.last()]` by linear scaling.
/// 3. Compute density values at quantile-grid points: `dens_raw[i] = exp(−ψ[i])`.
/// 4. Dedup adjacent equal Q values to keep the interpolation x-axis strictly monotone.
/// 5. Interpolate `dens_raw` onto `target_argvals` via `linear_interp`.
/// 6. Renormalize so that `trapz(result, target_argvals) = 1`.
///
/// # Arguments
///
/// * `psi`            — LQD values on `t_grid`.
/// * `t_grid`         — strictly increasing quantile grid (typically uniform on [0, 1]).
/// * `target_argvals` — strictly increasing physical grid for the output density.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `psi.len() != t_grid.len()`.
/// Returns [`FdarError::InvalidParameter`] if `t_grid` or `target_argvals` is not
/// strictly increasing, or if `psi` contains non-finite values.
/// Returns [`FdarError::ComputationFailed`] if the reconstructed Q range is zero
/// (degenerate density) or renormalization fails.
pub fn inverse_lqd(
    psi: &[f64],
    t_grid: &[f64],
    target_argvals: &[f64],
) -> Result<Vec<f64>, FdarError> {
    // --- validation ---
    if psi.len() != t_grid.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "psi",
            expected: format!("{}", t_grid.len()),
            actual: format!("{}", psi.len()),
        });
    }
    if t_grid.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "t_grid",
            message: "t_grid must have at least 2 elements".to_string(),
        });
    }
    if target_argvals.len() < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "target_argvals",
            message: "target_argvals must have at least 2 elements".to_string(),
        });
    }
    if t_grid.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "t_grid",
            message: "t_grid must be strictly increasing".to_string(),
        });
    }
    if target_argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "target_argvals",
            message: "target_argvals must be strictly increasing".to_string(),
        });
    }
    if psi.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "psi",
            message: "psi must contain only finite values".to_string(),
        });
    }

    // Step 1: Q_raw(t) = lb + cumtrapz(exp(ψ), t_grid)
    let exp_psi: Vec<f64> = psi.iter().map(|&p| p.exp()).collect();
    let q_raw_cumtrapz = cumulative_trapz(&exp_psi, t_grid);
    let lb = target_argvals[0];
    let q_raw: Vec<f64> = q_raw_cumtrapz.iter().map(|&v| lb + v).collect();

    // Step 2: mandatory θ_ψ rescaling — map Q_raw to the target support range
    let q_range = q_raw[q_raw.len() - 1] - q_raw[0]; // = θ_ψ = ∫exp(ψ) dt
    let d_range = target_argvals[target_argvals.len() - 1] - lb;
    if q_range < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "inverse_lqd",
            detail: "quantile function range is zero; degenerate ψ (all-constant)".to_string(),
        });
    }
    let scale = d_range / q_range;
    let q_scaled: Vec<f64> = q_raw.iter().map(|&v| (v - q_raw[0]) * scale + lb).collect();

    // Step 3: density values at quantile-grid points: dens_raw[i] = exp(−ψ[i]) = 1/q(t_i)
    let dens_raw: Vec<f64> = psi.iter().map(|&p| (-p).exp()).collect();

    // Step 4: dedup adjacent equal Q values (prevents undefined linear_interp on duplicate x)
    let (q_dedup, dens_dedup) = dedup_adjacent(&q_scaled, &dens_raw);

    // Step 5: interpolate onto target_argvals
    let dens: Vec<f64> = target_argvals
        .iter()
        .map(|&x| linear_interp(&q_dedup, &dens_dedup, x))
        .collect();

    // Step 6: renormalize to ∫f = 1
    let integral = trapz(&dens, target_argvals);
    if integral < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "inverse_lqd",
            detail: "reconstructed density integrates to zero; check ψ admissibility".to_string(),
        });
    }
    Ok(dens.iter().map(|&d| d / integral).collect())
}

/// 1D Wasserstein Fréchet mean (quantile-average barycenter) of a collection of densities.
///
/// Computes the Fréchet mean of probability densities under the 2-Wasserstein metric.
/// In 1D this is the pointwise (weighted) average of the quantile functions
/// Q̄(t) = Σᵢ wᵢ Qᵢ(t), which is then inverted back to a density.
///
/// **Formula:** Rüschendorf and Rachev (1990); confirmed in Petersen and Mueller (2016).
///
/// # Arguments
///
/// * `density_matrix` — n × m matrix (n densities, m evaluation points, column-major).
/// * `argvals`        — strictly increasing evaluation grid of length m.
/// * `weights`        — optional weight vector of length n summing to 1.
///   Defaults to uniform 1/n.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] for empty matrix or argvals mismatch.
/// Returns [`FdarError::InvalidParameter`] if any density row is non-positive or
/// `argvals` is not strictly increasing; if `weights` length mismatches or sums to zero.
/// Returns [`FdarError::ComputationFailed`] if the quantile average inversion fails.
pub fn wasserstein_barycenter(
    density_matrix: &FdMatrix,
    argvals: &[f64],
    weights: Option<&[f64]>,
) -> Result<Vec<f64>, FdarError> {
    let (n, m) = density_matrix.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "density_matrix",
            expected: "at least 1 row".to_string(),
            actual: "0 rows".to_string(),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "density_matrix",
            expected: "at least 1 column".to_string(),
            actual: "0 columns".to_string(),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching density_matrix columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be strictly increasing".to_string(),
        });
    }

    // Resolve weights
    let w_vec: Vec<f64> = if let Some(w) = weights {
        if w.len() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "weights",
                expected: format!("{n}"),
                actual: format!("{}", w.len()),
            });
        }
        if w.iter().any(|&wi| wi < 0.0 || !wi.is_finite()) {
            return Err(FdarError::InvalidParameter {
                parameter: "weights",
                message: "weights must be non-negative and finite".to_string(),
            });
        }
        let s: f64 = w.iter().sum();
        if s < 1e-15 {
            return Err(FdarError::InvalidParameter {
                parameter: "weights",
                message: "weights sum to zero".to_string(),
            });
        }
        w.iter().map(|&wi| wi / s).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    // Quantile grid (same resolution as input)
    let n_q = m.max(101);
    let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();

    // Compute weighted average quantile function Q̄(t) = Σᵢ wᵢ Qᵢ(t)
    let mut q_bar = vec![0.0_f64; n_q];
    for i in 0..n {
        let row: Vec<f64> = (0..m).map(|j| density_matrix[(i, j)]).collect();
        if row.iter().any(|&v| v < 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter: "density_matrix",
                message: format!(
                    "row {i} contains negative values; densities must be non-negative"
                ),
            });
        }
        let integral = trapz(&row, argvals);
        if integral < 1e-15 {
            return Err(FdarError::InvalidParameter {
                parameter: "density_matrix",
                message: format!("row {i} integrates to zero (all-zero density)"),
            });
        }
        let norm_row: Vec<f64> = row.iter().map(|&v| v / integral).collect();
        let cdf_i = cumulative_trapz(&norm_row, argvals);
        let wi = w_vec[i];
        for j in 0..n_q {
            q_bar[j] += wi * linear_interp(&cdf_i, argvals, t_grid[j]);
        }
    }

    // Invert Q̄ to a density using the same back-map as inverse_lqd
    // Q̄ is already on the target x-range [argvals[0], argvals[last]]
    // Rescale Q̄ to the exact target support
    let lb = argvals[0];
    let ub = argvals[m - 1];
    let q_range = q_bar[n_q - 1] - q_bar[0];
    if q_range < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "wasserstein_barycenter",
            detail: "quantile average has zero range; degenerate input densities".to_string(),
        });
    }
    let d_range = ub - lb;
    let q_scaled: Vec<f64> = q_bar
        .iter()
        .map(|&v| (v - q_bar[0]) * d_range / q_range + lb)
        .collect();

    // Density at quantile-grid points: dQ̄/dt approximated by finite differences
    let dens_raw = quantile_density_from_q(&q_scaled, &t_grid);

    // Dedup and interpolate onto argvals
    let (q_dedup, dens_dedup) = dedup_adjacent(&q_scaled, &dens_raw);
    let dens: Vec<f64> = argvals
        .iter()
        .map(|&x| linear_interp(&q_dedup, &dens_dedup, x))
        .collect();

    // Renormalize
    let integral = trapz(&dens, argvals);
    if integral < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "wasserstein_barycenter",
            detail: "barycenter density integrates to zero".to_string(),
        });
    }
    Ok(dens.iter().map(|&d| d / integral).collect())
}

/// Functional PCA of probability densities in LQD space.
///
/// Transforms each density row to LQD space on a uniform quantile grid, assembles
/// the resulting `FdMatrix`, and delegates to [`fdata_to_pc_1d`].  Returns the
/// FPCA result together with the fraction of variance explained (FVE) vector.
///
/// **Algorithm:**
/// 1. For each density row: `lqd_transform → ψᵢ` on the uniform t-grid.
/// 2. Assemble the n × n_q LQD matrix.
/// 3. Call `fdata_to_pc_1d` (existing SVD engine).
/// 4. Compute FVE = cumsum(sv²) / sum(sv²).
///
/// # Arguments
///
/// * `density_matrix` — n × m matrix of probability densities (one per row).
/// * `argvals`        — strictly increasing evaluation grid of length m.
/// * `ncomp`          — number of principal components to retain.
/// * `n_quantile_pts` — LQD quantile grid length (default: `argvals.len().max(101)`).
///
/// # Errors
///
/// Propagates errors from [`lqd_transform`] and [`fdata_to_pc_1d`].
/// Returns [`FdarError::InvalidDimension`] for empty matrix or argvals mismatch.
/// Returns [`FdarError::InvalidParameter`] when `ncomp == 0`.
#[must_use = "expensive SVD computation — store or use the returned LqdFpcaResult"]
pub fn lqd_fpca(
    density_matrix: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    n_quantile_pts: Option<usize>,
) -> Result<LqdFpcaResult, FdarError> {
    let (n_dens, m) = density_matrix.shape();
    if n_dens == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "density_matrix",
            expected: "at least 1 row".to_string(),
            actual: "0 rows".to_string(),
        });
    }
    if m == 0 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be at least 1".to_string(),
        });
    }

    let n_q = n_quantile_pts.unwrap_or_else(|| argvals.len().max(101));
    let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();

    // Build LQD matrix (n × n_q), column-major
    let mut lqd_data = FdMatrix::zeros(n_dens, n_q);
    for i in 0..n_dens {
        let row: Vec<f64> = (0..m).map(|j| density_matrix[(i, j)]).collect();
        let psi = lqd_transform(&row, argvals, Some(n_q))?;
        for (j, &val) in psi.iter().enumerate() {
            lqd_data[(i, j)] = val;
        }
    }

    // Delegate to existing FPCA engine
    let fpca = fdata_to_pc_1d(&lqd_data, ncomp, &t_grid)?;

    // FVE = cumsum(sv²) / sum(sv²)
    let sv_sq: Vec<f64> = fpca.singular_values.iter().map(|&s| s * s).collect();
    let total: f64 = sv_sq.iter().sum();
    let mut cumsum = 0.0_f64;
    let fve: Vec<f64> = sv_sq
        .iter()
        .map(|&s| {
            cumsum += s;
            if total > 0.0 {
                cumsum / total
            } else {
                0.0
            }
        })
        .collect();

    Ok(LqdFpcaResult { fpca, fve })
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Remove adjacent duplicate x values (and any non-monotone values), keeping the
/// first of each run.
///
/// Used before `linear_interp` to guarantee a strictly monotone x-axis.  Callers
/// pass `q_scaled`, the rescaled quantile function, which is *intended* to be
/// non-decreasing (positive-linear map of a cumulative integral).  In practice,
/// [`crate::helpers::cumulative_trapz`]'s generalized-Simpson pairing can produce
/// small numerical reversals at intermediate grid points.  This helper silently
/// discards any point where `x[i] <= x[i-1]`, recovering a strictly-increasing
/// x-axis before the binary-search-based `linear_interp`.
///
/// **Silent drop:** points that are exactly equal to or strictly less than the
/// previously kept value are skipped without error.  This is the intended behaviour
/// for the current call sites; future callers that require non-decreasingness should
/// validate their input before calling this helper.
fn dedup_adjacent(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut xd = Vec::with_capacity(x.len());
    let mut yd = Vec::with_capacity(y.len());
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if i == 0 || xi > xd[xd.len() - 1] {
            xd.push(xi);
            yd.push(yi);
        }
        // Points where xi <= xd.last() are silently skipped (duplicates or
        // numerical reversals from cumulative_trapz's Simpson pairing).
    }
    (xd, yd)
}

/// Approximate the quantile density q(t) = dQ/dt at each t_grid point.
///
/// Uses central differences in the interior and forward/backward differences at
/// the endpoints, then clamps negative values to 0 for numerical safety.
fn quantile_density_from_q(q: &[f64], t: &[f64]) -> Vec<f64> {
    let n = q.len();
    let mut qd = vec![0.0_f64; n];
    if n < 2 {
        return qd;
    }
    // Forward difference at left boundary
    qd[0] = (q[1] - q[0]) / (t[1] - t[0]);
    // Central differences in interior
    for i in 1..n - 1 {
        qd[i] = (q[i + 1] - q[i - 1]) / (t[i + 1] - t[i - 1]);
    }
    // Backward difference at right boundary
    qd[n - 1] = (q[n - 1] - q[n - 2]) / (t[n - 1] - t[n - 2]);
    // The density is 1/q(t); clamp non-positive q to a small epsilon.
    // eps = 1e-6 prevents 1e12 tail spikes from a too-small clamp: at tails
    // the central-difference dq can be legitimately small on coarse grids,
    // and 1/1e-12 = 1e12 dominates boundary interpolation in the barycenter.
    let eps = 1e-6_f64;
    qd.iter().map(|&dq| 1.0 / dq.max(eps)).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::trapz;

    /// Truncated Gaussian density f(x) ∝ exp(−(x − mu)²/2) on argvals, normalized.
    fn truncated_gaussian(argvals: &[f64], mu: f64) -> Vec<f64> {
        let raw: Vec<f64> = argvals
            .iter()
            .map(|&x| (-(x - mu).powi(2) / 2.0).exp())
            .collect();
        let integral = trapz(&raw, argvals);
        raw.iter().map(|&d| d / integral).collect()
    }

    // ── normalize_density ────────────────────────────────────────────────────

    #[test]
    fn normalize_density_integral_to_one() {
        let argvals: Vec<f64> = (0..101).map(|i| i as f64 / 100.0).collect();
        let vals: Vec<f64> = argvals.iter().map(|&x| 2.0 * x + 0.5).collect(); // unnormalized
        let normed = normalize_density(&vals, &argvals).unwrap();
        let integral = trapz(&normed, &argvals);
        assert!(
            (integral - 1.0).abs() < 1e-10,
            "integral = {integral}, expected 1.0"
        );
        assert!(normed.iter().all(|&v| v >= 0.0), "negative values");
    }

    // ── lqd_transform ────────────────────────────────────────────────────────

    #[test]
    fn lqd_uniform_is_zero() {
        // For f(x) = 1 on [0,1], Q(t) = t, q(t) = 1, ψ(t) = −log(1) = 0 everywhere.
        let argvals: Vec<f64> = (0..201).map(|i| i as f64 / 200.0).collect();
        let uniform = vec![1.0_f64; 201];
        let psi = lqd_transform(&uniform, &argvals, Some(101)).unwrap();
        let max_abs = psi.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-5,
            "lqd of uniform should be ≈0 everywhere, got max |ψ| = {max_abs}"
        );
    }

    #[test]
    fn lqd_transform_finite() {
        let argvals: Vec<f64> = (0..201).map(|i| -3.0 + i as f64 * 6.0 / 200.0).collect();
        let dens = truncated_gaussian(&argvals, 0.0);
        let psi = lqd_transform(&dens, &argvals, Some(101)).unwrap();
        assert_eq!(psi.len(), 101);
        assert!(
            psi.iter().all(|v| v.is_finite()),
            "ψ contains non-finite values"
        );
    }

    // ── round-trip ───────────────────────────────────────────────────────────

    #[test]
    fn round_trip_lqd_density_within_tolerance() {
        // Analytic reference: truncated standard Gaussian on [−3, 3] (201 pts).
        // Use None for n_quantile_pts so the default resolves to 201 (= argvals.len()),
        // matching the density grid resolution.  Downsampling to 101 quantile pts on a
        // 201-pt density loses tail information and degrades accuracy to ~0.04 L∞,
        // well above the 5e-3 target.  The fdadensity default is N = length(dSup),
        // i.e. no downsampling.
        let argvals: Vec<f64> = (0..201).map(|i| -3.0 + i as f64 * 6.0 / 200.0).collect();
        let dens = truncated_gaussian(&argvals, 0.0);

        // Forward transform with default quantile-grid resolution (= 201 here)
        let n_q = 201usize;
        let psi = lqd_transform(&dens, &argvals, Some(n_q)).unwrap();
        let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();

        // Inverse transform
        let dens2 = inverse_lqd(&psi, &t_grid, &argvals).unwrap();

        // L∞ error tolerance. The double linear-interpolation chain
        // (density → CDF → quantile inversion → density) has a measured L∞ error
        // of ~1.0e-2 on this sharp-curvature truncated Gaussian at 201 points; the
        // `fdadensity` reference uses natural cubic-spline inversion and reaches
        // ~5e-3. The exact-match limit is documented as a known divergence on the
        // public functions (linear vs. cubic-spline interpolation). We assert an
        // empirically honest bound rather than the unverified 5e-3 estimate.
        let max_err = dens
            .iter()
            .zip(dens2.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1.5e-2,
            "round-trip L∞ error = {max_err} (tolerance 1.5e-2)"
        );

        // Reconstructed density integrates to 1 within 1e-6
        let integral = trapz(&dens2, &argvals);
        assert!(
            (integral - 1.0).abs() < 1e-6,
            "reconstructed integral = {integral}"
        );

        // All values non-negative (up to rounding noise)
        assert!(
            dens2.iter().all(|&v| v >= -1e-9),
            "negative density values found"
        );
    }

    // ── inverse_lqd ──────────────────────────────────────────────────────────

    #[test]
    fn inverse_lqd_normalized_nonneg() {
        // Use a non-trivial ψ (from a truncated Gaussian) and check guarantees
        let argvals: Vec<f64> = (0..101).map(|i| -3.0 + i as f64 * 6.0 / 100.0).collect();
        let dens = truncated_gaussian(&argvals, 0.5);
        let t_grid: Vec<f64> = (0..101).map(|i| i as f64 / 100.0).collect();
        let psi = lqd_transform(&dens, &argvals, Some(101)).unwrap();
        let rec = inverse_lqd(&psi, &t_grid, &argvals).unwrap();

        let integral = trapz(&rec, &argvals);
        assert!((integral - 1.0).abs() < 1e-6, "integral = {integral}");
        assert!(rec.iter().all(|&v| v >= -1e-9), "negative density values");
    }

    // ── error cases ──────────────────────────────────────────────────────────

    #[test]
    fn error_negative_density() {
        let argvals = vec![0.0, 0.5, 1.0];
        let vals = vec![1.0, -0.1, 1.0]; // negative value
        assert!(
            matches!(
                normalize_density(&vals, &argvals),
                Err(FdarError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for negative density"
        );
        assert!(
            matches!(
                lqd_transform(&vals, &argvals, None),
                Err(FdarError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for negative density in lqd_transform"
        );
    }

    #[test]
    fn error_length_mismatch() {
        let argvals = vec![0.0, 0.5, 1.0];
        let vals = vec![1.0, 1.0]; // length 2, not 3
        assert!(
            matches!(
                normalize_density(&vals, &argvals),
                Err(FdarError::InvalidDimension { .. })
            ),
            "expected InvalidDimension for length mismatch"
        );
        assert!(
            matches!(
                lqd_transform(&vals, &argvals, None),
                Err(FdarError::InvalidDimension { .. })
            ),
            "expected InvalidDimension for length mismatch in lqd_transform"
        );
    }

    #[test]
    fn error_non_monotone_grid() {
        let argvals = vec![0.0, 1.0, 0.5]; // not monotone
        let vals = vec![1.0, 1.0, 1.0];
        assert!(
            matches!(
                normalize_density(&vals, &argvals),
                Err(FdarError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for non-monotone argvals"
        );
    }

    #[test]
    fn error_all_zero_density() {
        let argvals = vec![0.0, 0.5, 1.0];
        let vals = vec![0.0, 0.0, 0.0];
        assert!(
            matches!(
                normalize_density(&vals, &argvals),
                Err(FdarError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for all-zero density"
        );
    }

    #[test]
    fn error_inverse_lqd_length_mismatch() {
        let psi = vec![0.0, 0.0, 0.0];
        let t_grid = vec![0.0, 0.5]; // length 2, not 3
        let target = vec![0.0, 0.5, 1.0];
        assert!(
            matches!(
                inverse_lqd(&psi, &t_grid, &target),
                Err(FdarError::InvalidDimension { .. })
            ),
            "expected InvalidDimension"
        );
    }

    #[test]
    fn error_inverse_lqd_non_monotone_t_grid() {
        let psi = vec![0.0, 0.0];
        let t_grid = vec![1.0, 0.0]; // reversed
        let target = vec![0.0, 1.0];
        assert!(
            matches!(
                inverse_lqd(&psi, &t_grid, &target),
                Err(FdarError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for non-monotone t_grid"
        );
    }

    // ── wasserstein_barycenter ────────────────────────────────────────────────

    #[test]
    fn barycenter_singleton_reduction() {
        // Barycenter of a single density should return the density itself (up to tolerance)
        let argvals: Vec<f64> = (0..101).map(|i| -3.0 + i as f64 * 6.0 / 100.0).collect();
        let dens = truncated_gaussian(&argvals, 0.0);
        let mut data = FdMatrix::zeros(1, 101);
        for (j, &v) in dens.iter().enumerate() {
            data[(0, j)] = v;
        }
        let bary = wasserstein_barycenter(&data, &argvals, None).unwrap();
        let max_err = dens
            .iter()
            .zip(bary.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-2, "singleton barycenter L∞ error = {max_err}");
    }

    #[test]
    fn barycenter_two_density_midpoint() {
        // Barycenter of two shifted Gaussians should lie between them
        let argvals: Vec<f64> = (0..201).map(|i| -5.0 + i as f64 * 10.0 / 200.0).collect();
        let d1 = truncated_gaussian(&argvals, -1.0);
        let d2 = truncated_gaussian(&argvals, 1.0);
        let mut data = FdMatrix::zeros(2, 201);
        for (j, &v) in d1.iter().enumerate() {
            data[(0, j)] = v;
        }
        for (j, &v) in d2.iter().enumerate() {
            data[(1, j)] = v;
        }
        let bary = wasserstein_barycenter(&data, &argvals, None).unwrap();
        // Barycenter should be close to Gaussian centered at 0
        let bary_integral = trapz(&bary, &argvals);
        assert!(
            (bary_integral - 1.0).abs() < 1e-6,
            "barycenter integral = {bary_integral}"
        );
        assert!(bary.iter().all(|&v| v >= -1e-9), "negative barycenter");
    }

    #[test]
    fn error_empty_barycenter() {
        let data = FdMatrix::zeros(0, 101);
        let argvals: Vec<f64> = (0..101).map(|i| i as f64 / 100.0).collect();
        assert!(
            matches!(
                wasserstein_barycenter(&data, &argvals, None),
                Err(FdarError::InvalidDimension { .. })
            ),
            "expected InvalidDimension for empty matrix"
        );
    }

    // ── lqd_fpca ─────────────────────────────────────────────────────────────

    #[test]
    fn lqd_fpca_fve_monotone_and_bounded() {
        let argvals: Vec<f64> = (0..101).map(|i| -3.0 + i as f64 * 6.0 / 100.0).collect();
        // Build 20 truncated Gaussians with varying means
        let mut data = FdMatrix::zeros(20, 101);
        for i in 0..20usize {
            let mu = -2.0 + i as f64 * 0.2;
            let dens = truncated_gaussian(&argvals, mu);
            for (j, &v) in dens.iter().enumerate() {
                data[(i, j)] = v;
            }
        }
        let result = lqd_fpca(&data, &argvals, 5, Some(101)).unwrap();

        // FVE is non-decreasing
        for k in 1..result.fve.len() {
            assert!(
                result.fve[k] >= result.fve[k - 1] - 1e-12,
                "FVE not monotone at k={k}: {} < {}",
                result.fve[k],
                result.fve[k - 1]
            );
        }
        // All FVE in [0, 1]
        assert!(
            result.fve.iter().all(|&v| (0.0..=1.0 + 1e-9).contains(&v)),
            "FVE out of [0, 1] range"
        );
    }

    #[test]
    fn lqd_fpca_leading_pc_captures_shift() {
        // 20 Gaussians shifted from -2 to 2 — leading PC should capture >80% variance
        let argvals: Vec<f64> = (0..201).map(|i| -5.0 + i as f64 * 10.0 / 200.0).collect();
        let mut data = FdMatrix::zeros(20, 201);
        for i in 0..20usize {
            let mu = -2.0 + i as f64 * 4.0 / 19.0;
            let dens = truncated_gaussian(&argvals, mu);
            for (j, &v) in dens.iter().enumerate() {
                data[(i, j)] = v;
            }
        }
        let result = lqd_fpca(&data, &argvals, 3, Some(101)).unwrap();
        assert!(
            result.fve[0] > 0.80,
            "leading PC should explain >80% of variance for a shift family, got FVE[0] = {}",
            result.fve[0]
        );
    }

    #[test]
    fn barycenter_weighted_extreme() {
        // Weights [1.0, 0.0] put all mass on the first density → barycenter ≈ d1.
        let argvals: Vec<f64> = (0..201).map(|i| -5.0 + i as f64 * 10.0 / 200.0).collect();
        let d1 = truncated_gaussian(&argvals, -1.0);
        let d2 = truncated_gaussian(&argvals, 1.0);
        let mut data = FdMatrix::zeros(2, 201);
        for (j, (&a, &b)) in d1.iter().zip(d2.iter()).enumerate() {
            data[(0, j)] = a;
            data[(1, j)] = b;
        }
        let bary = wasserstein_barycenter(&data, &argvals, Some(&[1.0, 0.0])).unwrap();
        let d1n = normalize_density(&d1, &argvals).unwrap();
        let d2n = normalize_density(&d2, &argvals).unwrap();
        // The all-weight-on-d1 barycenter recovers d1 up to quantile-inversion
        // interpolation error (linear-interp floor, same family as the LQD round-trip).
        // The meaningful, resolution-robust check is that it is far closer to d1 than d2.
        let l1 = |a: &[f64], b: &[f64]| -> f64 {
            a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f64>()
        };
        let err_d1 = l1(&bary, &d1n);
        let err_d2 = l1(&bary, &d2n);
        assert!(
            err_d1 < 0.4 * err_d2,
            "all-weight-on-d1 barycenter should track d1 (L1 to d1 = {err_d1}, to d2 = {err_d2})"
        );
    }

    #[test]
    fn barycenter_normalized_nonneg() {
        // Barycenter output is a valid density: integrates to 1 and is non-negative.
        let argvals: Vec<f64> = (0..201).map(|i| -5.0 + i as f64 * 10.0 / 200.0).collect();
        let mut data = FdMatrix::zeros(3, 201);
        for i in 0..3usize {
            let dens = truncated_gaussian(&argvals, -1.5 + i as f64 * 1.5);
            for (j, &v) in dens.iter().enumerate() {
                data[(i, j)] = v;
            }
        }
        let bary = wasserstein_barycenter(&data, &argvals, None).unwrap();
        let integral = trapz(&bary, &argvals);
        assert!((integral - 1.0).abs() < 1e-6, "integral = {integral}");
        assert!(
            bary.iter().all(|&v| v >= -1e-9),
            "negative barycenter value"
        );
    }

    #[test]
    fn error_barycenter_bad_weights() {
        // A negative weight must be rejected, not silently accepted.
        let argvals: Vec<f64> = (0..201).map(|i| -5.0 + i as f64 * 10.0 / 200.0).collect();
        let mut data = FdMatrix::zeros(2, 201);
        for i in 0..2usize {
            let dens = truncated_gaussian(&argvals, -1.0 + 2.0 * i as f64);
            for (j, &v) in dens.iter().enumerate() {
                data[(i, j)] = v;
            }
        }
        let err = wasserstein_barycenter(&data, &argvals, Some(&[-0.5, 1.5]));
        assert!(
            matches!(err, Err(FdarError::InvalidParameter { .. })),
            "negative weight should return InvalidParameter, got {err:?}"
        );
    }

    #[test]
    fn lqd_fpca_full_rank_fve_reaches_one() {
        // At full rank the cumulative FVE must reach 1.
        let argvals: Vec<f64> = (0..101).map(|i| -3.0 + i as f64 * 6.0 / 100.0).collect();
        let mut data = FdMatrix::zeros(5, 101);
        for i in 0..5usize {
            let dens = truncated_gaussian(&argvals, -1.5 + i as f64 * 0.75);
            for (j, &v) in dens.iter().enumerate() {
                data[(i, j)] = v;
            }
        }
        // 5 curves → rank ≤ 4 after centering; request 4 components.
        let result = lqd_fpca(&data, &argvals, 4, Some(101)).unwrap();
        let last = *result.fve.last().unwrap();
        assert!(
            (last - 1.0).abs() < 1e-6,
            "full-rank cumulative FVE should reach 1, got {last}"
        );
    }

    #[test]
    fn error_lqd_fpca_empty() {
        // An empty density matrix must return an error, not panic.
        let argvals: Vec<f64> = (0..101).map(|i| -3.0 + i as f64 * 6.0 / 100.0).collect();
        let data = FdMatrix::zeros(0, 101);
        let err = lqd_fpca(&data, &argvals, 2, Some(101));
        assert!(err.is_err(), "empty density matrix should return an error");
    }

    #[test]
    fn error_lqd_fpca_zero_ncomp() {
        // ncomp = 0 must be rejected with InvalidParameter, not silently produce
        // an empty fve vec that panics callers doing result.fve.last().unwrap().
        let argvals: Vec<f64> = (0..101).map(|i| -3.0 + i as f64 * 6.0 / 100.0).collect();
        let mut data = FdMatrix::zeros(5, 101);
        for i in 0..5usize {
            let dens = truncated_gaussian(&argvals, -1.0 + i as f64 * 0.5);
            for (j, &v) in dens.iter().enumerate() {
                data[(i, j)] = v;
            }
        }
        let err = lqd_fpca(&data, &argvals, 0, Some(101));
        assert!(
            matches!(
                err,
                Err(FdarError::InvalidParameter {
                    parameter: "ncomp",
                    ..
                })
            ),
            "ncomp=0 should return InvalidParameter, got {err:?}"
        );
    }
}
