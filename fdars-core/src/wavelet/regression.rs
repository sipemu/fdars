//! Wavelet-domain scalar-on-function regression (`wcr`, WAV-03).
//!
//! `wcr` transforms each functional predictor curve into its multi-level DWT
//! coefficient pyramid (via the Phase 69 primitive), concatenates the bands into a
//! single per-curve coefficient vector, and fits a scalar-on-function regression
//! **in coefficient space** — either PCR (reusing [`crate::regression::fdata_to_pc_1d`])
//! or PLS (reusing [`crate::regression::fdata_to_pls_1d`]). The fitted
//! coefficient-space weights are then mapped back to the time-domain functional
//! coefficient β(t) by the inverse DWT ([`crate::wavelet::reconstruct`]).
//!
//! ## Why this recovers β(t) exactly
//!
//! The multi-level orthogonal DWT is a linear, orthonormal map `W`: the design row
//! for curve `i` is `c_i = W x_i` (wavelet coefficients). If the true relationship
//! is `y = α + C β_c` in coefficient space (with `C` the coefficient design), then
//! in the time domain `y = α + X (Wᵀ β_c)`, so the time-domain coefficient is
//! `β(t) = Wᵀ β_c` — exactly the inverse DWT of the coefficient-space weights.
//! [`coeff_weights_to_beta_t`] performs that inverse DWT.
//!
//! ## Shared seams (reused by the `wnet` regressor, Plan 70-02)
//!
//! - [`curves_to_coeff_design`] — the curves → concatenated-coefficient-design seam.
//! - [`coeff_weights_to_beta_t`] — the coefficient-weights → β(t) seam.
//!
//! No crate-root or prelude re-exports are added here (deferred to Phase 71); the
//! module is reachable only as `crate::wavelet::regression::...`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, fdata_to_pls_1d};
use crate::wavelet::{decompose_matrix, reconstruct, BoundaryMode, WaveletCoeffs, WaveletFamily};

/// Which coefficient-space fit `wcr` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum WcrMethod {
    /// Principal-component regression on the wavelet-coefficient design
    /// (reuses [`crate::regression::fdata_to_pc_1d`]).
    #[default]
    Pcr,
    /// Partial-least-squares regression on the wavelet-coefficient design
    /// (reuses [`crate::regression::fdata_to_pls_1d`]).
    Pls,
}

/// Configuration for [`wcr`].
///
/// The DWT parameters (`family`, `mode`, `level`) select the wavelet basis the
/// curves are transformed into; `ncomp` and `method` select the coefficient-space
/// fit. [`Default`] is db4 / periodic / auto-depth / PCR with `ncomp == 5`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct WcrConfig {
    /// Wavelet family for the DWT of each curve (default [`WaveletFamily::Daubechies(4)`]).
    pub family: WaveletFamily,
    /// Boundary handling for the DWT (default [`BoundaryMode::Periodic`]).
    pub mode: BoundaryMode,
    /// Explicit decomposition depth; `None` (default) uses the maximum useful level.
    pub level: Option<usize>,
    /// Number of coefficient-space components (FPC or PLS) to fit.
    pub ncomp: usize,
    /// Which coefficient-space regressor to use (default [`WcrMethod::Pcr`]).
    pub method: WcrMethod,
}

impl Default for WcrConfig {
    fn default() -> Self {
        Self {
            family: WaveletFamily::Daubechies(4),
            mode: BoundaryMode::Periodic,
            level: None,
            ncomp: 5,
            method: WcrMethod::Pcr,
        }
    }
}

/// Result of a [`wcr`] fit.
///
/// Carries the time-domain functional coefficient β(t), the coefficient-space
/// weights it was reconstructed from, fitted values / residuals, and the DWT
/// configuration a future `predict` (Phase 71) needs to reproduce the transform.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct WcrResult {
    /// Intercept α.
    pub intercept: f64,
    /// Time-domain functional coefficient β(t) (length `m` = curve length).
    pub beta_t: Vec<f64>,
    /// Fitted response values (length `n`).
    pub fitted_values: Vec<f64>,
    /// Residuals `y - ŷ` (length `n`).
    pub residuals: Vec<f64>,
    /// Effective number of coefficient-space components used.
    pub ncomp: usize,
    /// The fitting method used.
    pub method: WcrMethod,
    /// Coefficient-space functional coefficient (length `P` = total wavelet coefficients).
    pub coeff_weights: Vec<f64>,
    /// Wavelet family used for the DWT (for reproducing the transform in prediction).
    pub family: WaveletFamily,
    /// Boundary mode used for the DWT.
    pub mode: BoundaryMode,
    /// Effective decomposition depth used.
    pub level: usize,
}

/// Band layout of a per-curve wavelet-coefficient vector — everything
/// [`reconstruct`] needs to rebuild a [`WaveletCoeffs`] shell from a flat vector.
///
/// Coefficients are concatenated **finest-first**: `[approx ++ details[0] ++
/// details[1] ++ ...]`, matching [`WaveletCoeffs`] band order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CoeffLayout {
    /// Length of the coarse approximation band (the first `approx_len` coefficients).
    pub(crate) approx_len: usize,
    /// Detail-band lengths, finest-first (matching [`WaveletCoeffs::details`] order).
    pub(crate) detail_lens: Vec<usize>,
    /// Original signal length (curve length `m`).
    pub(crate) signal_len: usize,
    /// Wavelet family used for the transform.
    pub(crate) family: WaveletFamily,
    /// Boundary mode used for the transform.
    pub(crate) mode: BoundaryMode,
    /// Per-level analysis-input lengths, finest-first (the [`WaveletCoeffs::level_lens`]).
    pub(crate) level_lens: Vec<usize>,
}

impl CoeffLayout {
    /// Total number of wavelet coefficients per curve (`P` = approx + all details).
    pub(crate) fn total_len(&self) -> usize {
        self.approx_len + self.detail_lens.iter().sum::<usize>()
    }

    /// Number of decomposition levels.
    pub(crate) fn levels(&self) -> usize {
        self.detail_lens.len()
    }
}

/// Flatten one curve's [`WaveletCoeffs`] into a finest-first coefficient vector.
fn coeffs_to_row(coeffs: &WaveletCoeffs) -> Vec<f64> {
    let mut row = Vec::with_capacity(
        coeffs.approx.len() + coeffs.details.iter().map(Vec::len).sum::<usize>(),
    );
    row.extend_from_slice(&coeffs.approx);
    for band in &coeffs.details {
        row.extend_from_slice(band);
    }
    row
}

/// SHARED SEAM. Transform every curve (row) of `data` into its concatenated
/// wavelet-coefficient vector, assembling an `n × P` design matrix.
///
/// Each row of the returned [`FdMatrix`] is one curve's `[approx ++ details...]`
/// (finest-first). All curves must share the same length (guaranteed by a common
/// evaluation grid), so every row has the same layout — the returned [`CoeffLayout`]
/// records that shared band structure so coefficient-space weights can be scattered
/// back to a [`WaveletCoeffs`] for the inverse DWT.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty (surfaced from
///   [`decompose_matrix`]), or if the per-curve coefficient layouts disagree
///   (should not happen for a common grid).
/// - [`FdarError::InvalidParameter`] if the family is unsupported or the level is
///   out of range (surfaced from [`decompose_matrix`]).
pub(crate) fn curves_to_coeff_design(
    data: &FdMatrix,
    family: WaveletFamily,
    mode: BoundaryMode,
    level: Option<usize>,
) -> Result<(FdMatrix, CoeffLayout), FdarError> {
    let per_curve = decompose_matrix(data, family.clone(), mode, level)?;
    // per_curve is non-empty: decompose_matrix rejects zero-row matrices.
    let first = &per_curve[0];
    let layout = CoeffLayout {
        approx_len: first.approx.len(),
        detail_lens: first.details.iter().map(Vec::len).collect(),
        signal_len: first.signal_len,
        family,
        mode,
        level_lens: first.level_lens.clone(),
    };
    let p = layout.total_len();
    let n = per_curve.len();

    // Assemble the n × P design in column-major order, validating that every curve
    // produced the same band structure as curve 0.
    let mut flat = vec![0.0_f64; n * p];
    for (i, coeffs) in per_curve.iter().enumerate() {
        if coeffs.approx.len() != layout.approx_len
            || coeffs.details.len() != layout.detail_lens.len()
            || coeffs
                .details
                .iter()
                .zip(&layout.detail_lens)
                .any(|(band, &len)| band.len() != len)
            || coeffs.signal_len != layout.signal_len
        {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("all curves share curve-0 coefficient layout (P = {p})"),
                actual: format!("curve {i} produced a different band structure"),
            });
        }
        let row = coeffs_to_row(coeffs);
        for (j, &v) in row.iter().enumerate() {
            flat[i + j * n] = v;
        }
    }

    let design = FdMatrix::from_column_major(flat, n, p)?;
    Ok((design, layout))
}

/// SHARED SEAM. Map a `P`-length coefficient-space weight vector back to the time
/// domain via the inverse DWT.
///
/// Splits `weights` into the approximation band and the finest-first detail bands per
/// `layout`, packs them into a [`WaveletCoeffs`] shell, and calls [`reconstruct`],
/// yielding β(t) of length `layout.signal_len`.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `weights.len()` does not equal the total
///   coefficient count implied by `layout`.
/// - [`FdarError::InvalidParameter`] if the family is unsupported (surfaced from
///   [`reconstruct`]).
pub(crate) fn coeff_weights_to_beta_t(
    weights: &[f64],
    layout: &CoeffLayout,
) -> Result<Vec<f64>, FdarError> {
    let expected = layout.total_len();
    if weights.len() != expected {
        return Err(FdarError::InvalidDimension {
            parameter: "weights",
            expected: format!("{expected} coefficients (approx + all detail bands)"),
            actual: format!("{} coefficients", weights.len()),
        });
    }

    let approx = weights[..layout.approx_len].to_vec();
    let mut details: Vec<Vec<f64>> = Vec::with_capacity(layout.detail_lens.len());
    let mut offset = layout.approx_len;
    for &len in &layout.detail_lens {
        details.push(weights[offset..offset + len].to_vec());
        offset += len;
    }

    let coeffs = WaveletCoeffs {
        approx,
        details,
        levels: layout.levels(),
        signal_len: layout.signal_len,
        family: layout.family.clone(),
        mode: layout.mode,
        level_lens: layout.level_lens.clone(),
    };
    reconstruct(&coeffs)
}

// ---------------------------------------------------------------------------
// Local OLS helpers (mirrors scalar_on_function's private OLS path; those helpers
// are module-private there, so wcr carries a small self-contained normal-equations
// solver rather than widening their visibility).
// ---------------------------------------------------------------------------

/// Build the OLS design `[1, scores]` (n × (1 + ncomp)).
fn design_with_intercept(scores: &FdMatrix, ncomp: usize) -> FdMatrix {
    let n = scores.nrows();
    let mut design = FdMatrix::zeros(n, 1 + ncomp);
    for i in 0..n {
        design[(i, 0)] = 1.0;
        for k in 0..ncomp {
            design[(i, 1 + k)] = scores[(i, k)];
        }
    }
    design
}

/// Solve OLS `min ||Xb - y||²` via normal equations with Cholesky.
fn ols_solve(x: &FdMatrix, y: &[f64]) -> Result<Vec<f64>, FdarError> {
    let (n, p) = x.shape();
    if n < p || p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "design matrix",
            expected: format!("n >= p and p > 0 (p={p})"),
            actual: format!("n={n}, p={p}"),
        });
    }
    // X'X (p × p) and X'y (p).
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];
    for a in 0..p {
        for b in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, a)] * x[(i, b)];
            }
            xtx[a + b * p] = s;
        }
        let mut sy = 0.0;
        for i in 0..n {
            sy += x[(i, a)] * y[i];
        }
        xty[a] = sy;
    }
    let l = cholesky_factor(&xtx, p)?;
    Ok(cholesky_solve(&l, &xty, p))
}

/// Cholesky factor `A = L Lᵀ` (column-major `p × p`, lower-triangular `L`).
fn cholesky_factor(a: &[f64], p: usize) -> Result<Vec<f64>, FdarError> {
    let mut l = vec![0.0_f64; p * p];
    for j in 0..p {
        let mut diag = a[j + j * p];
        for k in 0..j {
            diag -= l[j + k * p] * l[j + k * p];
        }
        if diag <= 0.0 {
            return Err(FdarError::ComputationFailed {
                operation: "Cholesky factorization (wcr OLS)",
                detail: "design matrix X'X is not positive definite; try reducing ncomp"
                    .to_string(),
            });
        }
        let ljj = diag.sqrt();
        l[j + j * p] = ljj;
        for i in (j + 1)..p {
            let mut s = a[i + j * p];
            for k in 0..j {
                s -= l[i + k * p] * l[j + k * p];
            }
            l[i + j * p] = s / ljj;
        }
    }
    Ok(l)
}

/// Solve `L Lᵀ b = rhs` by forward then back substitution.
fn cholesky_solve(l: &[f64], rhs: &[f64], p: usize) -> Vec<f64> {
    // Forward: L z = rhs.
    let mut z = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i + k * p] * z[k];
        }
        z[i] = s / l[i + i * p];
    }
    // Back: Lᵀ b = z.
    let mut b = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in (i + 1)..p {
            s -= l[k + i * p] * b[k];
        }
        b[i] = s / l[i + i * p];
    }
    b
}

/// Recover the plain-dot coefficient-space functional coefficient β_coeff.
///
/// The fit's predictions satisfy `fitted_i = intercept + ⟨centered_row_i, β_coeff⟩`
/// (plain dot) for a unique `β_coeff` in the span of the (full-column-rank, `P ≤ n`)
/// coefficient design. This regresses the centered fitted contribution
/// `fitted_i - intercept` onto the column-centered design via the normal equations,
/// recovering that exact `β_coeff` independently of which reduced-rank method (PCR or
/// PLS) produced the fit or which internal integration weighting it used.
fn recover_coeff_weights(
    design: &FdMatrix,
    fitted: &[f64],
    intercept: f64,
) -> Result<Vec<f64>, FdarError> {
    let (n, p) = design.shape();
    // Column means (centering absorbs the intercept).
    let col_means: Vec<f64> = (0..p)
        .map(|j| design.column(j).iter().sum::<f64>() / n as f64)
        .collect();
    // Centered design X_c and centered target r = fitted - intercept.
    let mut xc = FdMatrix::zeros(n, p);
    for j in 0..p {
        for i in 0..n {
            xc[(i, j)] = design[(i, j)] - col_means[j];
        }
    }
    let r: Vec<f64> = fitted.iter().map(|&f| f - intercept).collect();
    // Normal equations X_c' X_c b = X_c' r.
    let mut xtx = vec![0.0_f64; p * p];
    let mut xtr = vec![0.0_f64; p];
    for a in 0..p {
        for b in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += xc[(i, a)] * xc[(i, b)];
            }
            xtx[a + b * p] = s;
        }
        let mut sr = 0.0;
        for i in 0..n {
            sr += xc[(i, a)] * r[i];
        }
        xtr[a] = sr;
    }
    // Ridge-nudge the diagonal for numerical stability against rank-deficient bands
    // (near-zero-variance coefficient columns from short signals); tiny relative to
    // the trace, so it does not perturb a well-posed recovery.
    let trace: f64 = (0..p).map(|j| xtx[j + j * p]).sum();
    let eps = 1e-10 * (trace / p as f64).max(1e-12);
    for j in 0..p {
        xtx[j + j * p] += eps;
    }
    let l = cholesky_factor(&xtx, p)?;
    Ok(cholesky_solve(&l, &xtr, p))
}

/// Compute fitted values `ŷ = X b`.
fn compute_fitted(design: &FdMatrix, coeffs: &[f64]) -> Vec<f64> {
    let (n, p) = design.shape();
    (0..n)
        .map(|i| {
            let mut yhat = 0.0;
            for j in 0..p {
                yhat += design[(i, j)] * coeffs[j];
            }
            yhat
        })
        .collect()
}

// ---------------------------------------------------------------------------
// wcr entry point
// ---------------------------------------------------------------------------

/// Fit the wavelet-domain scalar-on-function regressor `wcr` (WAV-03).
///
/// Transforms every curve into its wavelet-coefficient vector, fits PCR or PLS in
/// coefficient space (per `config.method`), and reconstructs the time-domain
/// functional coefficient β(t) via the inverse DWT.
///
/// The coefficient index is treated as an abstract basis: the PCR/PLS calls receive
/// a uniform grid `0..P` as their `argvals` (Simpson integration weights over that
/// grid), since wavelet coefficients carry no intrinsic spacing.
///
/// # Arguments
/// * `data` — functional predictor matrix (n × m), one curve per row.
/// * `y` — scalar response (length n).
/// * `config` — DWT + coefficient-space fit configuration.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows, zero columns,
///   or `y.len() != n`.
/// - [`FdarError::InvalidParameter`] if `config.ncomp == 0`, or if the DWT rejects
///   the family/level (surfaced from [`decompose_matrix`]).
/// - [`FdarError::ComputationFailed`] if the underlying PCA/PLS or OLS fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn wcr(data: &FdMatrix, y: &[f64], config: &WcrConfig) -> Result<WcrResult, FdarError> {
    let (n, m) = data.shape();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows (observations)".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column (evaluation point)".to_string(),
            actual: format!("{m} columns"),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements (== data rows)"),
            actual: format!("{} elements", y.len()),
        });
    }
    if config.ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be >= 1".to_string(),
        });
    }

    // Curves -> coefficient design (shared seam). Surfaces DWT errors unchanged.
    let (design, layout) =
        curves_to_coeff_design(data, config.family.clone(), config.mode, config.level)?;
    let p = design.ncols();

    // Coefficients form an abstract basis: use a uniform 0..P grid for integration.
    let argvals: Vec<f64> = (0..p).map(|j| j as f64).collect();

    // Clamp effective ncomp to the fittable rank. The OLS design is `[1, scores]`
    // (n × (ncomp + 1)), so `ols_solve` needs ncomp + 1 <= n, i.e. ncomp <= n - 1;
    // otherwise a valid small-n call (e.g. default ncomp = 5 with n <= 5) would be
    // rejected by ols_solve's `n < p` guard. `n >= 3` is enforced above, so
    // `n.saturating_sub(1) >= 2`.
    let ncomp = config.ncomp.min(n.saturating_sub(1)).min(p);

    // Fit in coefficient space: PCR or PLS yields reduced-rank scores, then OLS on
    // [1, scores] gives the intercept and fitted values.
    let (scores, ncomp) = match config.method {
        WcrMethod::Pcr => {
            let fpca = fdata_to_pc_1d(&design, ncomp, &argvals)?;
            let k = fpca.scores.ncols();
            (fpca.scores, k)
        }
        WcrMethod::Pls => {
            let pls = fdata_to_pls_1d(&design, y, ncomp, &argvals)?;
            let k = pls.scores.ncols();
            (pls.scores, k)
        }
    };
    let ols_design = design_with_intercept(&scores, ncomp);
    let coeffs = ols_solve(&ols_design, y)?;
    let intercept = coeffs[0];
    let fitted_values = compute_fitted(&ols_design, &coeffs);

    // Recover the coefficient-space functional coefficient β_coeff directly in the raw
    // wavelet-coefficient basis, so β(t) acts on a curve by the plain functional inner
    // product ⟨coeffs(curve), β_coeff⟩ (matching decompose → concatenate → dot).
    //
    // The score-projection recovery (Σ_k γ_k · rotation/weight_k) instead yields β_coeff
    // in each method's *internal* integration-weighted inner product (sqrt-weighted for
    // PCR's SVD, int-weighted for PLS's NIPALS), which does not match a plain dot. Since
    // the reduced-rank fitted values lie exactly in the span of the coefficient design
    // (full column rank here, P ≤ n), regressing the centered fitted contribution back
    // onto the centered design recovers the exact, method-agnostic plain-dot β_coeff.
    let coeff_weights = recover_coeff_weights(&design, &fitted_values, intercept)?;

    // β(t) via inverse DWT of the coefficient-space weights (shared seam).
    let beta_t = coeff_weights_to_beta_t(&coeff_weights, &layout)?;

    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();

    Ok(WcrResult {
        intercept,
        beta_t,
        fitted_values,
        residuals,
        ncomp,
        method: config.method,
        coeff_weights,
        family: config.family.clone(),
        mode: config.mode,
        level: layout.levels(),
    })
}

// ===========================================================================
// wnet — wavelet-domain elastic-net scalar-on-function regressor (WAV-04)
// ===========================================================================
//
// `wnet` is the sparse/elastic-net half of the wavelet-domain regressor pair.
// It reuses the shared `curves_to_coeff_design` / `coeff_weights_to_beta_t`
// seams above, but fits an **elastic-net** (L1 lasso + L2 ridge) directly on
// the wavelet-coefficient design via a NEW thin per-coefficient coordinate-
// descent adapter ([`elastic_net_cd`]), with a deterministic cross-validated λ
// ([`wnet_cv_lambda`]). A sparse wavelet basis is exactly where L1 shrinkage
// shines: localized signal concentrates in a few coefficients, and the L1
// penalty drives the rest to exactly zero.
//
// The per-coefficient CD is modeled on the group-lasso soft-threshold PATTERN
// in `scalar_on_function::additive` (partial-residual → coordinate update →
// shrink) but is scalar-per-coefficient (elastic-net), not group-lasso.

/// Configuration for [`wnet`].
///
/// The DWT parameters (`family`, `mode`, `level`) select the wavelet basis the
/// curves are transformed into (same defaults as [`WcrConfig`]: db4 / periodic /
/// auto-depth). `alpha` mixes L1 vs L2 (`alpha == 1` is pure lasso, `alpha == 0`
/// is pure ridge), and the remaining fields drive the deterministic K-fold
/// cross-validated λ search.
///
/// [`Default`] is db4 / periodic / auto-depth, `alpha == 0.5`, an auto geometric
/// λ grid of 50 values, 5 folds, fixed seed 0, `max_iter == 1000`, `tol == 1e-6`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct WnetConfig {
    /// Wavelet family for the DWT of each curve (default [`WaveletFamily::Daubechies(4)`]).
    pub family: WaveletFamily,
    /// Boundary handling for the DWT (default [`BoundaryMode::Periodic`]).
    pub mode: BoundaryMode,
    /// Explicit decomposition depth; `None` (default) uses the maximum useful level.
    pub level: Option<usize>,
    /// Elastic-net mixing parameter ∈ [0, 1]: `1.0` is pure L1 (lasso),
    /// `0.0` is pure L2 (ridge). Default `0.5`.
    pub alpha: f64,
    /// Explicit λ grid to search. `None` (default) auto-builds a geometric grid.
    pub lambda_grid: Option<Vec<f64>>,
    /// Number of λ values in the auto geometric grid (used when `lambda_grid` is
    /// `None`). Default `50`.
    pub n_lambda: usize,
    /// Number of cross-validation folds. Default `5`.
    pub n_folds: usize,
    /// Fixed RNG seed for the (deterministic) fold partition. Default `0`.
    pub seed: u64,
    /// Maximum coordinate-descent sweeps. Default `1000`.
    pub max_iter: usize,
    /// Coordinate-descent convergence tolerance (max |Δβ| per sweep). Default `1e-6`.
    pub tol: f64,
}

impl Default for WnetConfig {
    fn default() -> Self {
        Self {
            family: WaveletFamily::Daubechies(4),
            mode: BoundaryMode::Periodic,
            level: None,
            alpha: 0.5,
            lambda_grid: None,
            n_lambda: 50,
            n_folds: 5,
            seed: 0,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

/// Result of a [`wnet`] fit.
///
/// Carries the time-domain functional coefficient β(t), the sparse coefficient-
/// space weights it was reconstructed from, the indices of the nonzero
/// (selected) coefficients, the CV-selected λ, the elastic-net mixing `alpha`,
/// fitted values / residuals, and the DWT configuration a future `predict`
/// (Phase 71) needs to reproduce the transform.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct WnetResult {
    /// Intercept α.
    pub intercept: f64,
    /// Time-domain functional coefficient β(t) (length `m` = curve length).
    pub beta_t: Vec<f64>,
    /// Fitted response values (length `n`).
    pub fitted_values: Vec<f64>,
    /// Residuals `y - ŷ` (length `n`).
    pub residuals: Vec<f64>,
    /// Coefficient-space functional coefficient (length `P` = total wavelet
    /// coefficients) — sparse (many exact zeros).
    pub coeff_weights: Vec<f64>,
    /// Indices (into `coeff_weights`) of the nonzero/selected coefficients.
    pub selected: Vec<usize>,
    /// Cross-validation-selected λ.
    pub lambda: f64,
    /// Elastic-net mixing parameter used (`config.alpha`).
    pub alpha: f64,
    /// Wavelet family used for the DWT (for reproducing the transform in prediction).
    pub family: WaveletFamily,
    /// Boundary mode used for the DWT.
    pub mode: BoundaryMode,
    /// Effective decomposition depth used.
    pub level: usize,
}

/// Soft-threshold operator `sign(z)·max(|z| - γ, 0)` (the L1 proximal step).
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// SHARED CD ENGINE. Per-coefficient elastic-net coordinate descent on the raw
/// wavelet-coefficient design.
///
/// Fits `min_β (1/2n)‖y - α - Xβ‖² + λ[α_mix‖β‖₁ + ½(1-α_mix)‖β‖²]` by cyclic
/// coordinate descent. For coordinate `j`, the update uses the partial residual
/// `r = y_centered - Σ_{k≠j} βₖ X_c,ₖ` (maintained via a running fitted vector for
/// O(nP)/sweep), the coordinate gradient `z_j = (X_c,ⱼ · r)/n`, then applies the
/// L1 soft-threshold with the L2-ridge denominator:
/// `βⱼ = soft(z_j, λ·α_mix) / (‖X_c,ⱼ‖²/n + λ(1-α_mix))`.
///
/// Columns are centered internally (so the penalty is scale-consistent across
/// coefficients only up to their own norm — we do NOT rescale to unit variance,
/// keeping the coefficient-space geometry faithful to the DWT). The intercept is
/// recovered as `mean(y) - Σ βⱼ·mean(Xⱼ)` on the un-centered column means.
///
/// Returns `(intercept, coeff_weights)` where `coeff_weights` has length `P`.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `y.len()` does not equal `design.nrows()`.
/// - [`FdarError::InvalidParameter`] if `alpha` is outside `[0, 1]`, `lambda` is
///   negative or non-finite, `tol` is negative or non-finite, or `max_iter == 0`.
pub(crate) fn elastic_net_cd(
    design: &FdMatrix,
    y: &[f64],
    lambda: f64,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(f64, Vec<f64>), FdarError> {
    let (n, p) = design.shape();
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements (== design rows)"),
            actual: format!("{} elements", y.len()),
        });
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in [0, 1], got {alpha}"),
        });
    }
    if lambda < 0.0 || !lambda.is_finite() {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("lambda must be finite and >= 0, got {lambda}"),
        });
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "tol",
            message: format!("tol must be finite and >= 0, got {tol}"),
        });
    }
    if max_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter",
            message: "max_iter must be >= 1".to_string(),
        });
    }

    let n_f = n as f64;
    let mu_y = y.iter().sum::<f64>() / n_f;
    let y_centered: Vec<f64> = y.iter().map(|&v| v - mu_y).collect();

    // Per-column means and centered columns; precompute ‖X_c,ⱼ‖²/n.
    let col_means: Vec<f64> = (0..p)
        .map(|j| design.column(j).iter().sum::<f64>() / n_f)
        .collect();
    let mut xc = vec![0.0_f64; n * p]; // column-major, n × p
    let mut col_norm_sq_over_n = vec![0.0_f64; p];
    for j in 0..p {
        let mu = col_means[j];
        let mut norm_sq = 0.0;
        let col = design.column(j);
        for i in 0..n {
            let v = col[i] - mu;
            xc[i + j * n] = v;
            norm_sq += v * v;
        }
        col_norm_sq_over_n[j] = norm_sq / n_f;
    }

    // Coefficients start at zero; the running fit tracks Σⱼ βⱼ X_c,ⱼ so a
    // coordinate's partial residual is (y_centered - fit + βⱼ X_c,ⱼ) in O(n).
    let mut beta = vec![0.0_f64; p];
    let mut fit = vec![0.0_f64; n]; // Σⱼ βⱼ X_c,ⱼ
    let l1 = lambda * alpha;
    let l2 = lambda * (1.0 - alpha);

    for _sweep in 0..max_iter {
        let mut max_delta = 0.0_f64;
        for j in 0..p {
            let denom = col_norm_sq_over_n[j] + l2;
            if denom <= 0.0 {
                // Dead column (zero-variance) with no ridge: leave at zero.
                if beta[j] != 0.0 {
                    let old = beta[j];
                    for i in 0..n {
                        fit[i] -= old * xc[i + j * n];
                    }
                    max_delta = max_delta.max(old.abs());
                    beta[j] = 0.0;
                }
                continue;
            }
            // z_j = (X_c,ⱼ · partial_residual)/n where
            // partial_residual = y_centered - (fit - βⱼ X_c,ⱼ).
            let old = beta[j];
            let mut dot = 0.0;
            for i in 0..n {
                let r = y_centered[i] - fit[i] + old * xc[i + j * n];
                dot += xc[i + j * n] * r;
            }
            let z = dot / n_f;
            let new = soft_threshold(z, l1) / denom;
            if new != old {
                let diff = new - old;
                for i in 0..n {
                    fit[i] += diff * xc[i + j * n];
                }
                max_delta = max_delta.max(diff.abs());
                beta[j] = new;
            }
        }
        if max_delta < tol {
            break;
        }
    }

    // Intercept on un-centered column means: mu_y - Σ βⱼ·mean(Xⱼ).
    let intercept = mu_y - (0..p).map(|j| beta[j] * col_means[j]).sum::<f64>();
    Ok((intercept, beta))
}

/// Build the geometric λ grid used by [`wnet_cv_lambda`].
///
/// If `config.lambda_grid` is `Some`, that grid is returned verbatim (validated
/// non-empty by the caller). Otherwise a log-spaced grid of `config.n_lambda`
/// values from `λ_max` down to `λ_max · ε` (ε = 1e-3) is built, where `λ_max` is
/// the smallest λ that zeroes every coefficient:
/// `λ_max = max_j |X_c,ⱼ · y_centered| / (n · max(α, tiny))`.
///
/// The grid is returned in descending order (largest/sparsest λ first) so ties in
/// CV-MSE naturally resolve toward the larger λ when scanned.
fn build_lambda_grid(design: &FdMatrix, y: &[f64], config: &WnetConfig) -> Vec<f64> {
    if let Some(grid) = &config.lambda_grid {
        return grid.clone();
    }
    let (n, p) = design.shape();
    let n_f = n as f64;
    let mu_y = y.iter().sum::<f64>() / n_f;
    let y_centered: Vec<f64> = y.iter().map(|&v| v - mu_y).collect();

    // λ_max = max_j |X_c,ⱼ · y_centered| / (n·α_eff).
    let alpha_eff = config.alpha.max(1e-3);
    let mut max_corr = 0.0_f64;
    for j in 0..p {
        let mu = design.column(j).iter().sum::<f64>() / n_f;
        let col = design.column(j);
        let dot: f64 = (0..n).map(|i| (col[i] - mu) * y_centered[i]).sum();
        max_corr = max_corr.max(dot.abs());
    }
    let lambda_max = (max_corr / (n_f * alpha_eff)).max(1e-8);

    let n_lambda = config.n_lambda.max(1);
    if n_lambda == 1 {
        return vec![lambda_max];
    }
    let eps = 1e-3_f64;
    let log_max = lambda_max.ln();
    let log_min = (lambda_max * eps).ln();
    let step = (log_max - log_min) / (n_lambda as f64 - 1.0);
    (0..n_lambda)
        .map(|k| (log_max - step * k as f64).exp())
        .collect()
}

/// SHARED CV HELPER. Deterministic K-fold cross-validated λ selection for `wnet`.
///
/// Builds the geometric λ grid (or uses `config.lambda_grid`), partitions the `n`
/// observations into `config.n_folds` folds via [`crate::cv::create_folds`] with
/// the FIXED `config.seed` (so the partition — and therefore the selected λ — is
/// identical across runs), computes CV-MSE per λ (fit [`elastic_net_cd`] on each
/// training set, score on the held-out fold), and returns the λ minimizing
/// CV-MSE. Ties (within a small epsilon) resolve toward the LARGER λ (sparser).
///
/// # Errors
/// - [`FdarError::InvalidParameter`] if `config.n_folds < 2`,
///   `config.n_folds > n`, `config.alpha` is outside `[0, 1]`, or an explicit
///   `lambda_grid` is empty.
/// - [`FdarError::InvalidDimension`] if `y.len()` does not equal `design.nrows()`.
pub(crate) fn wnet_cv_lambda(
    design: &FdMatrix,
    y: &[f64],
    config: &WnetConfig,
) -> Result<f64, FdarError> {
    let (n, _p) = design.shape();
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements (== design rows)"),
            actual: format!("{} elements", y.len()),
        });
    }
    if config.n_folds < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_folds",
            message: format!("n_folds must be >= 2, got {}", config.n_folds),
        });
    }
    if config.n_folds > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_folds",
            message: format!(
                "n_folds ({}) must not exceed the number of observations ({n})",
                config.n_folds
            ),
        });
    }
    if !(0.0..=1.0).contains(&config.alpha) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in [0, 1], got {}", config.alpha),
        });
    }
    if let Some(grid) = &config.lambda_grid {
        if grid.is_empty() {
            return Err(FdarError::InvalidParameter {
                parameter: "lambda_grid",
                message: "explicit lambda_grid must be non-empty".to_string(),
            });
        }
    }

    let grid = build_lambda_grid(design, y, config);
    let folds = crate::cv::create_folds(n, config.n_folds, config.seed);

    // Precompute per-fold train/test index sets (shared across all λ).
    let fold_sets: Vec<(Vec<usize>, Vec<usize>)> = (0..config.n_folds)
        .map(|f| crate::cv::fold_indices(&folds, f))
        .collect();

    let mut best_lambda = grid[0];
    let mut best_mse = f64::INFINITY;
    let tie_eps = 1e-12;

    for &lam in &grid {
        let mut total_sse = 0.0_f64;
        let mut scored = 0usize;
        for (train_idx, test_idx) in &fold_sets {
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }
            let train_data = crate::cv::subset_rows(design, train_idx);
            let train_y = crate::cv::subset_vec(y, train_idx);
            let (intercept, beta) = elastic_net_cd(
                &train_data,
                &train_y,
                lam,
                config.alpha,
                config.max_iter,
                config.tol,
            )?;
            for &oi in test_idx {
                let mut yhat = intercept;
                for j in 0..design.ncols() {
                    yhat += design[(oi, j)] * beta[j];
                }
                let e = y[oi] - yhat;
                total_sse += e * e;
                scored += 1;
            }
        }
        if scored == 0 {
            continue;
        }
        let mse = total_sse / scored as f64;
        // Grid is descending (largest λ first). Strictly-less keeps the FIRST
        // (larger) λ on a tie; the epsilon guards float noise so a marginally
        // smaller MSE at a smaller λ does not override a near-equal larger λ.
        if mse < best_mse - tie_eps {
            best_mse = mse;
            best_lambda = lam;
        }
    }

    Ok(best_lambda)
}

// ---------------------------------------------------------------------------
// wnet entry point
// ---------------------------------------------------------------------------

/// Fit the wavelet-domain elastic-net scalar-on-function regressor `wnet` (WAV-04).
///
/// Transforms every curve into its wavelet-coefficient vector (shared seam
/// [`curves_to_coeff_design`]), selects a deterministic cross-validated λ
/// ([`wnet_cv_lambda`]), refits the per-coefficient elastic-net
/// ([`elastic_net_cd`]) at that λ on the full data, and reconstructs the
/// time-domain functional coefficient β(t) via the inverse DWT (shared seam
/// [`coeff_weights_to_beta_t`]).
///
/// Because a sparse wavelet basis concentrates localized signal in a few
/// coefficients, the L1 penalty drives the rest to exactly zero — the nonzero
/// indices are reported in [`WnetResult::selected`].
///
/// # Arguments
/// * `data` — functional predictor matrix (n × m), one curve per row.
/// * `y` — scalar response (length n).
/// * `config` — DWT + elastic-net + CV configuration.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows, zero
///   columns, or `y.len() != n`.
/// - [`FdarError::InvalidParameter`] if `config.alpha ∉ [0, 1]`,
///   `config.n_folds < 2`, `config.n_folds > n`, `config.max_iter == 0`,
///   `config.tol` is negative or non-finite, an explicit `config.lambda_grid`
///   is empty, or the DWT rejects the family/level (surfaced from
///   [`decompose_matrix`]).
/// - [`FdarError::ComputationFailed`] if the underlying transform fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn wnet(data: &FdMatrix, y: &[f64], config: &WnetConfig) -> Result<WnetResult, FdarError> {
    let (n, m) = data.shape();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows (observations)".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column (evaluation point)".to_string(),
            actual: format!("{m} columns"),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements (== data rows)"),
            actual: format!("{} elements", y.len()),
        });
    }
    if !(0.0..=1.0).contains(&config.alpha) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in [0, 1], got {}", config.alpha),
        });
    }
    if config.n_folds < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_folds",
            message: format!("n_folds must be >= 2, got {}", config.n_folds),
        });
    }
    if config.n_folds > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_folds",
            message: format!(
                "n_folds ({}) must not exceed the number of observations ({n})",
                config.n_folds
            ),
        });
    }
    if config.max_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter",
            message: "max_iter must be >= 1".to_string(),
        });
    }
    if !config.tol.is_finite() || config.tol < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "tol",
            message: format!("tol must be finite and >= 0, got {}", config.tol),
        });
    }
    if let Some(grid) = &config.lambda_grid {
        if grid.is_empty() {
            return Err(FdarError::InvalidParameter {
                parameter: "lambda_grid",
                message: "explicit lambda_grid must be non-empty".to_string(),
            });
        }
    }

    // Curves -> coefficient design (shared seam). Surfaces DWT errors unchanged.
    let (design, layout) =
        curves_to_coeff_design(data, config.family.clone(), config.mode, config.level)?;

    // Deterministic CV-selected λ, then refit on the full data at that λ.
    let lambda = wnet_cv_lambda(&design, y, config)?;
    let (intercept, coeff_weights) = elastic_net_cd(
        &design,
        y,
        lambda,
        config.alpha,
        config.max_iter,
        config.tol,
    )?;

    // Selected (nonzero) coefficients.
    let selected: Vec<usize> = coeff_weights
        .iter()
        .enumerate()
        .filter(|(_, &b)| b != 0.0)
        .map(|(j, _)| j)
        .collect();

    // Fitted values via the plain coefficient-space dot: ŷ = intercept + X·β.
    let fitted_values = compute_fitted_affine(&design, &coeff_weights, intercept);
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();

    // β(t) via inverse DWT of the coefficient-space weights (shared seam).
    let beta_t = coeff_weights_to_beta_t(&coeff_weights, &layout)?;

    Ok(WnetResult {
        intercept,
        beta_t,
        fitted_values,
        residuals,
        coeff_weights,
        selected,
        lambda,
        alpha: config.alpha,
        family: config.family.clone(),
        mode: config.mode,
        level: layout.levels(),
    })
}

/// Compute fitted values `ŷ = intercept + X β` (affine coefficient-space dot).
fn compute_fitted_affine(design: &FdMatrix, coeffs: &[f64], intercept: f64) -> Vec<f64> {
    let (n, p) = design.shape();
    (0..n)
        .map(|i| {
            let mut yhat = intercept;
            for j in 0..p {
                yhat += design[(i, j)] * coeffs[j];
            }
            yhat
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    /// Deterministic pseudo-random value stream (LCG) — spans full rank, no dep.
    /// Mirrors the DWT module's own test helper so the design is truly full-rank.
    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                2.0 * u - 1.0
            })
            .collect()
    }

    /// Build a spanning, full-rank n×m predicate design from independent
    /// pseudo-random rows (n ≫ m). Returns the FdMatrix.
    fn spanning_design(n: usize, m: usize, seed0: u64) -> FdMatrix {
        let mut flat = vec![0.0_f64; n * m];
        for i in 0..n {
            let row = pseudo_random(m, seed0 + i as u64);
            for j in 0..m {
                flat[i + j * n] = row[j];
            }
        }
        FdMatrix::from_column_major(flat, n, m).unwrap()
    }

    fn rel_l2(recovered: &[f64], truth: &[f64]) -> f64 {
        let num: f64 = recovered
            .iter()
            .zip(truth)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let den: f64 = truth.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-300);
        num / den
    }

    /// Fit `wcr` with a given method on a spanning design where y is generated
    /// from a known coefficient-space β, and assert β(t) recovers the inverse-DWT
    /// of that β within a tight relative L2 tolerance.
    fn recovery_for_method(method: WcrMethod) {
        let (n, m) = (120usize, 32usize);
        let data = spanning_design(n, m, 1000);
        let family = WaveletFamily::Daubechies(4);
        let mode = BoundaryMode::Periodic;

        // Coefficient design + layout (the exact seam wcr uses internally).
        let (design, layout) = curves_to_coeff_design(&data, family.clone(), mode, None).unwrap();
        let p = design.ncols();

        // Known coefficient-space β and intercept; y is exact (no noise) so a
        // full-rank fit must recover β_coeff exactly.
        let beta_coeff = pseudo_random(p, 77);
        let true_intercept = 0.37_f64;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = true_intercept;
                for j in 0..p {
                    acc += design[(i, j)] * beta_coeff[j];
                }
                acc
            })
            .collect();

        // The true time-domain coefficient is the inverse DWT of β_coeff.
        let beta_t_true = coeff_weights_to_beta_t(&beta_coeff, &layout).unwrap();

        // Fit with enough components to span the coefficient design (min(n, P)).
        let config = WcrConfig {
            family,
            mode,
            level: None,
            ncomp: p.min(n),
            method,
            ..Default::default()
        };
        let fit = wcr(&data, &y, &config).unwrap();

        assert_eq!(fit.method, method);
        assert_eq!(fit.beta_t.len(), m);
        assert_eq!(fit.coeff_weights.len(), p);

        let e = rel_l2(&fit.beta_t, &beta_t_true);
        assert!(
            e < 1e-6,
            "{method:?}: beta_t recovery rel L2 err {e} exceeds tolerance on spanning full-rank design"
        );

        // Finite outputs.
        assert!(fit.beta_t.iter().all(|x| x.is_finite()));
        assert!(fit.fitted_values.iter().all(|x| x.is_finite()));
        assert!(fit.residuals.iter().all(|x| x.is_finite()));
        // With an exact (noiseless) full-rank fit, residuals ≈ 0.
        let max_resid = fit
            .residuals
            .iter()
            .fold(0.0_f64, |acc, &r| acc.max(r.abs()));
        assert!(
            max_resid < 1e-6,
            "{method:?}: residuals not ~0 ({max_resid})"
        );
    }

    #[test]
    fn wcr_pcr_recovers_known_beta_t_on_spanning_design() {
        recovery_for_method(WcrMethod::Pcr);
    }

    #[test]
    fn wcr_pls_recovers_known_beta_t_on_spanning_design() {
        recovery_for_method(WcrMethod::Pls);
    }

    #[test]
    fn wcr_default_config_is_db4_periodic_auto_pcr() {
        let c = WcrConfig::default();
        assert_eq!(c.family, WaveletFamily::Daubechies(4));
        assert_eq!(c.mode, BoundaryMode::Periodic);
        assert_eq!(c.level, None);
        assert_eq!(c.method, WcrMethod::Pcr);
        assert_eq!(WcrMethod::default(), WcrMethod::Pcr);
    }

    #[test]
    fn curves_to_coeff_design_layout_and_shape() {
        let (n, m) = (10usize, 48usize);
        let data = spanning_design(n, m, 500);
        let (design, layout) = curves_to_coeff_design(
            &data,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            None,
        )
        .unwrap();
        assert_eq!(design.nrows(), n);
        assert_eq!(design.ncols(), layout.total_len());
        assert_eq!(layout.signal_len, m);
        assert_eq!(layout.levels(), layout.detail_lens.len());
    }

    #[test]
    fn coeff_weights_to_beta_t_inverts_decompose() {
        // A round-trip sanity: reconstruct of a curve's own coefficients == curve.
        let (n, m) = (4usize, 48usize);
        let data = spanning_design(n, m, 900);
        let (design, layout) = curves_to_coeff_design(
            &data,
            WaveletFamily::Daubechies(6),
            BoundaryMode::Periodic,
            None,
        )
        .unwrap();
        let row0: Vec<f64> = (0..design.ncols()).map(|j| design[(0, j)]).collect();
        let recon = coeff_weights_to_beta_t(&row0, &layout).unwrap();
        let orig = data.row(0);
        assert!(rel_l2(&recon, &orig) < 1e-10);
    }

    #[test]
    fn coeff_weights_to_beta_t_rejects_wrong_length() {
        let (n, m) = (4usize, 48usize);
        let data = spanning_design(n, m, 901);
        let (_design, layout) =
            curves_to_coeff_design(&data, WaveletFamily::Haar, BoundaryMode::Periodic, None)
                .unwrap();
        let wrong = vec![0.0; layout.total_len() + 1];
        assert!(matches!(
            coeff_weights_to_beta_t(&wrong, &layout),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    // --- Validation gate (SC4) ---

    fn base_config() -> WcrConfig {
        WcrConfig {
            ncomp: 3,
            ..Default::default()
        }
    }

    #[test]
    fn wcr_rejects_too_few_rows() {
        let data = spanning_design(2, 48, 1);
        let y = vec![0.0, 1.0];
        assert!(matches!(
            wcr(&data, &y, &base_config()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn wcr_rejects_mismatched_y_len() {
        let data = spanning_design(10, 48, 2);
        let y = vec![0.0; 9];
        assert!(matches!(
            wcr(&data, &y, &base_config()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn wcr_rejects_zero_ncomp() {
        let data = spanning_design(10, 48, 3);
        let y = vec![0.0; 10];
        let config = WcrConfig {
            ncomp: 0,
            ..Default::default()
        };
        assert!(matches!(
            wcr(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wcr_surfaces_unsupported_family() {
        let data = spanning_design(10, 48, 4);
        let y = vec![0.0; 10];
        let config = WcrConfig {
            family: WaveletFamily::Daubechies(11),
            ..base_config()
        };
        assert!(matches!(
            wcr(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wcr_surfaces_level_out_of_range() {
        let data = spanning_design(10, 48, 5);
        let y = vec![0.0; 10];
        let config = WcrConfig {
            level: Some(999),
            ..base_config()
        };
        assert!(matches!(
            wcr(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wcr_finite_outputs_both_methods() {
        let (n, m) = (100usize, 40usize);
        let data = spanning_design(n, m, 4242);
        let y = pseudo_random(n, 8080);
        for method in [WcrMethod::Pcr, WcrMethod::Pls] {
            let config = WcrConfig {
                ncomp: 8,
                method,
                ..Default::default()
            };
            let fit = wcr(&data, &y, &config).unwrap();
            assert!(fit.intercept.is_finite());
            assert!(fit.beta_t.iter().all(|x| x.is_finite()));
            assert!(fit.fitted_values.iter().all(|x| x.is_finite()));
            assert!(fit.residuals.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn wcr_small_n_default_config_succeeds() {
        // CR-01 regression: default WcrConfig has ncomp = 5. With a small sample
        // (n = 4), the old clamp `ncomp.min(n).min(p)` gave ncomp = 4, making the
        // OLS design n × (n + 1) = 4 × 5 which ols_solve rejected (n < p). The fix
        // clamps to `n - 1`, so the design stays overdetermined and the fit succeeds.
        let (n, m) = (4usize, 32usize);
        let data = spanning_design(n, m, 2468);
        let y = pseudo_random(n, 1357);
        let config = WcrConfig::default(); // ncomp = 5 > n
        let fit = wcr(&data, &y, &config).unwrap();
        // Effective ncomp clamped to n - 1 (= 3), never n.
        assert!(fit.ncomp < n, "ncomp {} exceeds n - 1", fit.ncomp);
        assert_eq!(fit.beta_t.len(), m);
        assert!(fit.intercept.is_finite());
        assert!(fit.beta_t.iter().all(|x| x.is_finite()));
        assert!(fit.fitted_values.iter().all(|x| x.is_finite()));
        assert!(fit.residuals.iter().all(|x| x.is_finite()));

        // Also confirm the documented minimum n = 3 works under the default config.
        let data3 = spanning_design(3, m, 2469);
        let y3 = pseudo_random(3, 1358);
        let fit3 = wcr(&data3, &y3, &WcrConfig::default()).unwrap();
        assert!(fit3.ncomp <= 2);
        assert!(fit3.beta_t.iter().all(|x| x.is_finite()));
    }

    // ===================================================================
    // wnet — wavelet-domain elastic-net regressor (WAV-04)
    // ===================================================================

    /// Build a synthetic sparse coefficient-space β (a handful of nonzero
    /// coefficients, the rest exactly zero) plus the spanning full-rank design
    /// and layout. Returns `(data, design, layout, beta_coeff, support)`.
    fn sparse_wnet_problem(
        n: usize,
        m: usize,
        seed0: u64,
    ) -> (FdMatrix, FdMatrix, CoeffLayout, Vec<f64>, Vec<usize>) {
        let data = spanning_design(n, m, seed0);
        let family = WaveletFamily::Daubechies(4);
        let mode = BoundaryMode::Periodic;
        let (design, layout) = curves_to_coeff_design(&data, family, mode, None).unwrap();
        let p = design.ncols();

        // Localize β in a few coefficients spread across the bands.
        let support: Vec<usize> = vec![0, 2, p / 2, p - 3]
            .into_iter()
            .filter(|&j| j < p)
            .collect();
        let mut beta_coeff = vec![0.0_f64; p];
        // Give the true-support coefficients large, well-separated magnitudes so
        // they clearly dominate the elastic-net solution.
        let mags = [4.0, -3.5, 5.0, -4.5];
        for (k, &j) in support.iter().enumerate() {
            beta_coeff[j] = mags[k % mags.len()];
        }
        (data, design, layout, beta_coeff, support)
    }

    #[test]
    fn wnet_elastic_net_cd_recovers_sparse_support() {
        // Fixed-λ path (Task 1): a moderate λ produces a sparse solution whose
        // nonzero coefficients concentrate on the true support.
        let (n, m) = (256usize, 32usize);
        let (_data, design, _layout, beta_coeff, support) = sparse_wnet_problem(n, m, 3000);
        let p = design.ncols();

        // y = intercept + X β (noiseless) — the localized signal.
        let intercept_true = 0.5_f64;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = intercept_true;
                for j in 0..p {
                    acc += design[(i, j)] * beta_coeff[j];
                }
                acc
            })
            .collect();

        // Moderate λ, alpha=0.9 (strongly L1) → sparse.
        let (intercept, beta) = elastic_net_cd(&design, &y, 0.05, 0.9, 2000, 1e-8).unwrap();

        assert!(intercept.is_finite());
        assert!(beta.iter().all(|b| b.is_finite()));

        let selected: Vec<usize> = beta
            .iter()
            .enumerate()
            .filter(|(_, &b)| b.abs() > 1e-8)
            .map(|(j, _)| j)
            .collect();

        // True support is among the selected set.
        for &j in &support {
            assert!(
                selected.contains(&j),
                "true-support coeff {j} not selected (selected={selected:?})"
            );
        }
        // The selected set is meaningfully sparse relative to P.
        assert!(
            selected.len() < p / 2,
            "selection not sparse: |selected|={} of P={p}",
            selected.len()
        );
    }

    #[test]
    fn wnet_fixed_lambda_end_to_end_finite() {
        // Task 1 tracer: full wnet path (with CV under the hood) yields finite
        // β(t)/fitted/coeff outputs on the localized-signal problem.
        let (n, m) = (200usize, 32usize);
        let (data, design, _layout, beta_coeff, _support) = sparse_wnet_problem(n, m, 3100);
        let p = design.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.25;
                for j in 0..p {
                    acc += design[(i, j)] * beta_coeff[j];
                }
                acc
            })
            .collect();

        let config = WnetConfig {
            n_lambda: 15,
            n_folds: 4,
            ..Default::default()
        };
        let fit = wnet(&data, &y, &config).unwrap();
        assert_eq!(fit.beta_t.len(), m);
        assert_eq!(fit.coeff_weights.len(), p);
        assert!(fit.intercept.is_finite());
        assert!(fit.beta_t.iter().all(|x| x.is_finite()));
        assert!(fit.fitted_values.iter().all(|x| x.is_finite()));
        assert!(fit.residuals.iter().all(|x| x.is_finite()));
        assert!(fit.coeff_weights.iter().all(|x| x.is_finite()));
        // selected indices match the nonzero coeff_weights.
        for &j in &fit.selected {
            assert!(fit.coeff_weights[j] != 0.0);
        }
    }

    #[test]
    fn wnet_default_config_is_db4_periodic_auto() {
        let c = WnetConfig::default();
        assert_eq!(c.family, WaveletFamily::Daubechies(4));
        assert_eq!(c.mode, BoundaryMode::Periodic);
        assert_eq!(c.level, None);
        assert!((c.alpha - 0.5).abs() < 1e-15);
        assert_eq!(c.lambda_grid, None);
        assert_eq!(c.n_lambda, 50);
        assert_eq!(c.n_folds, 5);
        assert_eq!(c.seed, 0);
    }

    // --- Deterministic CV-λ (SC3) ---

    #[test]
    fn wnet_cv_lambda_is_deterministic_across_runs() {
        let (n, m) = (200usize, 32usize);
        let (data, design, _layout, beta_coeff, _support) = sparse_wnet_problem(n, m, 3200);
        let p = design.ncols();
        // Add mild noise so CV-MSE is non-degenerate but λ still well-defined.
        let noise = pseudo_random(n, 9999);
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.1;
                for j in 0..p {
                    acc += design[(i, j)] * beta_coeff[j];
                }
                acc + 0.05 * noise[i]
            })
            .collect();

        let config = WnetConfig {
            alpha: 0.8,
            n_lambda: 20,
            n_folds: 5,
            seed: 0,
            ..Default::default()
        };
        let fit1 = wnet(&data, &y, &config).unwrap();
        let fit2 = wnet(&data, &y, &config).unwrap();
        assert_eq!(
            fit1.lambda, fit2.lambda,
            "CV-selected lambda differs across runs: {} vs {}",
            fit1.lambda, fit2.lambda
        );
        // Also exercise the helper directly.
        let l1 = wnet_cv_lambda(&design, &y, &config).unwrap();
        let l2 = wnet_cv_lambda(&design, &y, &config).unwrap();
        assert_eq!(l1, l2);
    }

    // --- β(t) recovery on SNR data (SC3) ---

    #[test]
    fn wnet_recovers_beta_t_on_snr_data() {
        // Spanning full-rank design, moderate SNR: the fit at the CV λ must be
        // non-degenerate and β(t) must track the injected β(t).
        let (n, m) = (300usize, 32usize);
        let (data, design, layout, beta_coeff, _support) = sparse_wnet_problem(n, m, 3300);
        let p = design.ncols();
        let beta_t_true = coeff_weights_to_beta_t(&beta_coeff, &layout).unwrap();

        // Signal variance vs noise: pick noise small relative to signal spread.
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += design[(i, j)] * beta_coeff[j];
                }
                acc
            })
            .collect();
        let sig_sd = {
            let mean = signal.iter().sum::<f64>() / n as f64;
            (signal.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
        };
        let noise = pseudo_random(n, 4141);
        let noise_scale = 0.05 * sig_sd; // ~20:1 SNR
        let y: Vec<f64> = (0..n)
            .map(|i| 0.3 + signal[i] + noise_scale * noise[i])
            .collect();

        let config = WnetConfig {
            alpha: 0.7,
            n_lambda: 30,
            n_folds: 5,
            ..Default::default()
        };
        let fit = wnet(&data, &y, &config).unwrap();

        // Non-degenerate: not all-zero.
        let nonzero = fit.coeff_weights.iter().filter(|&&b| b != 0.0).count();
        assert!(nonzero > 0, "degenerate all-zero fit at CV lambda");

        // β(t) tracks the injected β(t) within tolerance.
        let e = rel_l2(&fit.beta_t, &beta_t_true);
        assert!(
            e < 0.35,
            "wnet beta_t recovery rel L2 err {e} exceeds tolerance on SNR data"
        );
        assert!(fit.beta_t.iter().all(|x| x.is_finite()));
        assert!(fit.fitted_values.iter().all(|x| x.is_finite()));
    }

    // --- Validation gate (SC4) ---

    fn base_wnet_config() -> WnetConfig {
        WnetConfig {
            n_lambda: 10,
            n_folds: 3,
            ..Default::default()
        }
    }

    #[test]
    fn wnet_rejects_too_few_rows() {
        let data = spanning_design(2, 32, 10);
        let y = vec![0.0, 1.0];
        assert!(matches!(
            wnet(&data, &y, &base_wnet_config()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn wnet_rejects_zero_cols() {
        // An empty-column matrix is rejected before any DWT.
        let data = FdMatrix::zeros(5, 0);
        let y = vec![0.0; 5];
        assert!(matches!(
            wnet(&data, &y, &base_wnet_config()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn wnet_rejects_mismatched_y_len() {
        let data = spanning_design(10, 32, 11);
        let y = vec![0.0; 9];
        assert!(matches!(
            wnet(&data, &y, &base_wnet_config()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn wnet_rejects_alpha_out_of_range() {
        let data = spanning_design(10, 32, 12);
        let y = vec![0.0; 10];
        let config = WnetConfig {
            alpha: 1.5,
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
        let config = WnetConfig {
            alpha: -0.1,
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_rejects_too_few_folds() {
        let data = spanning_design(10, 32, 13);
        let y = vec![0.0; 10];
        let config = WnetConfig {
            n_folds: 1,
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_rejects_too_many_folds() {
        // WR-01: n_folds > n must be rejected rather than silently running fewer folds.
        let data = spanning_design(10, 32, 130);
        let y = vec![0.0; 10];
        let config = WnetConfig {
            n_folds: 11,
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
        // The shared CV helper also rejects it directly.
        let (design, _layout) = curves_to_coeff_design(
            &data,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            None,
        )
        .unwrap();
        assert!(matches!(
            wnet_cv_lambda(&design, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_rejects_negative_or_nan_tol() {
        // WR-02: negative or NaN tol is rejected (both via the entry and the CD engine).
        let data = spanning_design(10, 32, 131);
        let y = pseudo_random(10, 5);
        for bad in [-1e-6_f64, f64::NAN] {
            let config = WnetConfig {
                tol: bad,
                ..base_wnet_config()
            };
            assert!(matches!(
                wnet(&data, &y, &config),
                Err(FdarError::InvalidParameter { .. })
            ));
        }
        // elastic_net_cd rejects it directly too.
        let (design, _layout) = curves_to_coeff_design(
            &data,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            None,
        )
        .unwrap();
        assert!(matches!(
            elastic_net_cd(&design, &y, 0.1, 0.5, 100, -1.0),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            elastic_net_cd(&design, &y, 0.1, 0.5, 100, f64::NAN),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_rejects_zero_max_iter() {
        // WR-03: max_iter == 0 would silently return an all-zero-coefficient model.
        let data = spanning_design(10, 32, 132);
        let y = pseudo_random(10, 6);
        let config = WnetConfig {
            max_iter: 0,
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
        // elastic_net_cd rejects it directly too.
        let (design, _layout) = curves_to_coeff_design(
            &data,
            WaveletFamily::Daubechies(4),
            BoundaryMode::Periodic,
            None,
        )
        .unwrap();
        assert!(matches!(
            elastic_net_cd(&design, &y, 0.1, 0.5, 0, 1e-6),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_rejects_empty_lambda_grid() {
        let data = spanning_design(10, 32, 14);
        let y = vec![0.0; 10];
        let config = WnetConfig {
            lambda_grid: Some(vec![]),
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_surfaces_unsupported_family() {
        let data = spanning_design(10, 32, 15);
        let y = vec![0.0; 10];
        let config = WnetConfig {
            family: WaveletFamily::Daubechies(11),
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_surfaces_level_out_of_range() {
        let data = spanning_design(10, 32, 16);
        let y = vec![0.0; 10];
        let config = WnetConfig {
            level: Some(999),
            ..base_wnet_config()
        };
        assert!(matches!(
            wnet(&data, &y, &config),
            Err(FdarError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn wnet_finite_outputs_on_larger_snr_design() {
        let (n, m) = (256usize, 48usize);
        let (data, design, _layout, beta_coeff, _support) = sparse_wnet_problem(n, m, 3400);
        let p = design.ncols();
        let noise = pseudo_random(n, 2727);
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.2;
                for j in 0..p {
                    acc += design[(i, j)] * beta_coeff[j];
                }
                acc + 0.1 * noise[i]
            })
            .collect();

        let config = WnetConfig {
            alpha: 0.6,
            n_lambda: 25,
            n_folds: 5,
            ..Default::default()
        };
        let fit = wnet(&data, &y, &config).unwrap();
        assert!(fit.intercept.is_finite());
        assert!(fit.lambda.is_finite());
        assert!(fit.beta_t.iter().all(|x| x.is_finite()));
        assert!(fit.fitted_values.iter().all(|x| x.is_finite()));
        assert!(fit.residuals.iter().all(|x| x.is_finite()));
        assert!(fit.coeff_weights.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn wnet_explicit_lambda_grid_is_used() {
        // With a single-λ explicit grid, the CV selection must return that λ.
        let (n, m) = (120usize, 32usize);
        let (data, design, _layout, beta_coeff, _support) = sparse_wnet_problem(n, m, 3500);
        let p = design.ncols();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += design[(i, j)] * beta_coeff[j];
                }
                acc
            })
            .collect();
        let config = WnetConfig {
            lambda_grid: Some(vec![0.123]),
            ..base_wnet_config()
        };
        let fit = wnet(&data, &y, &config).unwrap();
        assert!((fit.lambda - 0.123).abs() < 1e-15);
    }
}
