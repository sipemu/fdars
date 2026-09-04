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

    // Clamp effective ncomp to the fittable rank, mirroring fregre_lm / fregre_pls.
    let ncomp = config.ncomp.min(n).min(p);

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
}
