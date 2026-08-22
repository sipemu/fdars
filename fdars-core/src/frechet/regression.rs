//! Global and local Fréchet regression over Euclidean predictors.
//!
//! Both routines predict a conditional density response (in 2-Wasserstein space)
//! at new predictor values. Global regression uses the Petersen–Müller global
//! linear weight scheme; local regression uses local-linear Gaussian-kernel
//! weights. Both feed **signed** weights into
//! [`crate::frechet::space::signed_quantile_average`] (sort-based isotonic
//! projection) — never [`crate::density_fda::wasserstein_barycenter`], which
//! rejects negative weights.

use super::{FrechetGlobalRegResult, FrechetLocalRegResult};
use crate::error::FdarError;
use crate::frechet::space::signed_quantile_average;
use crate::helpers::{gaussian_kernel, NUMERICAL_EPS};
use crate::linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve};
use crate::matrix::FdMatrix;

/// Validate the shared predictor/response/grid/xout shapes; returns `(n, p, m)`.
fn validate_reg_input(
    predictors: &FdMatrix,
    responses: &FdMatrix,
    argvals: &[f64],
    xout: &FdMatrix,
) -> Result<(usize, usize, usize), FdarError> {
    let (n, p) = predictors.shape();
    let (nr, m) = responses.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: "at least 1 observation".to_string(),
            actual: "0 rows".to_string(),
        });
    }
    if nr != n {
        return Err(FdarError::InvalidDimension {
            parameter: "responses",
            expected: format!("{n} rows (matching predictors)"),
            actual: format!("{nr} rows"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching responses columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be strictly increasing".to_string(),
        });
    }
    if xout.ncols() != p {
        return Err(FdarError::InvalidDimension {
            parameter: "xout",
            expected: format!("{p} columns (matching predictors)"),
            actual: format!("{} columns", xout.ncols()),
        });
    }
    Ok((n, p, m))
}

/// Global Fréchet regression with Euclidean predictors (Petersen & Müller 2019).
///
/// Predicts the conditional Fréchet-mean density response at each `xout` row using
/// the global linear weight scheme `sᵢ(x) = 1 + (Xᵢ − X̄)ᵀ Σ̂⁻¹ (x − X̄)` and the
/// signed weighted quantile average.
///
/// This function uses [`crate::frechet::space::signed_quantile_average`]
/// (sort-based isotonic projection), **NOT**
/// [`crate::density_fda::wasserstein_barycenter`], because the Petersen–Müller
/// weights `sᵢ(x)` can be **negative** (RESEARCH KEY RISK). Where R's
/// `frechet::GloWassReg` enforces a monotone predicted quantile via an `osqp`
/// quadratic program, this uses the zero-dependency sort-based monotone
/// projection.
///
/// # Errors
/// Returns [`FdarError`] for mismatched predictor/response/xout dimensions,
/// non-monotone `argvals`, a singular predictor covariance (after a 1e-6 ridge),
/// or a degenerate (zero-range) predicted quantile at an extreme extrapolation
/// point.
#[must_use = "expensive regression — store or use the returned prediction"]
pub fn frechet_global_reg(
    predictors: &FdMatrix,
    responses: &FdMatrix,
    argvals: &[f64],
    xout: &FdMatrix,
) -> Result<FrechetGlobalRegResult, FdarError> {
    let (n, p, m) = validate_reg_input(predictors, responses, argvals, xout)?;
    let n_out = xout.nrows();
    let n_q = m.max(101);

    // Predictor column means X̄.
    let mut x_bar = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += predictors[(i, j)];
        }
        x_bar[j] = s / n as f64;
    }

    // Sample covariance Σ̂ (p×p row-major), divide by n-1 (guard n==1), + ridge.
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let mut sigma = vec![0.0; p * p];
    for i in 0..n {
        for a in 0..p {
            let da = predictors[(i, a)] - x_bar[a];
            for b in 0..p {
                let db = predictors[(i, b)] - x_bar[b];
                sigma[a * p + b] += da * db;
            }
        }
    }
    for v in sigma.iter_mut() {
        *v /= denom;
    }
    for j in 0..p {
        sigma[j * p + j] += 1e-6;
    }
    let chol = cholesky_factor(&sigma, p)?;

    let mut predicted = FdMatrix::zeros(n_out, m);
    for r in 0..n_out {
        let diff_x: Vec<f64> = (0..p).map(|j| xout[(r, j)] - x_bar[j]).collect();
        let v = cholesky_forward_back(&chol, &diff_x, p); // Σ̂⁻¹ diff_x
                                                          // Signed weights sᵢ/n (Petersen–Müller: gx = colMeans(qin·s)).
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..p {
                dot += (predictors[(i, j)] - x_bar[j]) * v[j];
            }
            weights[i] = (1.0 + dot) / n as f64;
        }
        let dens = signed_quantile_average(responses, argvals, &weights, n_q)?;
        for j in 0..m {
            predicted[(r, j)] = dens[j];
        }
    }

    Ok(FrechetGlobalRegResult {
        predicted,
        xout: xout.clone(),
        x_bar,
    })
}

/// Local (local-linear, Gaussian-kernel-weighted) Fréchet regression
/// (Petersen & Müller 2019, `frechet::LocWassReg`).
///
/// Uses a product Gaussian kernel over the `p` predictor dimensions (single
/// bandwidth) and the Fan–Gijbels local-linear correction
/// `sᵢ = Kᵢ (1 − (Xᵢ − x₀)ᵀ μ₂⁻¹ μ₁)`.
///
/// Like [`frechet_global_reg`], this feeds signed weights into
/// [`crate::frechet::space::signed_quantile_average`], **NOT**
/// [`crate::density_fda::wasserstein_barycenter`], because the local-linear
/// correction weights can be **negative** (RESEARCH KEY RISK); the monotone
/// projection is the sort-based zero-dependency alternative to R's `osqp` QP.
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] for a non-positive/non-finite
/// `bandwidth`, plus the same shape/degeneracy errors as [`frechet_global_reg`].
#[must_use = "expensive regression — store or use the returned prediction"]
pub fn frechet_local_reg(
    predictors: &FdMatrix,
    responses: &FdMatrix,
    argvals: &[f64],
    xout: &FdMatrix,
    bandwidth: f64,
) -> Result<FrechetLocalRegResult, FdarError> {
    let (n, p, m) = validate_reg_input(predictors, responses, argvals, xout)?;
    if bandwidth <= 0.0 || !bandwidth.is_finite() {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!("bandwidth must be positive and finite, got {bandwidth}"),
        });
    }
    let n_out = xout.nrows();
    let n_q = m.max(101);

    let mut predicted = FdMatrix::zeros(n_out, m);
    for r in 0..n_out {
        let x0: Vec<f64> = (0..p).map(|j| xout[(r, j)]).collect();

        // Product Gaussian kernel weights.
        let mut kern = vec![0.0; n];
        for i in 0..n {
            let mut k = 1.0;
            for j in 0..p {
                k *= gaussian_kernel(predictors[(i, j)] - x0[j], bandwidth);
            }
            kern[i] = k;
        }

        // Local moments mu1 (p) and mu2 (p×p), scaled by 1/n.
        let mut mu1 = vec![0.0; p];
        let mut mu2 = vec![0.0; p * p];
        for i in 0..n {
            let ki = kern[i];
            for a in 0..p {
                let da = predictors[(i, a)] - x0[a];
                mu1[a] += ki * da;
                for b in 0..p {
                    let db = predictors[(i, b)] - x0[b];
                    mu2[a * p + b] += ki * da * db;
                }
            }
        }
        for v in mu1.iter_mut() {
            *v /= n as f64;
        }
        for v in mu2.iter_mut() {
            *v /= n as f64;
        }
        for j in 0..p {
            mu2[j * p + j] += 1e-6;
        }
        let a_vec = cholesky_solve(&mu2, &mu1, p)?; // μ₂⁻¹ μ₁

        // Local-linear signed weights sᵢ = Kᵢ (1 − (Xᵢ − x₀)ᵀ a).
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let mut corr = 0.0;
            for j in 0..p {
                corr += (predictors[(i, j)] - x0[j]) * a_vec[j];
            }
            weights[i] = kern[i] * (1.0 - corr);
        }
        let sum_w: f64 = weights.iter().sum();
        if sum_w.abs() < NUMERICAL_EPS {
            return Err(FdarError::ComputationFailed {
                operation: "frechet_local_reg",
                detail: "local weights sum to zero (bandwidth too small or no nearby points)"
                    .to_string(),
            });
        }
        for w in weights.iter_mut() {
            *w /= sum_w;
        }
        let dens = signed_quantile_average(responses, argvals, &weights, n_q)?;
        for j in 0..m {
            predicted[(r, j)] = dens[j];
        }
    }

    Ok(FrechetLocalRegResult {
        predicted,
        xout: xout.clone(),
        bandwidth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frechet::space::wasserstein2_distance;
    use crate::helpers::trapz;

    fn uniform_grid(m: usize, lb: f64, ub: f64) -> Vec<f64> {
        (0..m)
            .map(|j| lb + (ub - lb) * j as f64 / (m - 1) as f64)
            .collect()
    }

    fn truncated_gaussian(argvals: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
        let raw: Vec<f64> = argvals
            .iter()
            .map(|&x| (-(x - mu).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let integral = trapz(&raw, argvals);
        raw.iter().map(|&d| d / integral).collect()
    }

    /// n scalar predictors on a deterministic grid over [-1.5, 1.5], response
    /// density = N(xᵢ, sigma) on a grid [-6, 6] wide enough that the densities
    /// adequately fill the support (narrow densities on a wide grid produce
    /// quantile-tail artifacts in the density round-trip). Returns
    /// (predictors n×1, responses n×m, argvals).
    fn synthetic(n: usize, m: usize, sigma: f64) -> (FdMatrix, FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m, -6.0, 6.0);
        let mut predictors = FdMatrix::zeros(n, 1);
        let mut responses = FdMatrix::zeros(n, m);
        for i in 0..n {
            let xi = -1.5 + 3.0 * i as f64 / (n - 1) as f64;
            predictors[(i, 0)] = xi;
            let dens = truncated_gaussian(&argvals, xi, sigma);
            for j in 0..m {
                responses[(i, j)] = dens[j];
            }
        }
        (predictors, responses, argvals)
    }

    fn is_monotone_nondecreasing_cdf(density: &[f64], argvals: &[f64]) -> bool {
        // A valid density is non-negative and integrates to ~1.
        density.iter().all(|&v| v >= -1e-9) && (trapz(density, argvals) - 1.0).abs() < 1e-6
    }

    #[test]
    fn global_tracks_known_relationship() {
        let (predictors, responses, argvals) = synthetic(31, 101, 1.0);
        let mut xout = FdMatrix::zeros(1, 1);
        xout[(0, 0)] = 0.5;
        let res = frechet_global_reg(&predictors, &responses, &argvals, &xout).unwrap();
        assert_eq!(res.predicted.shape(), (1, 101));
        let pred: Vec<f64> = (0..101).map(|j| res.predicted[(0, j)]).collect();
        let truth = truncated_gaussian(&argvals, 0.5, 1.0);
        // Barycenter density round-trip has an inherent ~0.15 W2 floor.
        let w2 = wasserstein2_distance(&pred, &truth, &argvals).unwrap();
        assert!(w2 < 0.25, "w2 = {w2}");
    }

    #[test]
    fn global_accepts_negative_weights() {
        // Extrapolating well beyond the predictor range forces some sᵢ negative.
        let (predictors, responses, argvals) = synthetic(31, 101, 1.0);
        let mut xout = FdMatrix::zeros(1, 1);
        xout[(0, 0)] = 4.0; // far outside [-1.5, 1.5]
        let res = frechet_global_reg(&predictors, &responses, &argvals, &xout).unwrap();
        let pred: Vec<f64> = (0..101).map(|j| res.predicted[(0, j)]).collect();
        assert!(is_monotone_nondecreasing_cdf(&pred, &argvals));
    }

    #[test]
    fn global_rejects_bad_input() {
        let (predictors, responses, argvals) = synthetic(10, 40, 1.0);
        let xout = {
            let mut x = FdMatrix::zeros(1, 1);
            x[(0, 0)] = 0.0;
            x
        };
        // response count mismatch
        let bad_resp = FdMatrix::zeros(9, 40);
        assert!(matches!(
            frechet_global_reg(&predictors, &bad_resp, &argvals, &xout).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
        // non-increasing argvals
        let mut bad_arg = argvals.clone();
        bad_arg[1] = bad_arg[0];
        assert!(matches!(
            frechet_global_reg(&predictors, &responses, &bad_arg, &xout).unwrap_err(),
            FdarError::InvalidParameter { parameter, .. } if parameter == "argvals"
        ));
    }

    #[test]
    fn local_tracks_known_relationship() {
        let (predictors, responses, argvals) = synthetic(31, 101, 1.0);
        let mut xout = FdMatrix::zeros(1, 1);
        xout[(0, 0)] = 0.0;
        let res = frechet_local_reg(&predictors, &responses, &argvals, &xout, 0.6).unwrap();
        assert_eq!(res.predicted.shape(), (1, 101));
        let pred: Vec<f64> = (0..101).map(|j| res.predicted[(0, j)]).collect();
        let truth = truncated_gaussian(&argvals, 0.0, 1.0);
        let w2 = wasserstein2_distance(&pred, &truth, &argvals).unwrap();
        assert!(w2 < 0.25, "w2 = {w2}");
    }

    #[test]
    fn local_accepts_negative_weights() {
        // Local-linear correction produces negative weights for asymmetric neighborhoods.
        let (predictors, responses, argvals) = synthetic(31, 101, 1.0);
        let mut xout = FdMatrix::zeros(1, 1);
        xout[(0, 0)] = 1.2; // near the edge → asymmetric kernel neighborhood
        let res = frechet_local_reg(&predictors, &responses, &argvals, &xout, 0.5).unwrap();
        let pred: Vec<f64> = (0..101).map(|j| res.predicted[(0, j)]).collect();
        assert!(is_monotone_nondecreasing_cdf(&pred, &argvals));
    }

    #[test]
    fn local_rejects_bad_bandwidth() {
        let (predictors, responses, argvals) = synthetic(20, 40, 1.0);
        let mut xout = FdMatrix::zeros(1, 1);
        xout[(0, 0)] = 0.0;
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                frechet_local_reg(&predictors, &responses, &argvals, &xout, bad).unwrap_err(),
                FdarError::InvalidParameter { parameter, .. } if parameter == "bandwidth"
            ));
        }
    }
}
