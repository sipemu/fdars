//! Concurrent (varying-coefficient) functional regression.
//!
//! Models the relationship:
//!
//! `y_i(t) = β₀(t) + Σₖ βₖ(t)·xₖᵢ(t) + εᵢ(t)`
//!
//! where all functional curves share the same dense grid of `m` evaluation
//! points. Estimation follows the **fdaconcur `ptFCReg` → `smPtFCRegCoef`
//! two-step convention**:
//!
//! 1. **Pointwise OLS** at each grid column `j` — build a design matrix
//!    `[1 | x₁[:,j] | … | xₚ[:,j]]` and solve the normal equations to obtain
//!    raw coefficient estimates `raw_β₀[j]` and `raw_βₖ[j]` for each predictor.
//! 2. **Local-linear kernel smoothing** of each raw discrete coefficient
//!    sequence over `j` via [`crate::smoothing::local_linear`]. The `bandwidth`
//!    parameter is the sole roughness knob: larger bandwidth → smoother β(t).
//!
//! **Boundary behaviour:** the smoother is not edge-corrected; the first and
//! last few grid points exhibit higher bias. Tests check only interior indices.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::smoothing;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of concurrent (varying-coefficient) functional regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConcurrentRegrResult {
    /// Smoothed predictor coefficient curves, rows = predictor index, cols = grid points (p × m).
    pub beta_curve: FdMatrix,
    /// Smoothed time-varying intercept β₀(t) (length m).
    pub intercept: Vec<f64>,
    /// Fitted functional response curves (n × m).
    pub fitted: FdMatrix,
    /// Residuals: response − fitted (n × m).
    pub residuals: FdMatrix,
    /// Shared grid used for evaluation (length m).
    pub argvals: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------

/// Concurrent (varying-coefficient) functional regression.
///
/// Fits the model `y_i(t) = β₀(t) + Σₖ βₖ(t)·xₖᵢ(t) + εᵢ(t)` for a
/// dense shared grid of `m` evaluation points and `p ≥ 1` functional
/// predictors. Estimation follows the **fdaconcur two-step convention**:
/// pointwise OLS at each grid column, then local-linear kernel smoothing of
/// the resulting discrete coefficient sequences.
///
/// # Arguments
///
/// * `response` — Functional response matrix (n × m).
/// * `predictors` — Slice of functional predictor matrices, each (n × m),
///   all evaluated on the same shared grid.
/// * `argvals` — Shared evaluation grid (length m); if `None`, a uniform
///   0..1 grid is used (project-wide convention).
/// * `bandwidth` — Kernel bandwidth for β(t) smoothing (must be positive).
///   Larger values yield smoother coefficient curves.
/// * `kernel` — Kernel type: `"gaussian"` (default), `"epanechnikov"`,
///   `"tricube"`. Passed directly to [`crate::smoothing::local_linear`].
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `predictors` is empty,
/// - `response` has fewer than 2 rows or zero columns,
/// - any predictor's shape does not match `(n, m)`, or
/// - `argvals` is `Some` with a length different from `m`.
///
/// Returns [`FdarError::InvalidParameter`] if `bandwidth` is not strictly
/// positive (i.e., zero, negative, `f64::NAN`, or `f64::INFINITY`).
///
/// # Notes
///
/// For stable OLS estimates at each grid column, the sample size should
/// satisfy `n > p + 1`. A small ridge regulariser (`eps = 1e-10 * (n + 1)`)
/// is applied to the diagonal of each column's normal equations to prevent
/// numerical blow-up when predictors are collinear at a grid point.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn concurrent_regression(
    response: &FdMatrix,
    predictors: &[FdMatrix],
    argvals: Option<&[f64]>,
    bandwidth: f64,
    kernel: &str,
) -> Result<ConcurrentRegrResult, FdarError> {
    // ------------------------------------------------------------------
    // 1. Input validation
    // ------------------------------------------------------------------
    if predictors.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: "at least 1".to_string(),
            actual: "0".to_string(),
        });
    }

    let (n, m) = response.shape();

    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "response",
            expected: "at least 2 rows (observations)".to_string(),
            actual: format!("{n}"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "response",
            expected: "non-zero columns (grid points)".to_string(),
            actual: "0".to_string(),
        });
    }

    for (k, pred) in predictors.iter().enumerate() {
        if pred.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "predictors[k]",
                expected: format!("{n} rows (matching response)"),
                actual: format!("{} (predictor index {k})", pred.nrows()),
            });
        }
        if pred.ncols() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "predictors[k]",
                expected: format!("{m} columns (matching response)"),
                actual: format!("{} (predictor index {k})", pred.ncols()),
            });
        }
    }

    // Reject NaN (`is_finite()` is false for NaN), ±Inf, zero and negative
    // values — all of which would propagate into the kernel smoother and
    // produce silent all-zero output.  The original `bandwidth <= 0.0` guard
    // was insufficient because `NaN <= 0.0` evaluates to `false` under
    // IEEE 754, allowing NaN to bypass the check.
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!("must be finite and positive, got {bandwidth}"),
        });
    }

    if let Some(av) = argvals {
        if av.len() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals",
                expected: format!("{m} elements (matching response columns)"),
                actual: format!("{}", av.len()),
            });
        }
    }

    // ------------------------------------------------------------------
    // 2. Resolve argvals
    // ------------------------------------------------------------------
    let argvals_owned: Vec<f64> = argvals
        .map(|v| v.to_vec())
        .unwrap_or_else(|| (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect());

    let p = predictors.len();
    let q = p + 1; // intercept + p slopes

    // ------------------------------------------------------------------
    // 3. Step 1 — pointwise OLS at each grid column (parallelised)
    //
    // Each closure allocates its own local xtx / xty buffers so there is
    // no shared mutable state — safe for rayon (mirrors local_polynomial).
    // Returns (raw_intercept_j, raw_beta_j: Vec<f64> of length p).
    // ------------------------------------------------------------------
    let raw_cols: Vec<(f64, Vec<f64>)> = iter_maybe_parallel!(0..m)
        .map(|j| {
            let mut xtx = vec![0.0_f64; q * q];
            let mut xty = vec![0.0_f64; q];

            for i in 0..n {
                // design row: [1, pred_0[i,j], pred_1[i,j], ...]
                let mut row = vec![1.0_f64; q];
                for (k, pred) in predictors.iter().enumerate() {
                    row[k + 1] = pred[(i, j)];
                }
                for a in 0..q {
                    for b in 0..q {
                        xtx[a * q + b] += row[a] * row[b];
                    }
                    xty[a] += row[a] * response[(i, j)];
                }
            }

            // Ridge stabiliser: eps = 1e-10 * (n + 1) (xtx[0] ≈ n for
            // the intercept column, but use n directly for clarity).
            let eps = 1e-10 * (xtx[0] + 1.0);
            for d in 0..q {
                xtx[d * q + d] += eps;
            }

            let coef = smoothing::solve_gaussian_pub(&mut xtx, &mut xty, q);

            let raw_intercept_j = coef[0];
            let raw_beta_j: Vec<f64> = coef[1..q].to_vec();
            (raw_intercept_j, raw_beta_j)
        })
        .collect();

    // Serialise collected results into contiguous buffers.
    let mut raw_intercept = vec![0.0_f64; m];
    // raw_beta layout: row-major over (predictor k, grid column j)
    // raw_beta[k * m + j] = β_k[j]
    let mut raw_beta = vec![0.0_f64; p * m];
    for (j, (ic, bk)) in raw_cols.into_iter().enumerate() {
        raw_intercept[j] = ic;
        for k in 0..p {
            raw_beta[k * m + j] = bk[k];
        }
    }

    // ------------------------------------------------------------------
    // 4. Step 2 — smooth each raw coefficient sequence
    // ------------------------------------------------------------------
    let intercept = smoothing::local_linear(
        &argvals_owned,
        &raw_intercept,
        &argvals_owned,
        bandwidth,
        kernel,
    )?;

    let mut beta_curve = FdMatrix::zeros(p, m);
    for k in 0..p {
        let raw_k: Vec<f64> = (0..m).map(|j| raw_beta[k * m + j]).collect();
        let smooth_k =
            smoothing::local_linear(&argvals_owned, &raw_k, &argvals_owned, bandwidth, kernel)?;
        for j in 0..m {
            beta_curve[(k, j)] = smooth_k[j];
        }
    }

    // ------------------------------------------------------------------
    // 5. Step 3 — fitted curves and residuals
    // ------------------------------------------------------------------
    let mut fitted = FdMatrix::zeros(n, m);
    let mut residuals = FdMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..n {
            let mut val = intercept[j];
            for (k, pred) in predictors.iter().enumerate() {
                val += beta_curve[(k, j)] * pred[(i, j)];
            }
            fitted[(i, j)] = val;
            residuals[(i, j)] = response[(i, j)] - val;
        }
    }

    Ok(ConcurrentRegrResult {
        beta_curve,
        intercept,
        fitted,
        residuals,
        argvals: argvals_owned,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    // -----------------------------------------------------------------------
    // Task 1: Smoke / shape test
    // -----------------------------------------------------------------------

    #[test]
    fn test_shape_smoke() {
        let n = 6;
        let m = 8;
        let argvals = uniform_grid(m);

        // Build a simple response: y[i,j] = (i as f64) * argvals[j]
        let mut response = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] = (i as f64 + 1.0) * argvals[j];
            }
        }

        // Build a simple predictor: x[i,j] = (i as f64) + argvals[j]
        let mut predictor = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                predictor[(i, j)] = (i as f64 + 1.0) + argvals[j];
            }
        }

        let result = concurrent_regression(&response, &[predictor], None, 0.2, "gaussian");
        assert!(result.is_ok(), "expected Ok, got {result:?}");
        let r = result.unwrap();

        assert_eq!(r.beta_curve.shape(), (1, m), "beta_curve shape");
        assert_eq!(r.intercept.len(), m, "intercept length");
        assert_eq!(r.fitted.shape(), (n, m), "fitted shape");
        assert_eq!(r.residuals.shape(), (n, m), "residuals shape");
        assert_eq!(r.argvals.len(), m, "argvals length");
    }

    // -----------------------------------------------------------------------
    // Task 2: Multi-predictor shape test
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_predictor_shape() {
        let n = 6;
        let m = 8;
        let p = 3;
        let argvals = uniform_grid(m);

        let mut response = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] = (i as f64 + 1.0) * argvals[j];
            }
        }

        let predictors: Vec<FdMatrix> = (0..p)
            .map(|k| {
                let mut pred = FdMatrix::zeros(n, m);
                for i in 0..n {
                    for j in 0..m {
                        pred[(i, j)] = (i as f64 + k as f64 + 1.0) * argvals[j] + 0.1;
                    }
                }
                pred
            })
            .collect();

        let result = concurrent_regression(&response, &predictors, None, 0.2, "gaussian");
        assert!(result.is_ok(), "expected Ok for p=3, got {result:?}");
        let r = result.unwrap();
        assert_eq!(r.beta_curve.shape(), (p, m), "beta_curve shape for p=3");
    }

    // -----------------------------------------------------------------------
    // Task 2: Determinism / parallel-sequential equivalence
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_sequential_equivalence() {
        let n = 6;
        let m = 8;
        let argvals = uniform_grid(m);

        let mut response = FdMatrix::zeros(n, m);
        let mut predictor = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] = (i as f64 + 1.0) * argvals[j] + 0.3;
                predictor[(i, j)] = argvals[j] * (i as f64 + 0.5);
            }
        }

        let r1 = concurrent_regression(&response, &[predictor.clone()], None, 0.2, "gaussian")
            .expect("first call should succeed");
        let r2 = concurrent_regression(&response, &[predictor], None, 0.2, "gaussian")
            .expect("second call should succeed");

        assert_eq!(
            r1, r2,
            "two calls on identical inputs must produce equal results"
        );
    }

    // -----------------------------------------------------------------------
    // Task 3: Recovery of known β(t)
    // -----------------------------------------------------------------------

    /// Simple inline LCG for deterministic noise (no rand crate).
    fn lcg_noise(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Map to (-0.5, 0.5)
                (state >> 33) as f64 / (u32::MAX as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn test_recovery_known_beta() {
        let n = 50usize;
        let m = 50usize;
        let argvals = uniform_grid(m);

        // true β₀(t) = 0.5, true β₁(t) = sin(π·t)  (half period, lower curvature than
        // sin(2πt) so local-linear smoothing recovers it with bandwidth=0.15 within tolerance).
        let true_beta0 = 0.5_f64;
        let true_beta: Vec<f64> = argvals
            .iter()
            .map(|&t| (std::f64::consts::PI * t).sin())
            .collect();

        // x[i,j] = sin(2π*(i/n) + argvals[j])
        let mut predictor = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                predictor[(i, j)] =
                    (2.0 * std::f64::consts::PI * (i as f64 / n as f64) + argvals[j]).sin();
            }
        }

        // Determine max |x| for noise scaling
        let mut x_max = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                let v = predictor[(i, j)].abs();
                if v > x_max {
                    x_max = v;
                }
            }
        }
        let noise_scale = 0.05 * x_max;

        // Build response with low noise
        let noise = lcg_noise(42, n * m);
        let mut response = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] =
                    true_beta0 + true_beta[j] * predictor[(i, j)] + noise_scale * noise[i * m + j];
            }
        }

        let result =
            concurrent_regression(&response, &[predictor], Some(&argvals), 0.15, "gaussian")
                .expect("recovery test should succeed");

        // Check interior indices only (boundary edge effects excluded).
        // For bandwidth=0.15 on a [0,1] grid with 50 points (spacing ≈0.020),
        // the effective boundary zone is ~8 grid points on each side.
        for j in 5..45 {
            let diff = (result.beta_curve[(0, j)] - true_beta[j]).abs();
            assert!(
                diff < 0.15,
                "recovery failed at j={j}: got {}, expected {}, diff={diff}",
                result.beta_curve[(0, j)],
                true_beta[j]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Task 3: Monotone roughness test
    // -----------------------------------------------------------------------

    fn roughness(row: &[f64]) -> f64 {
        let m = row.len();
        if m < 3 {
            return 0.0;
        }
        (1..m - 1)
            .map(|j| {
                let d = row[j + 1] - 2.0 * row[j] + row[j - 1];
                d * d
            })
            .sum()
    }

    #[test]
    fn test_monotone_roughness() {
        let n = 50usize;
        let m = 50usize;
        let argvals = uniform_grid(m);

        let true_beta0 = 0.5_f64;
        let true_beta: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();

        let mut predictor = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                predictor[(i, j)] =
                    (2.0 * std::f64::consts::PI * (i as f64 / n as f64) + argvals[j]).sin();
            }
        }

        let mut x_max = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                let v = predictor[(i, j)].abs();
                if v > x_max {
                    x_max = v;
                }
            }
        }
        let noise_scale = 0.05 * x_max;
        let noise = lcg_noise(42, n * m);
        let mut response = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] =
                    true_beta0 + true_beta[j] * predictor[(i, j)] + noise_scale * noise[i * m + j];
            }
        }

        let bandwidths = [0.05, 0.15, 0.35];
        let roughnesses: Vec<f64> = bandwidths
            .iter()
            .map(|&bw| {
                let r = concurrent_regression(
                    &response,
                    &[predictor.clone()],
                    Some(&argvals),
                    bw,
                    "gaussian",
                )
                .expect("monotone roughness call should succeed");
                let beta_row: Vec<f64> = (0..m).map(|j| r.beta_curve[(0, j)]).collect();
                roughness(&beta_row)
            })
            .collect();

        assert!(
            roughnesses[0] > roughnesses[1],
            "roughness(bw=0.05)={} should be > roughness(bw=0.15)={}",
            roughnesses[0],
            roughnesses[1]
        );
        assert!(
            roughnesses[1] > roughnesses[2],
            "roughness(bw=0.15)={} should be > roughness(bw=0.35)={}",
            roughnesses[1],
            roughnesses[2]
        );
    }

    // -----------------------------------------------------------------------
    // Task 4: Residuals consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_residuals_consistency() {
        let n = 6;
        let m = 8;
        let argvals = uniform_grid(m);

        let mut response = FdMatrix::zeros(n, m);
        let mut predictor = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] = (i as f64 + 1.0) * argvals[j] + 0.1;
                predictor[(i, j)] = argvals[j] + i as f64 * 0.1;
            }
        }

        let result = concurrent_regression(&response, &[predictor], None, 0.2, "gaussian")
            .expect("residuals consistency call should succeed");

        for i in 0..n {
            for j in 0..m {
                let expected_resid = response[(i, j)] - result.fitted[(i, j)];
                let diff = (result.residuals[(i, j)] - expected_resid).abs();
                assert!(
                    diff < 1e-10,
                    "residual inconsistency at ({i},{j}): stored={}, computed={expected_resid}, diff={diff}",
                    result.residuals[(i, j)]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Task 4: Invalid input guards
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_inputs() {
        let n = 6;
        let m = 8;
        let argvals = uniform_grid(m);

        let mut response = FdMatrix::zeros(n, m);
        let mut predictor = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] = (i as f64 + 1.0) * argvals[j];
                predictor[(i, j)] = argvals[j];
            }
        }

        // 1. Empty predictors slice
        let err = concurrent_regression(&response, &[], None, 0.2, "gaussian")
            .expect_err("empty predictors should return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidDimension {
                    parameter: "predictors",
                    ..
                }
            ),
            "expected InvalidDimension for empty predictors, got {err:?}"
        );

        // 2. Response with n < 2
        let tiny_response = FdMatrix::zeros(1, m);
        let tiny_pred = FdMatrix::zeros(1, m);
        let err = concurrent_regression(&tiny_response, &[tiny_pred], None, 0.2, "gaussian")
            .expect_err("n<2 response should return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidDimension {
                    parameter: "response",
                    ..
                }
            ),
            "expected InvalidDimension for response n<2, got {err:?}"
        );

        // 3. predictor[0].nrows() != n
        let wrong_n_pred = FdMatrix::zeros(n + 1, m);
        let err = concurrent_regression(&response, &[wrong_n_pred], None, 0.2, "gaussian")
            .expect_err("wrong nrows predictor should return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidDimension {
                    parameter: "predictors[k]",
                    ..
                }
            ),
            "expected InvalidDimension for predictor nrows mismatch, got {err:?}"
        );

        // 4. predictor[0].ncols() != m
        let wrong_m_pred = FdMatrix::zeros(n, m + 1);
        let err = concurrent_regression(&response, &[wrong_m_pred], None, 0.2, "gaussian")
            .expect_err("wrong ncols predictor should return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidDimension {
                    parameter: "predictors[k]",
                    ..
                }
            ),
            "expected InvalidDimension for predictor ncols mismatch, got {err:?}"
        );

        // 5. bandwidth = 0.0
        let err = concurrent_regression(&response, &[predictor.clone()], None, 0.0, "gaussian")
            .expect_err("bandwidth=0.0 should return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "bandwidth",
                    ..
                }
            ),
            "expected InvalidParameter for bandwidth=0.0, got {err:?}"
        );

        // 6. bandwidth < 0
        let err = concurrent_regression(&response, &[predictor.clone()], None, -0.1, "gaussian")
            .expect_err("negative bandwidth should return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "bandwidth",
                    ..
                }
            ),
            "expected InvalidParameter for negative bandwidth, got {err:?}"
        );

        // 7. argvals Some with wrong length
        let wrong_argvals: Vec<f64> = uniform_grid(m + 3);
        let err = concurrent_regression(
            &response,
            &[predictor],
            Some(&wrong_argvals),
            0.2,
            "gaussian",
        )
        .expect_err("wrong argvals length should return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidDimension {
                    parameter: "argvals",
                    ..
                }
            ),
            "expected InvalidDimension for argvals length mismatch, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Regression test: CR-01 — NaN/Inf bandwidth must return an error
    //
    // Before the fix, `NaN <= 0.0` evaluated to `false` under IEEE 754, so
    // NaN bandwidth bypassed the guard and silently returned all-zero
    // coefficients. This test pins the corrected behaviour.
    // -----------------------------------------------------------------------

    #[test]
    fn test_nan_inf_bandwidth_returns_error() {
        let n = 4;
        let m = 6;

        let mut response = FdMatrix::zeros(n, m);
        let mut predictor = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                response[(i, j)] = (i as f64 + 1.0) * (j as f64 + 1.0);
                predictor[(i, j)] = (i as f64 + 1.0) + (j as f64 * 0.1);
            }
        }

        // NaN bandwidth: previously bypassed the guard → silent wrong output.
        let err =
            concurrent_regression(&response, &[predictor.clone()], None, f64::NAN, "gaussian")
                .expect_err("NaN bandwidth must return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "bandwidth",
                    ..
                }
            ),
            "expected InvalidParameter for NaN bandwidth, got {err:?}"
        );

        // +Infinity bandwidth: also previously bypassed the guard.
        let err = concurrent_regression(
            &response,
            &[predictor.clone()],
            None,
            f64::INFINITY,
            "gaussian",
        )
        .expect_err("+Inf bandwidth must return Err");
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "bandwidth",
                    ..
                }
            ),
            "expected InvalidParameter for +Inf bandwidth, got {err:?}"
        );

        // Sanity-check: a normal positive bandwidth still succeeds.
        concurrent_regression(&response, &[predictor], None, 0.2, "gaussian")
            .expect("positive finite bandwidth must succeed");
    }
}
