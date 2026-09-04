//! Structured-penalty scalar-on-function regression via PEER.
//!
//! PEER (Partially Empirical Eigenvectors for Regression) estimates the
//! coefficient function β(t) via penalized normal equations
//! `(W_c'W_c + λQ)β = W_c'y_c`, where `W_c` is the centered, integration-
//! weighted design matrix and `Q` is a penalty chosen from three families.
//!
//! # Penalty families
//!
//! - [`PeerPenalty::Ridge`] — identity penalty (uniform shrinkage).
//! - [`PeerPenalty::Difference`] — second-difference roughness (D'D, order 2).
//! - [`PeerPenalty::Decree`] — caller-supplied structured penalty matrix.
//!
//! # Quick start
//!
//! ```no_run
//! use fdars_core::matrix::FdMatrix;
//! use fdars_core::peer::{peer, PeerConfig, PeerPenalty};
//!
//! let data = FdMatrix::zeros(50, 40);
//! let y = vec![0.0_f64; 50];
//! let argvals: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
//! let config = PeerConfig { penalty: PeerPenalty::Difference { order: 2 }, lambda: 1.0 };
//! let _result = peer(&data, &y, &argvals, &config);
//! ```

use crate::error::FdarError;
use crate::function_on_scalar::penalty_matrix;
use crate::helpers::simpsons_weights;
use crate::linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve};
use crate::matrix::FdMatrix;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Penalty family for the PEER estimator.
///
/// Selects the penalty matrix Q (m×m) entering the penalized normal equations
/// `(W_c'W_c + λQ)β = W_c'y_c`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PeerPenalty {
    /// Identity penalty Q = I_m (uniform ridge shrinkage).
    Ridge,
    /// Second-difference roughness penalty Q = D'D (only `order = 2` is
    /// supported in this release; other orders return
    /// [`FdarError::InvalidParameter`]).
    Difference { order: usize },
    /// Caller-supplied penalty matrix.
    ///
    /// The tuple holds `(q_flat, p)` where `q_flat` is a flat row-major m×m
    /// symmetric PSD matrix (length `p*p`) and `p` is its dimension. `p` must
    /// equal the number of argvals grid points `m`; a mismatch returns
    /// [`FdarError::InvalidDimension`].
    ///
    /// Q must be symmetric (row-major and column-major are equivalent for
    /// symmetric matrices) and positive semi-definite for Cholesky stability.
    Decree(Vec<f64>, usize),
}

impl Default for PeerPenalty {
    fn default() -> Self {
        PeerPenalty::Difference { order: 2 }
    }
}

/// Configuration for the [`peer`] estimator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PeerConfig {
    /// Penalty family selecting the structured Q matrix.
    pub penalty: PeerPenalty,
    /// Smoothing parameter λ ≥ 0.  Default `1.0`.
    pub lambda: f64,
}

impl Default for PeerConfig {
    fn default() -> Self {
        Self {
            penalty: PeerPenalty::default(),
            lambda: 1.0,
        }
    }
}

/// Result of the [`peer`] estimator.
///
/// Carries the estimated coefficient function β(t), model diagnostics, and
/// the penalty configuration used — all on the `argvals` grid.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use = "expensive computation whose result should not be discarded"]
pub struct PeerResult {
    /// Estimated coefficient function β(t), length m (on the argvals grid).
    pub beta: Vec<f64>,
    /// Intercept (= ȳ, the mean of the response vector).
    pub intercept: f64,
    /// Fitted values ŷ_i, length n.
    pub fitted_values: Vec<f64>,
    /// Effective degrees of freedom tr(H) = tr((W_c'W_c + λQ)^{-1} W_c'W_c).
    pub effective_df: f64,
    /// Smoothing parameter λ that was used.
    pub lambda: f64,
    /// Penalty family that was used.
    pub penalty_type: PeerPenalty,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Fit the PEER scalar-on-function regression model.
///
/// Estimates the coefficient function β(t) on the `argvals` grid by solving
/// the penalized normal equations `(W_c'W_c + λQ)β = W_c'y_c`, where
/// `W_c[i,j] = (data[(i,j)] · w[j]) − column_mean` and `w` are Simpson's
/// integration weights.
///
/// # Arguments
///
/// * `data`    — n×m functional predictor matrix (rows = observations,
///   columns = evaluation points; column-major [`FdMatrix`]).
/// * `y`       — scalar response vector, length n.
/// * `argvals` — evaluation grid, length m (must equal `data.ncols()`).
/// * `config`  — penalty family and λ.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] when dimensions are inconsistent
/// or a Decree Q has wrong size, [`FdarError::InvalidParameter`] for
/// unsupported `Difference` orders, and [`FdarError::ComputationFailed`]
/// when the penalized system is numerically singular.
pub fn peer(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    config: &PeerConfig,
) -> Result<PeerResult, FdarError> {
    let (n, m) = data.shape();

    // --- Entry validation ---
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 observation".to_string(),
            actual: "0 rows".to_string(),
        });
    }
    if m < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 evaluation points (m >= 3)".to_string(),
            actual: format!("{m} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }

    // 1. Integration weights w[j]
    let w = simpsons_weights(argvals);

    // 2. Weighted design: wmat[i,j] = data[i,j] * w[j]
    let mut wmat = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            wmat[(i, j)] = data[(i, j)] * w[j];
        }
    }

    // 3. Center response and design
    let y_bar: f64 = y.iter().sum::<f64>() / n as f64;
    let yc: Vec<f64> = y.iter().map(|&yi| yi - y_bar).collect();

    let w_bar: Vec<f64> = (0..m)
        .map(|j| (0..n).map(|i| wmat[(i, j)]).sum::<f64>() / n as f64)
        .collect();

    let mut wc = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            wc[(i, j)] = wmat[(i, j)] - w_bar[j];
        }
    }

    // 4. Build penalty Q (m×m, row-major)
    let q = build_q(m, &config.penalty)?;

    // 5. WtW (m×m, row-major, symmetric) and wty (length m)
    let mut wtw = vec![0.0_f64; m * m];
    for j in 0..m {
        for k in j..m {
            let s: f64 = (0..n).map(|i| wc[(i, j)] * wc[(i, k)]).sum();
            wtw[j * m + k] = s;
            wtw[k * m + j] = s;
        }
    }
    let wty: Vec<f64> = (0..m)
        .map(|j| (0..n).map(|i| wc[(i, j)] * yc[i]).sum())
        .collect();

    // 6. A = WtW + λQ; solve A β = wty via Cholesky
    let lambda = config.lambda;
    let mut a = vec![0.0_f64; m * m];
    for i in 0..m * m {
        a[i] = wtw[i] + lambda * q[i];
    }
    let beta = cholesky_solve(&a, &wty, m)?;

    // NaN guard: non-finite β should not be returned silently
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::ComputationFailed {
            operation: "peer",
            detail: "non-finite coefficient (singular penalized system)".into(),
        });
    }

    // 7. Effective degrees of freedom: tr(H) = tr(A^{-1} WtW)
    let effective_df = compute_peer_trace_hat(&wtw, &q, lambda, m, n);

    // 8. Fitted values: ŷ[i] = ȳ + Σ_j wc[i,j] · β[j]
    let fitted_values: Vec<f64> = (0..n)
        .map(|i| y_bar + (0..m).map(|j| wc[(i, j)] * beta[j]).sum::<f64>())
        .collect();

    Ok(PeerResult {
        beta,
        intercept: y_bar,
        fitted_values,
        effective_df,
        lambda,
        penalty_type: config.penalty.clone(),
    })
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build the m×m penalty matrix Q (flat row-major) for the given family.
fn build_q(m: usize, penalty: &PeerPenalty) -> Result<Vec<f64>, FdarError> {
    match penalty {
        PeerPenalty::Ridge => {
            let mut q = vec![0.0_f64; m * m];
            for i in 0..m {
                q[i * m + i] = 1.0;
            }
            Ok(q)
        }
        PeerPenalty::Difference { order: 2 } => Ok(penalty_matrix(m)),
        PeerPenalty::Difference { order } => Err(FdarError::InvalidParameter {
            parameter: "penalty",
            message: format!(
                "Difference order {order} unsupported; only order 2 is available in this release"
            ),
        }),
        PeerPenalty::Decree(q_raw, p_q) => {
            if *p_q != m || q_raw.len() != m * m {
                return Err(FdarError::InvalidDimension {
                    parameter: "penalty",
                    expected: format!("{m}x{m} ({} elems)", m * m),
                    actual: format!("{p_q}x{p_q} ({} elems)", q_raw.len()),
                });
            }
            Ok(q_raw.clone())
        }
    }
}

/// Compute effective df = tr(H) = tr(A^{-1} WtW) where A = WtW + λQ.
///
/// Uses the column-solve trick: for each column j of WtW, solve A z = WtW[:,j]
/// and accumulate z[j].  Falls back to `m as f64` on Cholesky failure;
/// clamps result to `n as f64`.
fn compute_peer_trace_hat(wtw: &[f64], q: &[f64], lambda: f64, m: usize, n: usize) -> f64 {
    let mut a = vec![0.0_f64; m * m];
    for i in 0..m * m {
        a[i] = wtw[i] + lambda * q[i];
    }
    let Ok(l) = cholesky_factor(&a, m) else {
        return m as f64; // fallback
    };
    let mut trace = 0.0_f64;
    for j in 0..m {
        // Extract column j of WtW (stored row-major: WtW[row, col] = wtw[row*m + col])
        let col: Vec<f64> = (0..m).map(|i| wtw[i * m + j]).collect();
        let z = cholesky_forward_back(&l, &col, m);
        trace += z[j];
    }
    trace.min(n as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    /// Stateless deterministic pseudo-random unit in [-1, 1] (splitmix64 finalizer).
    /// Used to build spanning design curves without an RNG dependency.
    fn hash_unit(k: u64) -> f64 {
        let mut z = k
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Build the shared synthetic fixture: n=200 curves, m=40 grid, true β(t)=sin(π·t).
    /// Generates y_i = Σ_j X_i(t_j)·β(t_j)·w_j + small_noise.
    ///
    /// The design curves are deterministic pseudo-random (spanning the full m-dim
    /// grid space, so W has full column rank and β(t) is identifiable) — a smooth
    /// low-rank family (e.g. phase-shifted single-frequency sinusoids) would live
    /// in a 2-D subspace and leave β(t) unrecoverable regardless of the estimator.
    /// n ≫ m keeps W'W well-conditioned and makes the fixed λ negligible relative to
    /// the signal, so the penalty bias stays well under the recovery tolerance.
    fn make_fixture() -> (FdMatrix, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (n, m) = (200_usize, 40_usize);
        let t = uniform_grid(m);
        let true_beta: Vec<f64> = t
            .iter()
            .map(|&ti| (std::f64::consts::PI * ti).sin())
            .collect();
        let w = simpsons_weights(&t);

        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..m {
                let xi = hash_unit((i * m + j) as u64);
                data[(i, j)] = xi;
                y[i] += xi * true_beta[j] * w[j];
            }
            // Small deterministic noise
            y[i] += 0.005 * hash_unit(1_000_000 + i as u64);
        }
        (data, y, t, true_beta)
    }

    // -------------------------------------------------------------------------
    // Task 1 tests: Difference{2} tracer
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_difference_beta_recovery() {
        let (data, y, t, true_beta) = make_fixture();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: 1e-4,
        };
        let result = peer(&data, &y, &t, &config).expect("peer() should succeed");

        let max_err = result
            .beta
            .iter()
            .zip(true_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 0.1,
            "Difference beta recovery error: {max_err} >= 0.1"
        );
        assert!(
            result.fitted_values.iter().all(|v| v.is_finite()),
            "fitted_values contain non-finite values"
        );
    }

    #[test]
    fn test_peer_result_shape() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        let n = y.len();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: 1e-4,
        };
        let result = peer(&data, &y, &t, &config).expect("peer() should succeed");

        assert_eq!(result.beta.len(), m, "beta length should be m={m}");
        assert_eq!(
            result.fitted_values.len(),
            n,
            "fitted_values length should be n={n}"
        );
        assert!(
            result.effective_df.is_finite() && result.effective_df > 0.0,
            "effective_df={} must be finite and positive",
            result.effective_df
        );
        assert!(
            (result.lambda - 1e-4).abs() < 1e-15,
            "lambda should be 1e-4, got {}",
            result.lambda
        );
        assert_eq!(
            result.penalty_type,
            PeerPenalty::Difference { order: 2 },
            "penalty_type should be Difference{{order:2}}"
        );

        let y_bar = y.iter().sum::<f64>() / n as f64;
        assert!(
            (result.intercept - y_bar).abs() < 1e-9,
            "intercept={} should equal mean(y)={y_bar} within 1e-9",
            result.intercept
        );
    }

    // -------------------------------------------------------------------------
    // Task 2 tests: all three penalty families
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_ridge_fits() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        let config = PeerConfig {
            penalty: PeerPenalty::Ridge,
            lambda: 1e-4,
        };
        let result = peer(&data, &y, &t, &config).expect("peer() with Ridge should succeed");
        assert_eq!(result.beta.len(), m);
        assert!(
            result.beta.iter().all(|v| v.is_finite()),
            "Ridge beta contains non-finite values"
        );
        assert!(
            result.fitted_values.iter().all(|v| v.is_finite()),
            "Ridge fitted_values contain non-finite values"
        );
    }

    #[test]
    fn test_peer_decree_fits() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        // Use the same penalty_matrix as a caller-supplied Decree matrix
        let q_flat = penalty_matrix(m);
        let config = PeerConfig {
            penalty: PeerPenalty::Decree(q_flat, m),
            lambda: 1e-4,
        };
        let result = peer(&data, &y, &t, &config).expect("peer() with Decree should succeed");
        assert_eq!(result.beta.len(), m);
        assert!(
            result.beta.iter().all(|v| v.is_finite()),
            "Decree beta contains non-finite values"
        );
        assert!(
            result.fitted_values.iter().all(|v| v.is_finite()),
            "Decree fitted_values contain non-finite values"
        );
    }

    #[test]
    fn test_peer_difference_order_rejected() {
        let (data, y, t, _) = make_fixture();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 3 },
            lambda: 1e-4,
        };
        let err = peer(&data, &y, &t, &config).expect_err("order 3 should be rejected");
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "expected InvalidParameter, got {err:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Task 3: Decree partition-structured Q yields β(t) distinct from Difference{2}
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_decree_distinct_from_roughness() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        // Use a moderate lambda so the penalty visibly shapes β(t)
        let lambda = 1.0;

        // Fit (a) with plain Difference{2}
        let config_diff = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda,
        };
        let res_diff = peer(&data, &y, &t, &config_diff).expect("Difference{2} fit should succeed");

        // Build a partition-aware Q: a second-difference operator whose stencil
        // is NOT applied across the mid-grid boundary b = m/2.
        // This leaves the two halves independently penalized.
        let b = m / 2;
        let mut q_partition = vec![0.0_f64; m * m];
        for i in 0..m.saturating_sub(2) {
            // Skip stencil rows that straddle the boundary
            if i + 1 == b || i + 2 == b || i == b {
                continue;
            }
            let coeffs = [(i, 1.0_f64), (i + 1, -2.0), (i + 2, 1.0)];
            for &(r, cr) in &coeffs {
                for &(c, cc) in &coeffs {
                    q_partition[r * m + c] += cr * cc;
                }
            }
        }
        // Add a small ridge to each diagonal to ensure PSD for Cholesky stability
        for i in 0..m {
            q_partition[i * m + i] += 1e-6;
        }

        let config_dec = PeerConfig {
            penalty: PeerPenalty::Decree(q_partition, m),
            lambda,
        };
        let res_dec =
            peer(&data, &y, &t, &config_dec).expect("Decree partition fit should succeed");

        // Both fits must be all-finite
        assert!(
            res_diff.beta.iter().all(|v| v.is_finite()),
            "Difference beta contains non-finite"
        );
        assert!(
            res_dec.beta.iter().all(|v| v.is_finite()),
            "Decree beta contains non-finite"
        );

        // The two β(t) must differ meaningfully (> 1e-3 max pointwise)
        let max_diff = res_diff
            .beta
            .iter()
            .zip(res_dec.beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-3,
            "Decree partition Q should yield a β(t) distinct from Difference{{2}} \
             by > 1e-3; got max_diff={max_diff}"
        );
    }

    // -------------------------------------------------------------------------
    // Task 4: Error/NaN surface — wrong-dim Q, dim mismatch, no NaN
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_decree_wrong_dim() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        // Q sized (m-1)×(m-1) ≠ m×m
        let config = PeerConfig {
            penalty: PeerPenalty::Decree(vec![0.0; (m - 1) * (m - 1)], m - 1),
            lambda: 1.0,
        };
        let err = peer(&data, &y, &t, &config).expect_err("wrong-dim Decree should return Err");
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "expected InvalidDimension, got {err:?}"
        );
    }

    #[test]
    fn test_peer_argvals_mismatch() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        // argvals has length m+1 instead of m
        let argvals_bad: Vec<f64> = (0..=m).map(|i| i as f64 / m as f64).collect();
        let config = PeerConfig::default();
        let err = peer(&data, &y, &argvals_bad, &config)
            .expect_err("mismatched argvals should return Err");
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "expected InvalidDimension, got {err:?}"
        );
    }

    #[test]
    fn test_peer_no_nan_all_families() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        let lambda = 1.0;

        // Ridge
        let config_ridge = PeerConfig {
            penalty: PeerPenalty::Ridge,
            lambda,
        };
        let res_ridge = peer(&data, &y, &t, &config_ridge).expect("Ridge should succeed");
        assert!(
            res_ridge.beta.iter().all(|v| v.is_finite()),
            "Ridge beta contains NaN/Inf"
        );
        assert!(
            res_ridge.fitted_values.iter().all(|v| v.is_finite()),
            "Ridge fitted_values contains NaN/Inf"
        );
        assert!(
            res_ridge.effective_df.is_finite(),
            "Ridge effective_df is not finite"
        );

        // Difference{2}
        let config_diff = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda,
        };
        let res_diff = peer(&data, &y, &t, &config_diff).expect("Difference{2} should succeed");
        assert!(
            res_diff.beta.iter().all(|v| v.is_finite()),
            "Difference beta contains NaN/Inf"
        );
        assert!(
            res_diff.fitted_values.iter().all(|v| v.is_finite()),
            "Difference fitted_values contains NaN/Inf"
        );
        assert!(
            res_diff.effective_df.is_finite(),
            "Difference effective_df is not finite"
        );

        // Decree using penalty_matrix(m) as a valid caller-supplied Q
        let q_flat = penalty_matrix(m);
        let config_dec = PeerConfig {
            penalty: PeerPenalty::Decree(q_flat, m),
            lambda,
        };
        let res_dec = peer(&data, &y, &t, &config_dec).expect("Decree should succeed");
        assert!(
            res_dec.beta.iter().all(|v| v.is_finite()),
            "Decree beta contains NaN/Inf"
        );
        assert!(
            res_dec.fitted_values.iter().all(|v| v.is_finite()),
            "Decree fitted_values contains NaN/Inf"
        );
        assert!(
            res_dec.effective_df.is_finite(),
            "Decree effective_df is not finite"
        );
    }
}
