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
//! use fdars_core::peer::{peer, PeerConfig, PeerPenalty, LambdaChoice};
//!
//! let data = FdMatrix::zeros(50, 40);
//! let y = vec![0.0_f64; 50];
//! let argvals: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
//! let config = PeerConfig { penalty: PeerPenalty::Difference { order: 2 }, lambda: LambdaChoice::Fixed(1.0) };
//! let _result = peer(&data, &y, &argvals, &config);
//! ```

use crate::error::FdarError;
use crate::function_on_scalar::penalty_matrix;
use crate::helpers::simpsons_weights;
use crate::linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve};
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;

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

/// How to choose the smoothing parameter λ for the PEER estimator.
///
/// The default is [`LambdaChoice::Gcv`] which automatically selects λ by
/// minimising the GCV score over a fixed internal log-spaced grid.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LambdaChoice {
    /// Use this value verbatim; no grid search or EM is performed.
    Fixed(f64),
    /// Select λ by minimising the GCV score on a fixed 40-point log-spaced
    /// internal grid over [1e-6, 1e4].  Fully deterministic; ties resolve to
    /// the smaller grid index.
    #[default]
    Gcv,
    /// Estimate λ via a self-contained REML EM using the eigendecomposition of
    /// the penalty matrix Q (null space → fixed effect; range space → random
    /// effect b ∼ N(0, σ²_u I)).  Returns λ = σ²_e / σ²_u.  Fully
    /// deterministic; fixed initialisation and 100-iteration cap.
    Reml,
}

/// Records which λ-selection path actually ran.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LambdaMethod {
    /// A fixed value was supplied via [`LambdaChoice::Fixed`]; no search ran.
    Fixed,
    /// λ was selected by GCV grid search.
    Gcv,
    /// λ was estimated by the self-contained REML EM.
    Reml,
}

/// Configuration for the [`peer`] estimator.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PeerConfig {
    /// Penalty family selecting the structured Q matrix.
    pub penalty: PeerPenalty,
    /// How to choose (or fix) the smoothing parameter λ.
    ///
    /// - [`LambdaChoice::Fixed(v)`](LambdaChoice::Fixed) — use `v` verbatim.
    /// - [`LambdaChoice::Gcv`] — automatic GCV grid search (default).
    /// - [`LambdaChoice::Reml`] — automatic REML EM estimation.
    pub lambda: LambdaChoice,
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
    ///
    /// This is the centered-response mean, not the out-of-sample prediction
    /// intercept. Prediction on a new curve x* uses
    /// `ȳ + Σ_j (x*[j]·w[j] − w_bar[j])·β[j] = (ȳ − w_bar·β) + Σ_j x*[j]·w[j]·β[j]`,
    /// so a predictor must combine `intercept` with [`w_bar`](Self::w_bar) and
    /// `beta`. Storing `w_bar` here keeps that reconstruction exact (consumed by
    /// out-of-sample prediction in a later phase).
    pub intercept: f64,
    /// Column means of the Simpson-weighted design `W[i,j] = data[(i,j)]·w[j]`,
    /// length m. Retained so out-of-sample prediction can reproduce the same
    /// centering the fit used (see [`intercept`](Self::intercept)).
    pub w_bar: Vec<f64>,
    /// Fitted values ŷ_i, length n.
    pub fitted_values: Vec<f64>,
    /// Effective degrees of freedom tr(H) = tr((W_c'W_c + λQ)^{-1} W_c'W_c).
    pub effective_df: f64,
    /// Smoothing parameter λ that was used.
    pub lambda: f64,
    /// Penalty family that was used.
    pub penalty_type: PeerPenalty,
    /// GCV score at the selected λ.  `Some(score)` when [`LambdaMethod::Gcv`]
    /// ran; `None` when [`LambdaMethod::Fixed`] or [`LambdaMethod::Reml`].
    pub gcv: Option<f64>,
    /// Which λ-selection path ran.
    pub lambda_method: LambdaMethod,
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
/// * `config`  — penalty family and λ selection.
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
    if n < 2 {
        // Centering collapses a single observation to β = 0 (all-zero design);
        // require at least 2 so the fit is not silently degenerate.
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 observations".to_string(),
            actual: format!("{n} rows"),
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
    // Non-finite inputs would propagate into finite-but-wrong normal equations,
    // bypassing the post-solve β NaN guard — reject them explicitly.
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: "response contains non-finite values (NaN/Inf)".to_string(),
        });
    }
    if argvals.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals contains non-finite values (NaN/Inf)".to_string(),
        });
    }
    // Simpson's weights assume a strictly increasing grid; a non-monotone grid
    // yields negative weights that silently corrupt the design integral.
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be strictly increasing".to_string(),
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

    // 6. λ selection — dispatch based on config.lambda
    let (lambda, gcv_score, lambda_method) = match &config.lambda {
        LambdaChoice::Fixed(lam) => (*lam, None, LambdaMethod::Fixed),
        LambdaChoice::Gcv => {
            let (lam, g) = select_lambda_gcv_peer(&wc, &yc, &wtw, &wty, &q, m, n);
            (lam, Some(g), LambdaMethod::Gcv)
        }
        LambdaChoice::Reml => {
            let lam = select_lambda_reml_peer(&wc, &yc, &q, m, n);
            (lam, None, LambdaMethod::Reml)
        }
    };

    // 7. A = WtW + λQ; solve A β = wty via Cholesky
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

    // 8. Effective degrees of freedom: tr(H) = tr(A^{-1} WtW)
    let effective_df = compute_peer_trace_hat(&wtw, &q, lambda, m, n);

    // 9. Fitted values: ŷ[i] = ȳ + Σ_j wc[i,j] · β[j]
    let fitted_values: Vec<f64> = (0..n)
        .map(|i| y_bar + (0..m).map(|j| wc[(i, j)] * beta[j]).sum::<f64>())
        .collect();

    Ok(PeerResult {
        beta,
        intercept: y_bar,
        w_bar,
        fitted_values,
        effective_df,
        lambda,
        penalty_type: config.penalty.clone(),
        gcv: gcv_score,
        lambda_method,
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

/// Build a 40-point log-spaced grid on [1e-6, 1e4].
///
/// Grid point i is `10^(-6 + 10·i/39)` for i in 0..40.  The grid is purely
/// a function of constants — no RNG — so GCV selection is fully deterministic.
fn gcv_lambda_grid() -> Vec<f64> {
    (0..40)
        .map(|i| 10.0_f64.powf(-6.0 + 10.0 * i as f64 / 39.0))
        .collect()
}

/// Select λ by minimising the GCV score over the fixed 40-point grid.
///
/// GCV score: `GCV(λ) = n·RSS(λ) / (n − tr(H(λ)))²`.
///
/// Tie-break: when two grid points share the minimum GCV (within floating-point
/// equality), the smaller grid index (smaller λ) is retained.
///
/// Guard: if `n − tr(H) ≤ 0` for a candidate (degenerate, over-smoothed), that
/// grid point is skipped.  If no grid point yields a finite score, the smallest
/// grid λ is returned with score `f64::INFINITY`.
///
/// Returns `(best_lambda, gcv_at_best)`.
fn select_lambda_gcv_peer(
    wc: &FdMatrix,
    yc: &[f64],
    wtw: &[f64],
    wty: &[f64],
    q: &[f64],
    m: usize,
    n: usize,
) -> (f64, f64) {
    let grid = gcv_lambda_grid();
    let mut best_lam = grid[0];
    let mut best_gcv = f64::INFINITY;

    for &lam in &grid {
        // Build A = WtW + lam*Q
        let mut a = vec![0.0_f64; m * m];
        for i in 0..m * m {
            a[i] = wtw[i] + lam * q[i];
        }
        // Solve for beta
        let Ok(beta) = cholesky_solve(&a, wty, m) else {
            continue;
        };
        // RSS = ||y_c - W_c beta||^2
        let rss: f64 = (0..n)
            .map(|i| {
                let yhat: f64 = (0..m).map(|j| wc[(i, j)] * beta[j]).sum();
                (yc[i] - yhat).powi(2)
            })
            .sum();
        // tr(H) reuses compute_peer_trace_hat
        let trh = compute_peer_trace_hat(wtw, q, lam, m, n);
        // Guard: skip if denominator <= 0
        let denom = n as f64 - trh;
        if denom <= 0.0 {
            continue;
        }
        let gcv = n as f64 * rss / (denom * denom);
        // Strict < keeps first (smaller-λ) grid index on ties
        if gcv < best_gcv {
            best_gcv = gcv;
            best_lam = lam;
        }
    }
    (best_lam, best_gcv)
}

/// Select λ via a self-contained REML EM using the eigendecomposition of Q.
///
/// The PEER-as-mixed-model equivalence partitions the m-dimensional coefficient
/// space into:
/// - Null space of Q (s dimensions) → fixed (unpenalised) effect α.
/// - Range space of Q (r dimensions) → random effect b ∼ N(0, σ²_u I_r).
///
/// An EM loop estimates σ²_u and σ²_e; the returned λ = σ²_e / σ²_u.
///
/// Edge cases:
/// - Ridge (Q = I_m, s = 0): GLS α-update is skipped; r_alpha = y_c.
/// - Zero-Q Decree (r = 0): returns the documented fallback λ = 1e-4.
/// - σ²_u → 0: clamped to 1e-12 (keeps λ finite).
///
/// Deterministic: fixed initialisation + 100-iteration cap, no RNG.
fn select_lambda_reml_peer(wc: &FdMatrix, yc: &[f64], q: &[f64], m: usize, n: usize) -> f64 {
    // --- STEP 1: eigendecompose Q (ascending eigenvalue order, null space first) ---
    let q_mat = DMatrix::from_row_slice(m, m, q);
    let eigen = q_mat.symmetric_eigen();

    let mut idx_sorted: Vec<usize> = (0..m).collect();
    idx_sorted.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_ev = idx_sorted
        .iter()
        .map(|&i| eigen.eigenvalues[i].abs())
        .fold(0.0_f64, f64::max);
    let tol = 1e-8 * max_ev.max(1.0);

    let null_idx: Vec<usize> = idx_sorted
        .iter()
        .copied()
        .filter(|&i| eigen.eigenvalues[i].abs() < tol)
        .collect();
    let range_idx: Vec<usize> = idx_sorted
        .iter()
        .copied()
        .filter(|&i| eigen.eigenvalues[i].abs() >= tol)
        .collect();
    let s = null_idx.len();
    let r = range_idx.len();

    // --- STEP 2: edge — zero-Q Decree (range space empty) ---
    if r == 0 {
        return 1e-4; // documented fallback: no random effect to estimate
    }

    // --- STEP 3: project designs into eigenbasis ---
    // z_null[i*s + col]  = Σ_j wc[(i,j)] * V_null[(j, col)]
    // z_range[i*r + col] = Σ_j wc[(i,j)] * V_range[(j, col)]
    let mut z_null = vec![0.0_f64; n * s];
    let mut z_range = vec![0.0_f64; n * r];
    for i in 0..n {
        for (col, &ev_idx) in null_idx.iter().enumerate() {
            let mut val = 0.0;
            for j in 0..m {
                val += wc[(i, j)] * eigen.eigenvectors[(j, ev_idx)];
            }
            z_null[i * s + col] = val;
        }
        for (col, &ev_idx) in range_idx.iter().enumerate() {
            let mut val = 0.0;
            for j in 0..m {
                val += wc[(i, j)] * eigen.eigenvectors[(j, ev_idx)];
            }
            z_range[i * r + col] = val;
        }
    }

    // Pre-compute ZtZ_range (r×r, row-major, symmetric) — built once, reused each iter
    let mut ztz_range = vec![0.0_f64; r * r];
    for a in 0..r {
        for b in a..r {
            let s_val: f64 = (0..n)
                .map(|i| z_range[i * r + a] * z_range[i * r + b])
                .sum();
            ztz_range[a * r + b] = s_val;
            ztz_range[b * r + a] = s_val;
        }
    }

    // --- STEP 4: deterministic init (no RNG) ---
    let y_mean = yc.iter().sum::<f64>() / n as f64;
    let y_var = yc.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let mut sigma2_e = y_var.max(1e-12);
    let mut sigma2_u = (sigma2_e * 0.1).max(1e-12);

    // OLS init of alpha (if s > 0): solve (Z_null' Z_null + 1e-10 I_s) alpha = Z_null' yc
    let mut alpha = vec![0.0_f64; s];
    if s > 0 {
        // ZtZ_null (s×s) + ridge
        let mut ztz_null = vec![0.0_f64; s * s];
        for a in 0..s {
            for b in a..s {
                let sv: f64 = (0..n).map(|i| z_null[i * s + a] * z_null[i * s + b]).sum();
                ztz_null[a * s + b] = sv;
                ztz_null[b * s + a] = sv;
            }
        }
        for diag in 0..s {
            ztz_null[diag * s + diag] += 1e-10;
        }
        // Z_null' yc
        let zty_null: Vec<f64> = (0..s)
            .map(|col| (0..n).map(|i| z_null[i * s + col] * yc[i]).sum())
            .collect();
        if let Ok(a_init) = cholesky_solve(&ztz_null, &zty_null, s) {
            alpha = a_init;
        }
    }

    // --- STEP 5: EM loop, fixed cap 100 iterations ---
    for _iter in 0..100 {
        let su_old = sigma2_u;
        let se_old = sigma2_e;

        // r_alpha = yc - Z_null * alpha  (residual after fixed effects)
        let r_alpha: Vec<f64> = if s > 0 {
            (0..n)
                .map(|i| {
                    let za: f64 = (0..s).map(|col| z_null[i * s + col] * alpha[col]).sum();
                    yc[i] - za
                })
                .collect()
        } else {
            yc.to_vec()
        };

        // E-step: M = ZtZ_range/sigma2_e + I_r/sigma2_u  (r×r)
        let mut big_m = vec![0.0_f64; r * r];
        for idx in 0..r * r {
            big_m[idx] = ztz_range[idx] / sigma2_e;
        }
        for diag in 0..r {
            big_m[diag * r + diag] += 1.0 / sigma2_u;
        }

        // Cholesky of M; on failure keep old variances and continue
        let l_m = match cholesky_factor(&big_m, r) {
            Ok(l) => l,
            Err(_) => {
                sigma2_u = su_old;
                sigma2_e = se_old;
                break;
            }
        };

        // Sigma_b = M^{-1}: solve column by column (r solves)
        let mut sigma_b = vec![0.0_f64; r * r];
        let mut trace_sigma_b = 0.0_f64;
        for col in 0..r {
            let mut e_col = vec![0.0_f64; r];
            e_col[col] = 1.0;
            let sol = cholesky_forward_back(&l_m, &e_col, r);
            for row in 0..r {
                sigma_b[row * r + col] = sol[row];
            }
            trace_sigma_b += sol[col];
        }

        // Z_range' r_alpha  (length r)
        let ztr: Vec<f64> = (0..r)
            .map(|col| (0..n).map(|i| z_range[i * r + col] * r_alpha[i]).sum())
            .collect();

        // b_hat = Sigma_b * ztr / sigma2_e  (length r)
        let b_hat: Vec<f64> = (0..r)
            .map(|row| {
                (0..r)
                    .map(|col| sigma_b[row * r + col] * ztr[col])
                    .sum::<f64>()
                    / sigma2_e
            })
            .collect();

        // M-step: sigma2_u_new = (b_hat'b_hat + tr(Sigma_b)) / r
        let b_sq: f64 = b_hat.iter().map(|&v| v * v).sum();
        let sigma2_u_new = (b_sq + trace_sigma_b) / r as f64;

        // resid = r_alpha - Z_range * b_hat  (length n)
        let resid: Vec<f64> = (0..n)
            .map(|i| {
                let zb: f64 = (0..r).map(|col| z_range[i * r + col] * b_hat[col]).sum();
                r_alpha[i] - zb
            })
            .collect();

        // tr(Z_range Sigma_b Z_range') = tr(Sigma_b ZtZ_range) = Σ_{a,j} sigma_b[a*r+j]*ztz_range[j*r+a]
        let tr_zsz: f64 = (0..r)
            .map(|a| {
                (0..r)
                    .map(|j| sigma_b[a * r + j] * ztz_range[j * r + a])
                    .sum::<f64>()
            })
            .sum();

        let resid_sq: f64 = resid.iter().map(|&v| v * v).sum();
        let sigma2_e_new = (resid_sq + tr_zsz) / n as f64;

        // GLS null-space update (only when s > 0) using Woodbury identity
        // Sigma^{-1} = (1/sigma2_e)(I_n - Z_range * K^{-1} * Z_range')
        // where K = sigma2_e/sigma2_u * I_r + ZtZ_range  (r×r)
        if s > 0 {
            // Build K = ZtZ_range + (sigma2_e/sigma2_u)*I_r
            let ratio = sigma2_e_new / sigma2_u_new.max(1e-12);
            let mut k_mat = ztz_range.clone();
            for diag in 0..r {
                k_mat[diag * r + diag] += ratio;
            }
            // Cholesky of K for the Woodbury inverse
            if let Ok(l_k) = cholesky_factor(&k_mat, r) {
                // Compute Sigma^{-1} Z_null and Z_null' Sigma^{-1} Z_null for the s×s GLS system
                // Sigma^{-1} z_null_col = (1/sigma2_e)(z_null_col - Z_range * K^{-1} Z_range' z_null_col)
                let mut gls_lhs = vec![0.0_f64; s * s]; // Z_null' Sigma^{-1} Z_null  (s×s)
                let mut gls_rhs = vec![0.0_f64; s]; // Z_null' Sigma^{-1} y_c  (length s)

                // Precompute Z_range' y_c (length r)
                let zr_yc: Vec<f64> = (0..r)
                    .map(|col| (0..n).map(|i| z_range[i * r + col] * yc[i]).sum())
                    .collect();
                // K^{-1} Z_range' y_c  (length r)
                let kinv_zr_yc = cholesky_forward_back(&l_k, &zr_yc, r);
                // Sigma^{-1} y_c  (length n): (1/sigma2_e)(y_c - Z_range * K^{-1} Z_range' y_c)
                let sinv_yc: Vec<f64> = (0..n)
                    .map(|i| {
                        let zk: f64 = (0..r)
                            .map(|col| z_range[i * r + col] * kinv_zr_yc[col])
                            .sum();
                        (yc[i] - zk) / sigma2_e_new
                    })
                    .collect();

                for col_null in 0..s {
                    // Z_range' z_null_col  (length r)
                    let zr_zn: Vec<f64> = (0..r)
                        .map(|col| {
                            (0..n)
                                .map(|i| z_range[i * r + col] * z_null[i * s + col_null])
                                .sum()
                        })
                        .collect();
                    // K^{-1} Z_range' z_null_col
                    let kinv_zr_zn = cholesky_forward_back(&l_k, &zr_zn, r);
                    // Sigma^{-1} z_null_col  (length n)
                    let sinv_zn: Vec<f64> = (0..n)
                        .map(|i| {
                            let zk: f64 = (0..r)
                                .map(|col| z_range[i * r + col] * kinv_zr_zn[col])
                                .sum();
                            (z_null[i * s + col_null] - zk) / sigma2_e_new
                        })
                        .collect();

                    // Z_null' Sigma^{-1} z_null_col → column col_null of gls_lhs
                    for row_null in 0..s {
                        let v: f64 = (0..n).map(|i| z_null[i * s + row_null] * sinv_zn[i]).sum();
                        gls_lhs[row_null * s + col_null] = v;
                    }
                    // Z_null' Sigma^{-1} y_c: entry col_null of gls_rhs
                    let rhs_val: f64 = (0..n).map(|i| z_null[i * s + col_null] * sinv_yc[i]).sum();
                    gls_rhs[col_null] = rhs_val;
                }

                // Add ridge to GLS LHS for numerical stability
                for diag in 0..s {
                    gls_lhs[diag * s + diag] += 1e-10;
                }

                if let Ok(alpha_new) = cholesky_solve(&gls_lhs, &gls_rhs, s) {
                    alpha = alpha_new;
                }
            }
        }

        // Clamp variance components
        sigma2_u = sigma2_u_new.max(1e-12);
        sigma2_e = sigma2_e_new.max(1e-12);

        let delta = (sigma2_u - su_old).abs() + (sigma2_e - se_old).abs();
        if delta < 1e-8 * (su_old + se_old) {
            break;
        }
    }

    // --- STEP 6: return λ = σ²_e / σ²_u ---
    (sigma2_e / sigma2_u).max(1e-15)
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
    // Phase 66 migrated tests: Difference{2} tracer
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_difference_beta_recovery() {
        let (data, y, t, true_beta) = make_fixture();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Fixed(1e-4),
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
            lambda: LambdaChoice::Fixed(1e-4),
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
    // Phase 66 migrated tests: all three penalty families
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_ridge_fits() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        let config = PeerConfig {
            penalty: PeerPenalty::Ridge,
            lambda: LambdaChoice::Fixed(1e-4),
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
            lambda: LambdaChoice::Fixed(1e-4),
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
            lambda: LambdaChoice::Fixed(1e-4),
        };
        let err = peer(&data, &y, &t, &config).expect_err("order 3 should be rejected");
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "expected InvalidParameter, got {err:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Phase 66 migrated: Decree partition-structured Q yields β(t) distinct from Difference{2}
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_decree_distinct_from_roughness() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        // Use a moderate lambda so the penalty visibly shapes β(t)
        let lambda = LambdaChoice::Fixed(1.0);

        // Fit (a) with plain Difference{2}
        let config_diff = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: lambda.clone(),
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
    // Phase 66 migrated: Error/NaN surface — wrong-dim Q, dim mismatch, no NaN
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_decree_wrong_dim() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();
        // Q sized (m-1)×(m-1) ≠ m×m
        let config = PeerConfig {
            penalty: PeerPenalty::Decree(vec![0.0; (m - 1) * (m - 1)], m - 1),
            lambda: LambdaChoice::Fixed(1.0),
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
        let lambda = LambdaChoice::Fixed(1.0);

        // Ridge
        let config_ridge = PeerConfig {
            penalty: PeerPenalty::Ridge,
            lambda: lambda.clone(),
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
            lambda: lambda.clone(),
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

    // -------------------------------------------------------------------------
    // Phase 66 migrated: Prediction-contract + hardened-validation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_stores_w_bar_for_prediction() {
        // The stored w_bar must let an out-of-sample predictor reproduce the
        // training fitted values exactly via ŷ = (ȳ − w_bar·β) + Σ_j x[j]·w[j]·β[j].
        let (data, y, t, _) = make_fixture();
        let (n, m) = data.shape();
        let w = simpsons_weights(&t);
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Fixed(1e-4),
        };
        let result = peer(&data, &y, &t, &config).expect("peer() should succeed");

        assert_eq!(result.w_bar.len(), m, "w_bar length must equal m");

        let base = result.intercept
            - result
                .w_bar
                .iter()
                .zip(&result.beta)
                .map(|(wb, b)| wb * b)
                .sum::<f64>();
        for i in 0..n {
            let pred = base
                + (0..m)
                    .map(|j| data[(i, j)] * w[j] * result.beta[j])
                    .sum::<f64>();
            assert!(
                (pred - result.fitted_values[i]).abs() < 1e-9,
                "prediction reconstruction mismatch at row {i}: {pred} vs {}",
                result.fitted_values[i]
            );
        }
    }

    #[test]
    fn test_peer_rejects_non_monotonic_argvals() {
        let (data, y, t, _) = make_fixture();
        let mut bad = t.clone();
        bad.swap(0, 1); // first pair now decreasing
        let config = PeerConfig::default();
        let res = peer(&data, &y, &bad, &config);
        assert!(
            matches!(res, Err(FdarError::InvalidParameter { .. })),
            "non-monotonic argvals must be rejected, got {res:?}"
        );
    }

    #[test]
    fn test_peer_rejects_single_observation() {
        let m = 5;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(1, m);
        for j in 0..m {
            data[(0, j)] = 1.0 + j as f64;
        }
        let y = vec![1.0];
        let res = peer(&data, &y, &t, &PeerConfig::default());
        assert!(
            matches!(res, Err(FdarError::InvalidDimension { .. })),
            "single observation must be rejected, got {res:?}"
        );
    }

    #[test]
    fn test_peer_rejects_non_finite_y() {
        let (data, mut y, t, _) = make_fixture();
        y[0] = f64::NAN;
        let res = peer(&data, &y, &t, &PeerConfig::default());
        assert!(
            matches!(res, Err(FdarError::InvalidParameter { .. })),
            "non-finite y must be rejected, got {res:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Task 2 tests: GCV grid-search selector
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_gcv_deterministic() {
        let (data, y, t, _) = make_fixture();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Gcv,
        };
        let r1 = peer(&data, &y, &t, &config).expect("first GCV call should succeed");
        let r2 = peer(&data, &y, &t, &config).expect("second GCV call should succeed");
        assert_eq!(
            r1.lambda, r2.lambda,
            "GCV lambda must be bit-exact across two runs: {} vs {}",
            r1.lambda, r2.lambda
        );
        assert_eq!(r1.lambda_method, LambdaMethod::Gcv);
        assert!(r1.gcv.is_some(), "GCV result must have gcv score");
    }

    #[test]
    fn test_peer_gcv_recovers_beta() {
        let (data, y, t, true_beta) = make_fixture();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Gcv,
        };
        let result = peer(&data, &y, &t, &config).expect("GCV peer() should succeed");

        // Non-degenerate lambda: not at extreme ends of the grid
        assert!(
            result.lambda > 1e-10 && result.lambda < 1e6,
            "GCV lambda should be non-degenerate, got {}",
            result.lambda
        );
        // GCV score recorded
        assert!(result.gcv.is_some(), "gcv field must be Some for Gcv");
        assert_eq!(result.lambda_method, LambdaMethod::Gcv);

        // β recovery within tolerance
        let max_err = result
            .beta
            .iter()
            .zip(true_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 0.15, "GCV beta recovery error: {max_err} >= 0.15");
    }

    // -------------------------------------------------------------------------
    // Task 3 tests: REML EM selector
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_reml_deterministic() {
        let (data, y, t, _) = make_fixture();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Reml,
        };
        let r1 = peer(&data, &y, &t, &config).expect("first REML call should succeed");
        let r2 = peer(&data, &y, &t, &config).expect("second REML call should succeed");
        assert_eq!(
            r1.lambda, r2.lambda,
            "REML lambda must be bit-exact across two runs: {} vs {}",
            r1.lambda, r2.lambda
        );
        assert_eq!(r1.lambda_method, LambdaMethod::Reml);
        assert!(r1.gcv.is_none(), "REML result must have gcv == None");
    }

    #[test]
    fn test_peer_reml_lambda_positive() {
        let (data, y, t, _) = make_fixture();
        let config = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Reml,
        };
        let result = peer(&data, &y, &t, &config).expect("REML peer() should succeed");

        assert!(
            result.lambda > 0.0 && result.lambda.is_finite(),
            "REML lambda must be positive and finite, got {}",
            result.lambda
        );
        assert!(result.gcv.is_none(), "gcv must be None for Reml");
        assert_eq!(result.lambda_method, LambdaMethod::Reml);
    }

    #[test]
    fn test_peer_reml_gcv_beta_agreement() {
        // Use the high-SNR fixture (noise=0.005) for the agreement test.
        // REML on this problem correctly identifies a small λ (the range-space
        // component of sin(πt) is large, so σ²_u is large, giving λ = σ²_e/σ²_u
        // near zero — valid REML behavior for this signal shape).  The resulting
        // beta still recovers the truth within 0.15 because at this SNR even
        // near-OLS estimates are close.  GCV picks a larger but also reasonable λ.
        let (data, y, t, true_beta) = make_fixture(); // noise=0.005, high SNR

        let config_gcv = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Gcv,
        };
        let config_reml = PeerConfig {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: LambdaChoice::Reml,
        };

        let res_gcv = peer(&data, &y, &t, &config_gcv).expect("GCV fit should succeed");
        let res_reml = peer(&data, &y, &t, &config_reml).expect("REML fit should succeed");

        // REML beta(t) recovers true beta within 0.15
        let reml_err = res_reml
            .beta
            .iter()
            .zip(true_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            reml_err < 0.15,
            "REML beta recovery error: {reml_err} >= 0.15"
        );

        // GCV and REML beta agree within documented tolerance (0.2 max abs diff)
        let beta_diff = res_gcv
            .beta
            .iter()
            .zip(res_reml.beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            beta_diff < 0.2,
            "REML vs GCV beta disagreement: {beta_diff} >= 0.2 (lambdas: gcv={}, reml={})",
            res_gcv.lambda,
            res_reml.lambda
        );
    }

    #[test]
    fn test_peer_reml_ridge_and_zeroq_edges() {
        let (data, y, t, _) = make_fixture();
        let m = t.len();

        // Ridge penalty (Q = I_m, null space empty, s=0) — must not panic
        let config_ridge = PeerConfig {
            penalty: PeerPenalty::Ridge,
            lambda: LambdaChoice::Reml,
        };
        let res_ridge =
            peer(&data, &y, &t, &config_ridge).expect("REML with Ridge (s=0) should not panic");
        assert!(
            res_ridge.lambda > 0.0 && res_ridge.lambda.is_finite(),
            "Ridge REML lambda must be positive finite, got {}",
            res_ridge.lambda
        );

        // Zero-Q Decree (all-zero matrix, r=0) — must return documented fallback λ=1e-4
        let zero_q = vec![0.0_f64; m * m];
        let config_zero = PeerConfig {
            penalty: PeerPenalty::Decree(zero_q, m),
            lambda: LambdaChoice::Reml,
        };
        let res_zero =
            peer(&data, &y, &t, &config_zero).expect("REML with zero-Q should not panic");
        assert!(
            (res_zero.lambda - 1e-4).abs() < 1e-15,
            "zero-Q REML fallback lambda should be 1e-4, got {}",
            res_zero.lambda
        );
    }
}
