//! 2D Function-on-Scalar Regression (FOSR).
//!
//! Extends the 1D function-on-scalar model to surface-valued functional
//! responses observed on a regular 2D grid:
//! ```text
//! Y_i(s,t) = beta_0(s,t) + sum_j z_{ij} beta_j(s,t) + epsilon_i(s,t)
//! ```
//!
//! The estimation uses a two-step approach:
//! 1. Pointwise OLS at each grid point to obtain raw coefficient surfaces.
//! 2. Tensor-product roughness penalty smoothing of each coefficient surface.
//!
//! # Methods
//!
//! - [`fosr_2d`]: Penalized 2D FOSR with anisotropic smoothing
//! - [`predict_fosr_2d`]: Predict new surfaces from a fitted model

use crate::error::FdarError;
use crate::function_on_scalar::{
    build_fosr_design, compute_xty_matrix, penalty_matrix, pointwise_r_squared,
};
use crate::linalg::{cholesky_factor, cholesky_forward_back, compute_xtx};
use crate::matrix::FdMatrix;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// 2D grid description for surface-valued functional data.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid2d {
    /// Grid points along the first (row) dimension.
    pub argvals_s: Vec<f64>,
    /// Grid points along the second (column) dimension.
    pub argvals_t: Vec<f64>,
}

impl Grid2d {
    /// Create a new 2D grid from argument vectors.
    pub fn new(argvals_s: Vec<f64>, argvals_t: Vec<f64>) -> Self {
        Self {
            argvals_s,
            argvals_t,
        }
    }

    /// Number of grid points in the first dimension.
    #[inline]
    pub fn m1(&self) -> usize {
        self.argvals_s.len()
    }

    /// Number of grid points in the second dimension.
    #[inline]
    pub fn m2(&self) -> usize {
        self.argvals_t.len()
    }

    /// Total number of grid points (m1 * m2).
    #[inline]
    pub fn total(&self) -> usize {
        self.m1() * self.m2()
    }
}

/// Result of 2D function-on-scalar regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FosrResult2d {
    /// Intercept surface beta_0(s,t), flattened column-major (length m1*m2).
    pub intercept: Vec<f64>,
    /// Coefficient surfaces beta_j(s,t), p x (m1*m2) matrix (row j = flattened beta_j).
    pub beta: FdMatrix,
    /// Fitted surface values, n x (m1*m2) matrix.
    pub fitted: FdMatrix,
    /// Residual surfaces, n x (m1*m2) matrix.
    pub residuals: FdMatrix,
    /// Pointwise R^2(s,t), flattened (length m1*m2).
    pub r_squared_pointwise: Vec<f64>,
    /// Global R^2 (average of pointwise).
    pub r_squared: f64,
    /// Standard error surfaces for each beta_j (p x (m1*m2)), or None if
    /// the system is underdetermined.
    pub beta_se: Option<FdMatrix>,
    /// Smoothing parameter in the s-direction.
    pub lambda_s: f64,
    /// Smoothing parameter in the t-direction.
    pub lambda_t: f64,
    /// Generalised cross-validation score.
    pub gcv: f64,
    /// Grid specification.
    pub grid: Grid2d,
}

impl FosrResult2d {
    /// Reshape the j-th coefficient surface into an m1 x m2 matrix.
    ///
    /// # Panics
    /// Panics if `j >= p` (number of predictors).
    #[must_use]
    pub fn beta_surface(&self, j: usize) -> FdMatrix {
        let m1 = self.grid.m1();
        let m2 = self.grid.m2();
        let m_total = m1 * m2;
        let mut mat = FdMatrix::zeros(m1, m2);
        for g in 0..m_total {
            // Column-major: g = s_idx + t_idx * m1
            let s_idx = g % m1;
            let t_idx = g / m1;
            mat[(s_idx, t_idx)] = self.beta[(j, g)];
        }
        mat
    }

    /// Reshape pointwise R^2(s,t) into an m1 x m2 matrix.
    #[must_use]
    pub fn r_squared_surface(&self) -> FdMatrix {
        let m1 = self.grid.m1();
        let m2 = self.grid.m2();
        let m_total = m1 * m2;
        let mut mat = FdMatrix::zeros(m1, m2);
        for g in 0..m_total {
            let s_idx = g % m1;
            let t_idx = g / m1;
            mat[(s_idx, t_idx)] = self.r_squared_pointwise[g];
        }
        mat
    }

    /// Reshape the residual for observation `i` into an m1 x m2 matrix.
    ///
    /// # Panics
    /// Panics if `i >= n` (number of observations).
    #[must_use]
    pub fn residual_surface(&self, i: usize) -> FdMatrix {
        let m1 = self.grid.m1();
        let m2 = self.grid.m2();
        let m_total = m1 * m2;
        let mut mat = FdMatrix::zeros(m1, m2);
        for g in 0..m_total {
            let s_idx = g % m1;
            let t_idx = g / m1;
            mat[(s_idx, t_idx)] = self.residuals[(i, g)];
        }
        mat
    }

    /// Predict functional surfaces for new predictors. Delegates to
    /// [`predict_fosr_2d`].
    pub fn predict(&self, new_predictors: &FdMatrix) -> Result<FdMatrix, FdarError> {
        predict_fosr_2d(self, new_predictors)
    }
}

/// Kronecker product of two flat row-major matrices.
///
/// Given A (rows_a x cols_a) and B (rows_b x cols_b), produces
/// C = A kron B of size (rows_a * rows_b) x (cols_a * cols_b) in
/// row-major layout.
fn kronecker_product(
    a: &[f64],
    rows_a: usize,
    cols_a: usize,
    b: &[f64],
    rows_b: usize,
    cols_b: usize,
) -> Vec<f64> {
    let out_rows = rows_a * rows_b;
    let out_cols = cols_a * cols_b;
    let mut c = vec![0.0; out_rows * out_cols];
    for ia in 0..rows_a {
        for ja in 0..cols_a {
            let a_val = a[ia * cols_a + ja];
            for ib in 0..rows_b {
                for jb in 0..cols_b {
                    let row = ia * rows_b + ib;
                    let col = ja * cols_b + jb;
                    c[row * out_cols + col] = a_val * b[ib * cols_b + jb];
                }
            }
        }
    }
    c
}

/// Identity matrix as flat row-major vector.
fn identity_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Build the 2D tensor-product penalty matrix.
///
/// P_2d = lambda_s * (P_s kron I_t) + lambda_t * (I_s kron P_t)
///
/// where P_s = D_s'D_s (m1 x m1) and P_t = D_t'D_t (m2 x m2) are the
/// second-difference penalty matrices.
fn penalty_matrix_2d(m1: usize, m2: usize, lambda_s: f64, lambda_t: f64) -> Vec<f64> {
    let m_total = m1 * m2;
    let ps = penalty_matrix(m1);
    let pt = penalty_matrix(m2);
    let i_t = identity_matrix(m2);
    let i_s = identity_matrix(m1);

    let ps_kron_it = kronecker_product(&ps, m1, m1, &i_t, m2, m2);
    let is_kron_pt = kronecker_product(&i_s, m1, m1, &pt, m2, m2);

    let mut p2d = vec![0.0; m_total * m_total];
    for i in 0..m_total * m_total {
        p2d[i] = lambda_s * ps_kron_it[i] + lambda_t * is_kron_pt[i];
    }
    p2d
}

// ---------------------------------------------------------------------------
// Core fitting routines
// ---------------------------------------------------------------------------

/// Compute fitted values Y_hat = X * beta and residuals Y - Y_hat.
fn compute_fitted_residuals(
    design: &FdMatrix,
    beta: &FdMatrix,
    data: &FdMatrix,
) -> (FdMatrix, FdMatrix) {
    let (n, m_total) = data.shape();
    let p_total = design.ncols();
    let mut fitted = FdMatrix::zeros(n, m_total);
    let mut residuals = FdMatrix::zeros(n, m_total);
    for i in 0..n {
        for g in 0..m_total {
            let mut yhat = 0.0;
            for j in 0..p_total {
                yhat += design[(i, j)] * beta[(j, g)];
            }
            fitted[(i, g)] = yhat;
            residuals[(i, g)] = data[(i, g)] - yhat;
        }
    }
    (fitted, residuals)
}

/// Compute GCV: (1/n*m) * sum(r^2) / (1 - tr(H)/n)^2.
fn compute_gcv(residuals: &FdMatrix, trace_h: f64) -> f64 {
    let (n, m) = residuals.shape();
    let denom = (1.0 - trace_h / n as f64).max(1e-10);
    let ss_res: f64 = residuals.as_slice().iter().map(|v| v * v).sum();
    ss_res / (n as f64 * m as f64 * denom * denom)
}

/// Compute trace of hat matrix for OLS: tr(H) = p_total (since X(X'X)^{-1}X'
/// projects onto a p_total-dimensional subspace).
fn trace_hat_ols(p_total: usize) -> f64 {
    p_total as f64
}

/// Smooth a raw coefficient vector (length m_total) using the 2D penalty.
///
/// Solves (I + P_2d) * beta_smooth = beta_raw via Cholesky.
fn smooth_coefficient_surface(
    beta_raw: &[f64],
    penalty_2d: &[f64],
    m_total: usize,
) -> Result<Vec<f64>, FdarError> {
    // Build (I + P_2d)
    let mut a = penalty_2d.to_vec();
    for i in 0..m_total {
        a[i * m_total + i] += 1.0;
    }
    let l = cholesky_factor(&a, m_total)?;
    Ok(cholesky_forward_back(&l, beta_raw, m_total))
}

/// Compute standard errors from the diagonal of (X'X)^{-1} and residual
/// variance at each grid point.
fn compute_beta_se_2d(
    xtx: &[f64],
    residuals: &FdMatrix,
    p_total: usize,
    n: usize,
) -> Option<FdMatrix> {
    let m_total = residuals.ncols();
    let l = cholesky_factor(xtx, p_total).ok()?;

    // Diagonal of (X'X)^{-1}
    let a_inv_diag: Vec<f64> = (0..p_total)
        .map(|j| {
            let mut ej = vec![0.0; p_total];
            ej[j] = 1.0;
            let v = cholesky_forward_back(&l, &ej, p_total);
            v[j]
        })
        .collect();

    let df = (n - p_total).max(1) as f64;
    // We return SE for the predictor coefficients only (drop intercept).
    let p = p_total - 1;
    let mut se = FdMatrix::zeros(p, m_total);
    for g in 0..m_total {
        let sigma2: f64 = (0..n).map(|i| residuals[(i, g)].powi(2)).sum::<f64>() / df;
        for j in 0..p {
            // j+1 to skip intercept row in a_inv_diag
            se[(j, g)] = (sigma2 * a_inv_diag[j + 1]).max(0.0).sqrt();
        }
    }
    Some(se)
}

// ---------------------------------------------------------------------------
// GCV lambda selection
// ---------------------------------------------------------------------------

/// Select (lambda_s, lambda_t) via GCV over a 2D grid of candidate values.
fn select_lambdas_gcv(
    xtx: &[f64],
    xty: &FdMatrix,
    design: &FdMatrix,
    data: &FdMatrix,
    m1: usize,
    m2: usize,
    fix_lambda_s: Option<f64>,
    fix_lambda_t: Option<f64>,
) -> (f64, f64) {
    let candidates = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0];
    let p_total = design.ncols();
    let m_total = m1 * m2;

    let ls_candidates: Vec<f64> = if let Some(ls) = fix_lambda_s {
        vec![ls]
    } else {
        candidates.to_vec()
    };
    let lt_candidates: Vec<f64> = if let Some(lt) = fix_lambda_t {
        vec![lt]
    } else {
        candidates.to_vec()
    };

    // Pre-compute OLS inverse and raw beta
    let Ok(l_xtx) = cholesky_factor(xtx, p_total) else {
        return (0.0, 0.0);
    };

    // beta_ols: p_total x m_total
    let mut beta_ols = FdMatrix::zeros(p_total, m_total);
    for g in 0..m_total {
        let b: Vec<f64> = (0..p_total).map(|j| xty[(j, g)]).collect();
        let x = cholesky_forward_back(&l_xtx, &b, p_total);
        for j in 0..p_total {
            beta_ols[(j, g)] = x[j];
        }
    }

    let trace_h = trace_hat_ols(p_total);
    let mut best_ls = 0.0;
    let mut best_lt = 0.0;
    let mut best_gcv = f64::INFINITY;

    for &ls in &ls_candidates {
        for &lt in &lt_candidates {
            if ls == 0.0 && lt == 0.0 {
                // No smoothing: use raw OLS
                let (_, residuals) = compute_fitted_residuals(design, &beta_ols, data);
                let gcv = compute_gcv(&residuals, trace_h);
                if gcv < best_gcv {
                    best_gcv = gcv;
                    best_ls = ls;
                    best_lt = lt;
                }
                continue;
            }

            let p2d = penalty_matrix_2d(m1, m2, ls, lt);
            let mut beta_smooth = FdMatrix::zeros(p_total, m_total);
            let mut ok = true;
            for j in 0..p_total {
                let raw: Vec<f64> = (0..m_total).map(|g| beta_ols[(j, g)]).collect();
                if let Ok(smoothed) = smooth_coefficient_surface(&raw, &p2d, m_total) {
                    for g in 0..m_total {
                        beta_smooth[(j, g)] = smoothed[g];
                    }
                } else {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }

            let (_, residuals) = compute_fitted_residuals(design, &beta_smooth, data);
            let gcv = compute_gcv(&residuals, trace_h);
            if gcv < best_gcv {
                best_gcv = gcv;
                best_ls = ls;
                best_lt = lt;
            }
        }
    }

    (best_ls, best_lt)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 2D Function-on-Scalar Regression with tensor-product penalty.
///
/// Fits the model
/// ```text
/// Y_i(s,t) = beta_0(s,t) + sum_j x_{ij} beta_j(s,t) + epsilon_i(s,t)
/// ```
/// with anisotropic roughness penalty
/// `lambda_s * ||d^2 beta/ds^2||^2 + lambda_t * ||d^2 beta/dt^2||^2`.
///
/// # Arguments
/// * `data` - Functional response matrix (n x m_total where m_total = m1*m2)
/// * `predictors` - Scalar predictor matrix (n x p)
/// * `grid` - 2D grid specification
/// * `lambda_s` - Smoothing parameter in the s-direction (negative = GCV)
/// * `lambda_t` - Smoothing parameter in the t-direction (negative = GCV)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data.ncols() != grid.total()`,
/// `predictors.nrows() != data.nrows()`, or `n < p + 2`.
/// Returns [`FdarError::InvalidParameter`] if the grid has zero size in
/// either dimension.
/// Returns [`FdarError::ComputationFailed`] if the Cholesky factorization
/// fails during OLS or smoothing.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::function_on_scalar_2d::{fosr_2d, Grid2d};
///
/// let m1 = 5;
/// let m2 = 4;
/// let n = 20;
/// let grid = Grid2d::new(
///     (0..m1).map(|i| i as f64 / (m1 - 1) as f64).collect(),
///     (0..m2).map(|i| i as f64 / (m2 - 1) as f64).collect(),
/// );
/// let data = FdMatrix::from_column_major(
///     (0..n * m1 * m2).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * 0.3 + j * 0.7).sin()
///     }).collect(),
///     n, m1 * m2,
/// ).unwrap();
/// let predictors = FdMatrix::from_column_major(
///     (0..n * 2).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         (i * 0.4 + j * 1.5).cos()
///     }).collect(),
///     n, 2,
/// ).unwrap();
/// let result = fosr_2d(&data, &predictors, &grid, 0.1, 0.1).unwrap();
/// assert_eq!(result.fitted.shape(), (n, m1 * m2));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fosr_2d(
    data: &FdMatrix,
    predictors: &FdMatrix,
    grid: &Grid2d,
    lambda_s: f64,
    lambda_t: f64,
) -> Result<FosrResult2d, FdarError> {
    let (n, m_data) = data.shape();
    let p = predictors.ncols();
    let m1 = grid.m1();
    let m2 = grid.m2();
    let m_total = grid.total();

    // ---- Validate inputs ----

    if m1 == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "grid",
            message: "argvals_s must not be empty".to_string(),
        });
    }
    if m2 == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "grid",
            message: "argvals_t must not be empty".to_string(),
        });
    }
    if m_data != m_total {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{m_total} columns (grid m1*m2 = {m1}*{m2})"),
            actual: format!("{m_data} columns"),
        });
    }
    if predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: format!("{n} rows (matching data)"),
            actual: format!("{} rows", predictors.nrows()),
        });
    }
    if n < p + 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("at least {} observations (p + 2)", p + 2),
            actual: format!("{n} observations"),
        });
    }

    // ---- Step 1: Build design and compute OLS ----

    let design = build_fosr_design(predictors, n);
    let p_total = design.ncols(); // p + 1
    let xtx = compute_xtx(&design);
    let xty = compute_xty_matrix(&design, data);

    let l_xtx = cholesky_factor(&xtx, p_total)?;

    // Pointwise OLS: beta_ols[:,g] = (X'X)^{-1} X' y[:,g]
    let mut beta_ols = FdMatrix::zeros(p_total, m_total);
    for g in 0..m_total {
        let b: Vec<f64> = (0..p_total).map(|j| xty[(j, g)]).collect();
        let x = cholesky_forward_back(&l_xtx, &b, p_total);
        for j in 0..p_total {
            beta_ols[(j, g)] = x[j];
        }
    }

    // ---- Step 2: Determine lambda values ----

    let fix_ls = if lambda_s >= 0.0 {
        Some(lambda_s)
    } else {
        None
    };
    let fix_lt = if lambda_t >= 0.0 {
        Some(lambda_t)
    } else {
        None
    };

    let (lambda_s_final, lambda_t_final) = if fix_ls.is_some() && fix_lt.is_some() {
        (lambda_s, lambda_t)
    } else {
        select_lambdas_gcv(&xtx, &xty, &design, data, m1, m2, fix_ls, fix_lt)
    };

    // ---- Step 3: Smooth coefficient surfaces ----

    let beta_smooth = if lambda_s_final == 0.0 && lambda_t_final == 0.0 {
        beta_ols
    } else {
        let p2d = penalty_matrix_2d(m1, m2, lambda_s_final, lambda_t_final);
        let mut smoothed = FdMatrix::zeros(p_total, m_total);
        for j in 0..p_total {
            let raw: Vec<f64> = (0..m_total).map(|g| beta_ols[(j, g)]).collect();
            let s = smooth_coefficient_surface(&raw, &p2d, m_total)?;
            for g in 0..m_total {
                smoothed[(j, g)] = s[g];
            }
        }
        smoothed
    };

    // ---- Step 4: Compute diagnostics ----

    let (fitted, residuals) = compute_fitted_residuals(&design, &beta_smooth, data);

    let r_squared_pointwise = pointwise_r_squared(data, &fitted);
    let r_squared = if m_total > 0 {
        r_squared_pointwise.iter().sum::<f64>() / m_total as f64
    } else {
        0.0
    };

    let trace_h = trace_hat_ols(p_total);
    let gcv = compute_gcv(&residuals, trace_h);

    let beta_se = compute_beta_se_2d(&xtx, &residuals, p_total, n);

    // Extract intercept (row 0 of beta_smooth)
    let intercept: Vec<f64> = (0..m_total).map(|g| beta_smooth[(0, g)]).collect();

    // Extract predictor coefficients (rows 1..p_total)
    let mut beta_out = FdMatrix::zeros(p, m_total);
    for j in 0..p {
        for g in 0..m_total {
            beta_out[(j, g)] = beta_smooth[(j + 1, g)];
        }
    }

    Ok(FosrResult2d {
        intercept,
        beta: beta_out,
        fitted,
        residuals,
        r_squared_pointwise,
        r_squared,
        beta_se,
        lambda_s: lambda_s_final,
        lambda_t: lambda_t_final,
        gcv,
        grid: grid.clone(),
    })
}

/// Predict functional surfaces for new observations.
///
/// # Arguments
/// * `result` - Fitted [`FosrResult2d`]
/// * `new_predictors` - New scalar predictors (n_new x p)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the number of predictor columns
/// does not match the fitted model.
#[must_use = "prediction result should not be discarded"]
pub fn predict_fosr_2d(
    result: &FosrResult2d,
    new_predictors: &FdMatrix,
) -> Result<FdMatrix, FdarError> {
    let n_new = new_predictors.nrows();
    let m_total = result.intercept.len();
    let p = result.beta.nrows();

    if new_predictors.ncols() != p {
        return Err(FdarError::InvalidDimension {
            parameter: "new_predictors",
            expected: format!("{p} columns (matching fitted model)"),
            actual: format!("{} columns", new_predictors.ncols()),
        });
    }

    let mut predicted = FdMatrix::zeros(n_new, m_total);
    for i in 0..n_new {
        for g in 0..m_total {
            let mut yhat = result.intercept[g];
            for j in 0..p {
                yhat += new_predictors[(i, j)] * result.beta[(j, g)];
            }
            predicted[(i, g)] = yhat;
        }
    }
    Ok(predicted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid_1d(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect()
    }

    fn make_grid(m1: usize, m2: usize) -> Grid2d {
        Grid2d::new(uniform_grid_1d(m1), uniform_grid_1d(m2))
    }

    /// Generate test data: Y_i(s,t) = intercept(s,t) + z_1 * beta_1(s,t) + z_2 * beta_2(s,t) + noise
    fn generate_2d_data(
        n: usize,
        m1: usize,
        m2: usize,
        noise_scale: f64,
    ) -> (FdMatrix, FdMatrix, Grid2d) {
        let grid = make_grid(m1, m2);
        let m_total = m1 * m2;
        let mut y = FdMatrix::zeros(n, m_total);
        let mut z = FdMatrix::zeros(n, 2);

        for i in 0..n {
            let z1 = (i as f64) / (n as f64);
            let z2 = if i % 2 == 0 { 1.0 } else { 0.0 };
            z[(i, 0)] = z1;
            z[(i, 1)] = z2;

            for si in 0..m1 {
                for ti in 0..m2 {
                    let g = si + ti * m1; // column-major flat index
                    let s = grid.argvals_s[si];
                    let t = grid.argvals_t[ti];

                    let intercept = s + t;
                    let beta1 = s * t;
                    let beta2 = s - t;
                    let noise = noise_scale * ((i * 13 + si * 7 + ti * 3) % 100) as f64 / 100.0;

                    y[(i, g)] = intercept + z1 * beta1 + z2 * beta2 + noise;
                }
            }
        }
        (y, z, grid)
    }

    #[test]
    fn test_grid2d_basic() {
        let grid = make_grid(5, 4);
        assert_eq!(grid.m1(), 5);
        assert_eq!(grid.m2(), 4);
        assert_eq!(grid.total(), 20);
    }

    #[test]
    fn test_kronecker_product_small() {
        // A = [[1,2],[3,4]], B = [[0,5],[6,7]]
        // A kron B = [[0,5,0,10],[6,7,12,14],[0,15,0,20],[18,21,24,28]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 5.0, 6.0, 7.0];
        let c = kronecker_product(&a, 2, 2, &b, 2, 2);
        assert_eq!(c.len(), 16);
        #[rustfmt::skip]
        let expected = vec![
            0.0, 5.0, 0.0, 10.0,
            6.0, 7.0, 12.0, 14.0,
            0.0, 15.0, 0.0, 20.0,
            18.0, 21.0, 24.0, 28.0,
        ];
        for (i, (&ci, &ei)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (ci - ei).abs() < 1e-12,
                "kronecker mismatch at index {i}: got {ci}, expected {ei}"
            );
        }
    }

    #[test]
    fn test_penalty_2d_symmetry() {
        let m1 = 5;
        let m2 = 4;
        let p2d = penalty_matrix_2d(m1, m2, 1.0, 1.0);
        let m_total = m1 * m2;
        for i in 0..m_total {
            for j in 0..m_total {
                assert!(
                    (p2d[i * m_total + j] - p2d[j * m_total + i]).abs() < 1e-12,
                    "P_2d not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_penalty_2d_shape() {
        let m1 = 5;
        let m2 = 4;
        let p2d = penalty_matrix_2d(m1, m2, 1.0, 2.0);
        assert_eq!(p2d.len(), (m1 * m2) * (m1 * m2));
    }

    #[test]
    fn test_fosr_2d_constant_response() {
        let n = 20;
        let m1 = 5;
        let m2 = 4;
        let grid = make_grid(m1, m2);
        let m_total = m1 * m2;

        // Y_i(s,t) = 3.0 for all i,s,t
        let mut y = FdMatrix::zeros(n, m_total);
        for i in 0..n {
            for g in 0..m_total {
                y[(i, g)] = 3.0;
            }
        }

        let mut z = FdMatrix::zeros(n, 2);
        for i in 0..n {
            z[(i, 0)] = i as f64;
            z[(i, 1)] = (i % 3) as f64;
        }

        let result = fosr_2d(&y, &z, &grid, 0.0, 0.0).unwrap();

        // Intercept should be near 3.0 everywhere
        for g in 0..m_total {
            assert!(
                (result.intercept[g] - 3.0).abs() < 1e-8,
                "intercept[{g}] = {}, expected 3.0",
                result.intercept[g]
            );
        }

        // Beta coefficients should be near zero
        for j in 0..2 {
            for g in 0..m_total {
                assert!(
                    result.beta[(j, g)].abs() < 1e-8,
                    "beta[{j},{g}] = {}, expected ~0",
                    result.beta[(j, g)]
                );
            }
        }
    }

    #[test]
    fn test_fosr_2d_single_predictor() {
        let (y, z, grid) = generate_2d_data(20, 5, 4, 0.01);
        let result = fosr_2d(&y, &z, &grid, 0.0, 0.0).unwrap();

        // With low noise, R^2 should be high
        assert!(
            result.r_squared > 0.8,
            "R^2 = {}, expected > 0.8",
            result.r_squared
        );
    }

    #[test]
    fn test_fosr_2d_fitted_plus_residuals() {
        let (y, z, grid) = generate_2d_data(20, 5, 4, 0.05);
        let result = fosr_2d(&y, &z, &grid, 0.0, 0.0).unwrap();

        let (n, m_total) = y.shape();
        for i in 0..n {
            for g in 0..m_total {
                let reconstructed = result.fitted[(i, g)] + result.residuals[(i, g)];
                assert!(
                    (reconstructed - y[(i, g)]).abs() < 1e-10,
                    "fitted + residuals != y at ({i},{g})"
                );
            }
        }
    }

    #[test]
    fn test_fosr_2d_r_squared_range() {
        let (y, z, grid) = generate_2d_data(20, 5, 4, 0.05);
        let result = fosr_2d(&y, &z, &grid, 0.0, 0.0).unwrap();

        for (g, &r2) in result.r_squared_pointwise.iter().enumerate() {
            assert!(
                (-0.01..=1.0 + 1e-10).contains(&r2),
                "R^2 out of range at grid point {g}: {r2}"
            );
        }
    }

    #[test]
    fn test_fosr_2d_predict_matches_fitted() {
        let (y, z, grid) = generate_2d_data(20, 5, 4, 0.05);
        let result = fosr_2d(&y, &z, &grid, 0.0, 0.0).unwrap();

        let preds = predict_fosr_2d(&result, &z).unwrap();
        let (n, m_total) = y.shape();
        for i in 0..n {
            for g in 0..m_total {
                assert!(
                    (preds[(i, g)] - result.fitted[(i, g)]).abs() < 1e-8,
                    "prediction != fitted at ({i},{g})"
                );
            }
        }
    }

    #[test]
    fn test_fosr_2d_reshape_beta_surface() {
        let (y, z, grid) = generate_2d_data(20, 5, 4, 0.05);
        let result = fosr_2d(&y, &z, &grid, 0.0, 0.0).unwrap();

        let surface = result.beta_surface(0);
        assert_eq!(surface.shape(), (5, 4));

        let r2_surface = result.r_squared_surface();
        assert_eq!(r2_surface.shape(), (5, 4));

        let resid_surface = result.residual_surface(0);
        assert_eq!(resid_surface.shape(), (5, 4));
    }

    #[test]
    fn test_fosr_2d_dimension_mismatch() {
        let grid = make_grid(5, 4);

        // Wrong number of columns in data
        let y = FdMatrix::zeros(20, 10); // 10 != 5*4=20
        let z = FdMatrix::zeros(20, 2);
        assert!(fosr_2d(&y, &z, &grid, 0.0, 0.0).is_err());

        // Mismatched rows between data and predictors
        let y = FdMatrix::zeros(20, 20);
        let z = FdMatrix::zeros(10, 2);
        assert!(fosr_2d(&y, &z, &grid, 0.0, 0.0).is_err());

        // Too few observations
        let y = FdMatrix::zeros(3, 20);
        let z = FdMatrix::zeros(3, 2);
        assert!(fosr_2d(&y, &z, &grid, 0.0, 0.0).is_err());

        // Empty grid
        let empty_grid = Grid2d::new(vec![], vec![0.0, 1.0]);
        let y = FdMatrix::zeros(20, 0);
        let z = FdMatrix::zeros(20, 2);
        assert!(fosr_2d(&y, &z, &empty_grid, 0.0, 0.0).is_err());

        // Predictor dimension mismatch in predict
        let grid = make_grid(3, 3);
        let y = FdMatrix::zeros(20, 9);
        let mut z = FdMatrix::zeros(20, 2);
        for i in 0..20 {
            z[(i, 0)] = i as f64;
            z[(i, 1)] = (i * 3 % 7) as f64;
        }
        let result = fosr_2d(&y, &z, &grid, 0.0, 0.0).unwrap();
        let z_bad = FdMatrix::zeros(5, 3); // 3 != 2
        assert!(predict_fosr_2d(&result, &z_bad).is_err());
    }

    #[test]
    fn test_fosr_2d_gcv() {
        let (y, z, grid) = generate_2d_data(20, 5, 4, 0.05);
        // Negative lambda triggers GCV selection
        let result = fosr_2d(&y, &z, &grid, -1.0, -1.0).unwrap();
        assert!(result.lambda_s >= 0.0);
        assert!(result.lambda_t >= 0.0);
        assert!(result.gcv > 0.0);
        assert!(result.r_squared > 0.5);
    }
}
