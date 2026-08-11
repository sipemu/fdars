//! Helper functions for numerical integration and common operations.

/// Small epsilon for numerical comparisons (e.g., avoiding division by zero).
pub const NUMERICAL_EPS: f64 = 1e-10;

/// Default convergence tolerance for iterative algorithms.
pub const DEFAULT_CONVERGENCE_TOL: f64 = 1e-6;

/// Sort a slice using total ordering that treats NaN as equal.
pub fn sort_nan_safe(slice: &mut [f64]) {
    slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Extract curves from column-major data matrix.
///
/// Converts a flat column-major matrix into a vector of curve vectors,
/// where each curve contains all evaluation points for one observation.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
///
/// # Returns
/// Vector of n curves, each containing m values
pub fn extract_curves(data: &crate::matrix::FdMatrix) -> Vec<Vec<f64>> {
    data.rows()
}

/// Compute L2 distance between two curves using integration weights.
///
/// # Arguments
/// * `curve1` - First curve values
/// * `curve2` - Second curve values
/// * `weights` - Integration weights
///
/// # Returns
/// L2 distance between the curves
pub fn l2_distance(curve1: &[f64], curve2: &[f64], weights: &[f64]) -> f64 {
    let mut dist_sq = 0.0;
    for i in 0..curve1.len() {
        let diff = curve1[i] - curve2[i];
        dist_sq += diff * diff * weights[i];
    }
    dist_sq.sqrt()
}

/// Compute Simpson's 1/3 rule integration weights for a grid.
///
/// For odd n (even number of intervals): standard composite Simpson's 1/3 rule.
/// For even n: Simpson's 1/3 for first n-1 points, trapezoidal for last interval.
/// For non-uniform grids: generalized Simpson's weights per sub-interval pair.
///
/// # Arguments
/// * `argvals` - Grid points (evaluation points)
///
/// # Returns
/// Vector of integration weights
pub fn simpsons_weights(argvals: &[f64]) -> Vec<f64> {
    let n = argvals.len();
    if n < 2 {
        return vec![1.0; n];
    }

    let mut weights = vec![0.0; n];

    if n == 2 {
        // Trapezoidal rule
        let h = argvals[1] - argvals[0];
        weights[0] = h / 2.0;
        weights[1] = h / 2.0;
        return weights;
    }

    // Check if grid is uniform
    let h0 = argvals[1] - argvals[0];
    let is_uniform = argvals
        .windows(2)
        .all(|w| ((w[1] - w[0]) - h0).abs() < 1e-12 * h0.abs());

    if is_uniform {
        simpsons_weights_uniform(&mut weights, n, h0);
    } else {
        simpsons_weights_nonuniform(&mut weights, argvals, n);
    }

    weights
}

/// Uniform grid Simpson's 1/3 weights.
fn simpsons_weights_uniform(weights: &mut [f64], n: usize, h0: f64) {
    let n_intervals = n - 1;
    if n_intervals % 2 == 0 {
        // Even number of intervals (odd n): pure Simpson's
        weights[0] = h0 / 3.0;
        weights[n - 1] = h0 / 3.0;
        for i in 1..n - 1 {
            weights[i] = if i % 2 == 1 {
                4.0 * h0 / 3.0
            } else {
                2.0 * h0 / 3.0
            };
        }
    } else {
        // Odd number of intervals (even n): Simpson's + trapezoidal for last
        let n_simp = n - 1;
        weights[0] = h0 / 3.0;
        weights[n_simp - 1] = h0 / 3.0;
        for i in 1..n_simp - 1 {
            weights[i] = if i % 2 == 1 {
                4.0 * h0 / 3.0
            } else {
                2.0 * h0 / 3.0
            };
        }
        weights[n_simp - 1] += h0 / 2.0;
        weights[n - 1] += h0 / 2.0;
    }
}

/// Non-uniform grid generalized Simpson's weights.
fn simpsons_weights_nonuniform(weights: &mut [f64], argvals: &[f64], n: usize) {
    let n_intervals = n - 1;
    let n_pairs = n_intervals / 2;

    for k in 0..n_pairs {
        let i0 = 2 * k;
        let i1 = i0 + 1;
        let i2 = i0 + 2;
        let h1 = argvals[i1] - argvals[i0];
        let h2 = argvals[i2] - argvals[i1];
        let h_sum = h1 + h2;

        weights[i0] += (2.0 * h1 - h2) * h_sum / (6.0 * h1);
        weights[i1] += h_sum * h_sum * h_sum / (6.0 * h1 * h2);
        weights[i2] += (2.0 * h2 - h1) * h_sum / (6.0 * h2);
    }

    if n_intervals % 2 == 1 {
        let h_last = argvals[n - 1] - argvals[n - 2];
        weights[n - 2] += h_last / 2.0;
        weights[n - 1] += h_last / 2.0;
    }
}

/// Compute 2D integration weights using tensor product of 1D weights.
///
/// Returns a flattened vector of weights for an m1 x m2 grid.
///
/// # Arguments
/// * `argvals_s` - Grid points in s direction
/// * `argvals_t` - Grid points in t direction
///
/// # Returns
/// Flattened vector of integration weights (column-major: s-varies-fastest, matching FdMatrix surface layout)
pub fn simpsons_weights_2d(argvals_s: &[f64], argvals_t: &[f64]) -> Vec<f64> {
    let weights_s = simpsons_weights(argvals_s);
    let weights_t = simpsons_weights(argvals_t);
    let m1 = argvals_s.len();
    let m2 = argvals_t.len();

    let mut weights = vec![0.0; m1 * m2];
    for i in 0..m1 {
        for j in 0..m2 {
            weights[i + j * m1] = weights_s[i] * weights_t[j];
        }
    }
    weights
}

/// Linear interpolation at point `t` using binary search.
///
/// Clamps to boundary values outside the domain of `x`.
pub fn linear_interp(x: &[f64], y: &[f64], t: f64) -> f64 {
    if t <= x[0] {
        return y[0];
    }
    let last = x.len() - 1;
    if t >= x[last] {
        return y[last];
    }

    let idx = match x.binary_search_by(|v| v.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => return y[i],
        Err(i) => i,
    };

    let t0 = x[idx - 1];
    let t1 = x[idx];
    let y0 = y[idx - 1];
    let y1 = y[idx];
    y0 + (y1 - y0) * (t - t0) / (t1 - t0)
}

/// Cumulative integration using Simpson's rule where possible.
///
/// For pairs of intervals uses Simpson's 1/3 rule for higher accuracy.
/// Falls back to trapezoidal for the last interval if n is even.
pub fn cumulative_trapz(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    if n < 2 {
        return out;
    }

    // Process pairs of intervals with Simpson's rule
    let mut k = 1;
    while k + 1 < n {
        let h1 = x[k] - x[k - 1];
        let h2 = x[k + 1] - x[k];
        let h_sum = h1 + h2;

        // Generalized Simpson's for this pair of intervals
        let integral = h_sum / 6.0
            * (y[k - 1] * (2.0 * h1 - h2) / h1
                + y[k] * h_sum * h_sum / (h1 * h2)
                + y[k + 1] * (2.0 * h2 - h1) / h2);

        out[k] = out[k - 1] + {
            // First sub-interval: use trapezoidal for the intermediate value
            0.5 * (y[k] + y[k - 1]) * h1
        };
        out[k + 1] = out[k - 1] + integral;
        k += 2;
    }

    // If there's a remaining interval, use trapezoidal
    if k < n {
        out[k] = out[k - 1] + 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
    }

    out
}

/// Trapezoidal integration of `y` over `x`.
pub fn trapz(y: &[f64], x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for k in 1..y.len() {
        sum += 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
    }
    sum
}

/// Gaussian kernel: K(d, h) = exp(-d² / (2h²)).
///
/// This is the un-normalized version used by Nadaraya-Watson regression
/// and kernel classification. For density estimation with normalization,
/// see the smoothing module.
pub fn gaussian_kernel(d: f64, h: f64) -> f64 {
    if h < 1e-15 {
        return 0.0;
    }
    (-d * d / (2.0 * h * h)).exp()
}

/// Extract bandwidth candidates from a flat n×n distance matrix.
///
/// Collects the upper-triangle nonzero distances, sorts them, and returns
/// `n_quantiles` evenly-spaced quantile values. Used for LOO-CV bandwidth
/// grid search in kernel regression and classification.
pub fn bandwidth_candidates_from_dists(dists: &[f64], n: usize, n_quantiles: usize) -> Vec<f64> {
    let mut nonzero: Vec<f64> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| dists[i * n + j]))
        .filter(|&d| d > 0.0)
        .collect();
    sort_nan_safe(&mut nonzero);

    if nonzero.is_empty() {
        return Vec::new();
    }

    (1..=n_quantiles)
        .map(|q| {
            let p = q as f64 / (n_quantiles + 1) as f64;
            let idx = ((nonzero.len() as f64 * p) as usize).min(nonzero.len() - 1);
            nonzero[idx]
        })
        .filter(|&h| h > 1e-15)
        .collect()
}

/// Compute a quantile from a sorted slice.
///
/// `p` should be in [0, 1]. Uses linear interpolation between adjacent values.
pub fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 || p <= 0.0 {
        return sorted[0];
    }
    if p >= 1.0 {
        return sorted[sorted.len() - 1];
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute R² (coefficient of determination).
pub fn r_squared(y_true: &[f64], residuals: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0 {
        return f64::NAN;
    }
    let mean = y_true.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y_true.iter().map(|&y| (y - mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    if ss_tot > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Compute adjusted R².
pub fn r_squared_adj(y_true: &[f64], residuals: &[f64], p: usize) -> f64 {
    let n = y_true.len();
    let r2 = r_squared(y_true, residuals);
    if n <= p + 1 {
        return r2;
    }
    1.0 - (1.0 - r2) * (n - 1) as f64 / (n - p - 1) as f64
}

/// Compute AIC from residual sum of squares.
///
/// AIC = n * ln(RSS/n) + 2p
pub fn aic(n: usize, rss: f64, p: usize) -> f64 {
    let nf = n as f64;
    nf * (rss / nf).ln() + 2.0 * p as f64
}

/// Compute BIC from residual sum of squares.
///
/// BIC = n * ln(RSS/n) + ln(n) * p
pub fn bic(n: usize, rss: f64, p: usize) -> f64 {
    let nf = n as f64;
    nf * (rss / nf).ln() + nf.ln() * p as f64
}

/// Interpolation method for resampling functional data.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent points.
    Linear,
    /// Cubic Hermite interpolation (monotone, C1 continuous).
    CubicHermite,
}

/// Interpolate functional data to a new grid.
///
/// Resamples each curve from `data` evaluated at `argvals` to the new
/// evaluation points `new_argvals`.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Original evaluation points (length m, must be sorted)
/// * `new_argvals` - New evaluation points (length m_new, must be sorted, within original domain)
/// * `method` - Interpolation method
///
/// # Returns
/// Interpolated matrix (n x m_new)
#[must_use]
pub fn fdata_interpolate(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    new_argvals: &[f64],
    method: InterpolationMethod,
) -> crate::matrix::FdMatrix {
    let (n, m) = data.shape();
    let m_new = new_argvals.len();
    if n == 0 || m < 2 || m_new == 0 {
        return crate::matrix::FdMatrix::zeros(n.max(1), m_new.max(1));
    }

    let mut result = crate::matrix::FdMatrix::zeros(n, m_new);

    for i in 0..n {
        let y: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        for (j, &t) in new_argvals.iter().enumerate() {
            result[(i, j)] = match method {
                InterpolationMethod::Linear => linear_interp(argvals, &y, t),
                InterpolationMethod::CubicHermite => cubic_hermite_interp(argvals, &y, t),
            };
        }
    }

    result
}

/// Fit an order-k B-spline interpolant per curve and evaluate at arbitrary query points.
///
/// For each curve in `data` (sampled at `argvals`), fits a B-spline of the given `order`
/// using least-squares via SVD pseudoinverse, then evaluates at `query_points` using the
/// same knot vector. Returns a new `FdMatrix` with shape `(n, query_points.len())`.
///
/// Uses the fit-then-evaluate pattern from the existing B-spline basis system
/// (`basis::bspline`), without P-spline smoothing — this is interpolation, not smoothing.
///
/// # Arguments
/// * `data`         — Functional data matrix (`n × m`)
/// * `argvals`      — Original evaluation points (`length m`, must be sorted)
/// * `query_points` — Points to evaluate at (must lie within `[argvals[0], argvals[m-1]]`)
/// * `order`        — B-spline order (1 = linear, 2 = quadratic, 4 = cubic, …); must be in `[1, m)`
///
/// # Returns
/// Interpolated `FdMatrix` of shape `(n, query_points.len())`
///
/// # Errors
/// * `FdarError::InvalidDimension` — `argvals.len() != data.ncols()` or `query_points` is empty
/// * `FdarError::InvalidParameter` — `order` is 0 or ≥ `m`, or any query point is outside
///   `[argvals[0], argvals[m-1]]`
/// * `FdarError::ComputationFailed` — SVD pseudoinverse could not be computed
pub fn spline_interpolate(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    query_points: &[f64],
    order: usize,
) -> Result<crate::matrix::FdMatrix, crate::FdarError> {
    let (n, m) = data.shape();

    // --- Input validation ---
    if argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if query_points.is_empty() {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "query_points",
            expected: ">= 1".to_string(),
            actual: "0".to_string(),
        });
    }
    if order == 0 || order >= m {
        return Err(crate::FdarError::InvalidParameter {
            parameter: "order",
            message: format!("must be in [1, {m}), got {order}"),
        });
    }
    let t_min = argvals[0];
    let t_max = argvals[m - 1];
    for &q in query_points {
        if q < t_min || q > t_max {
            return Err(crate::FdarError::InvalidParameter {
                parameter: "query_points",
                message: format!(
                    "all query points must lie in [{t_min}, {t_max}]; found {q} which is outside the interpolation domain"
                ),
            });
        }
    }

    // --- Build knot vector and basis matrix on argvals ---
    // nknots chosen so nbasis = nknots + order ≈ m (interpolating system)
    let nknots = m.saturating_sub(order).max(2);
    let knots = crate::basis::bspline::construct_bspline_knots(t_min, t_max, nknots, order);

    // basis_vals: column-major, length m * nbasis; layout: basis[ti + k*m] = B_k(argvals[ti])
    let basis_vals = crate::basis::bspline::bspline_basis(argvals, nknots, order);
    let nbasis = basis_vals.len() / m;

    // Form B (m × nbasis) — same layout as pspline.rs:86-87
    let b_mat = nalgebra::DMatrix::from_column_slice(m, nbasis, &basis_vals);

    // Compute pseudoinverse of B once via SVD and reuse across all n curves.
    // Mirrors the pattern in basis/helpers.rs:svd_pseudoinverse (which is pub(super)).
    let tol = NUMERICAL_EPS * b_mat.nrows().max(b_mat.ncols()) as f64;
    let svd = nalgebra::SVD::new(b_mat.clone(), true, true);
    let pinv = svd
        .pseudo_inverse(tol)
        .map_err(|e| crate::FdarError::ComputationFailed {
            operation: "spline_interpolate SVD pseudoinverse",
            detail: e.to_string(),
        })?;
    // pinv: nbasis × m

    // --- Build query basis on same knots ---
    // basis_query: column-major, length m_q * nbasis; layout: basis_query[j + k*m_q] = B_k(query[j])
    let m_q = query_points.len();
    let basis_query = crate::basis::bspline::bspline_basis_from_knots(query_points, &knots, order);

    // --- Evaluate per curve ---
    let mut out = crate::matrix::FdMatrix::zeros(n, m_q);
    for i in 0..n {
        // Gather curve i as a column vector (length m)
        let y_vec: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        let y_col = nalgebra::DVector::from_vec(y_vec);

        // Solve: coefs = pinv * y  (shape: nbasis × 1)
        let coefs = &pinv * y_col;

        // Evaluate: out[i, j] = sum_k coefs[k] * basis_query[j + k*m_q]
        for j in 0..m_q {
            let mut val = 0.0;
            for k in 0..nbasis {
                val += coefs[k] * basis_query[j + k * m_q];
            }
            out[(i, j)] = val;
        }
    }

    Ok(out)
}

/// Interpolate functional data to a new grid using B-splines with explicit extrapolation control.
///
/// Like [`spline_interpolate`] but applies `policy` for any query point that falls outside the
/// domain `[argvals[0], argvals[m-1]]` instead of always returning an error. In-range queries
/// produce identical values to the [`spline_interpolate`] path.
///
/// # Arguments
/// * `data`         — Functional data matrix (`n × m`)
/// * `argvals`      — Original evaluation points (length `m`, must be sorted)
/// * `query_points` — Points to evaluate at (may include out-of-range values)
/// * `order`        — B-spline order (1 = linear, 2 = quadratic, 4 = cubic, …); must be in `[1, m)`
/// * `policy`       — Extrapolation policy for out-of-range query points
///
/// # Returns
/// Interpolated `FdMatrix` of shape `(n, query_points.len())`
///
/// # Errors
/// * `FdarError::InvalidDimension` — `argvals.len() != data.ncols()` or `query_points` is empty
/// * `FdarError::InvalidParameter` — `order` is 0 or ≥ `m`; or `policy == Exception` and any
///   query point is outside `[argvals[0], argvals[m-1]]`; or `policy == Periodic` and the domain
///   length is zero
/// * `FdarError::ComputationFailed` — SVD pseudoinverse could not be computed
///
/// # Policy Semantics
/// * `Boundary` — clamp OOB query to the nearest boundary (`t_min` or `t_max`) before spline eval
/// * `Exception` — return `Err(FdarError::InvalidParameter { parameter: "query_points", .. })` on first OOB
/// * `Fill(v)` — set OOB output cells to constant `v`; in-range cells use spline interpolation
/// * `Periodic` — wrap OOB query modulo the domain length (same recipe as in
///   [`fdata_interpolate_with_policy`]); requires `argvals[0] < argvals[m-1]`
pub fn spline_interpolate_with_policy(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    query_points: &[f64],
    order: usize,
    policy: ExtrapolationPolicy,
) -> Result<crate::matrix::FdMatrix, crate::FdarError> {
    let (n, m) = data.shape();

    // --- Input validation (mirrors spline_interpolate) ---
    if argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if query_points.is_empty() {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "query_points",
            expected: ">= 1".to_string(),
            actual: "0".to_string(),
        });
    }
    if order == 0 || order >= m {
        return Err(crate::FdarError::InvalidParameter {
            parameter: "order",
            message: format!("must be in [1, {m}), got {order}"),
        });
    }

    let t_min = argvals[0];
    let t_max = argvals[m - 1];
    let domain_len = t_max - t_min;

    // Periodic requires a positive domain length.
    if domain_len <= 0.0 && matches!(policy, ExtrapolationPolicy::Periodic) {
        return Err(crate::FdarError::InvalidParameter {
            parameter: "argvals",
            message: "Periodic extrapolation requires a positive domain length \
                      (argvals[0] < argvals[m-1])"
                .to_string(),
        });
    }

    let m_q = query_points.len();

    // --- Map each query point to an effective in-range point (or mark as Fill) ---
    // `effective[j]` holds the remapped query to pass to the spline; `fill_mask[j]` is true
    // when the output should be the Fill constant instead.
    let mut effective = Vec::with_capacity(m_q);
    let mut fill_mask = vec![false; m_q];

    for (j, &q) in query_points.iter().enumerate() {
        let in_range = q >= t_min && q <= t_max;
        if in_range {
            effective.push(q);
        } else {
            match &policy {
                ExtrapolationPolicy::Boundary => effective.push(q.clamp(t_min, t_max)),
                ExtrapolationPolicy::Exception => {
                    return Err(crate::FdarError::InvalidParameter {
                        parameter: "query_points",
                        message: format!(
                            "query {q} is outside domain [{t_min}, {t_max}]"
                        ),
                    });
                }
                ExtrapolationPolicy::Fill(_) => {
                    // Placeholder; we will overwrite this column from the fill value.
                    fill_mask[j] = true;
                    effective.push(t_min); // dummy in-range value — will be discarded
                }
                ExtrapolationPolicy::Periodic => {
                    let wrapped =
                        t_min + ((q - t_min) % domain_len + domain_len) % domain_len;
                    effective.push(wrapped);
                }
            }
        }
    }

    // --- Run the core spline logic on the effective (all in-range) query points ---
    // Reuse the same SVD-based evaluation as spline_interpolate.
    let nknots = m.saturating_sub(order).max(2);
    let knots =
        crate::basis::bspline::construct_bspline_knots(t_min, t_max, nknots, order);
    let basis_vals = crate::basis::bspline::bspline_basis(argvals, nknots, order);
    let nbasis = basis_vals.len() / m;
    let b_mat = nalgebra::DMatrix::from_column_slice(m, nbasis, &basis_vals);
    let tol = NUMERICAL_EPS * b_mat.nrows().max(b_mat.ncols()) as f64;
    let svd = nalgebra::SVD::new(b_mat.clone(), true, true);
    let pinv = svd
        .pseudo_inverse(tol)
        .map_err(|e| crate::FdarError::ComputationFailed {
            operation: "spline_interpolate_with_policy SVD pseudoinverse",
            detail: e.to_string(),
        })?;
    let basis_query =
        crate::basis::bspline::bspline_basis_from_knots(&effective, &knots, order);

    let mut out = crate::matrix::FdMatrix::zeros(n, m_q);
    for i in 0..n {
        let y_vec: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        let y_col = nalgebra::DVector::from_vec(y_vec);
        let coefs = &pinv * y_col;

        for j in 0..m_q {
            if fill_mask[j] {
                // Fill policy: overwrite with constant.
                if let ExtrapolationPolicy::Fill(v) = policy {
                    out[(i, j)] = v;
                }
            } else {
                let mut val = 0.0_f64;
                for k in 0..nbasis {
                    val += coefs[k] * basis_query[j + k * m_q];
                }
                out[(i, j)] = val;
            }
        }
    }

    Ok(out)
}

/// Cubic Hermite interpolation at a single point.
///
/// Uses Fritsch-Carlson monotone slopes for C1 interpolation.
fn cubic_hermite_interp(x: &[f64], y: &[f64], t: f64) -> f64 {
    let n = x.len();
    if n < 2 {
        return if n == 1 { y[0] } else { 0.0 };
    }

    // Clamp to domain
    if t <= x[0] {
        return y[0];
    }
    if t >= x[n - 1] {
        return y[n - 1];
    }

    // Find interval via binary search
    let k = match x.binary_search_by(|v| v.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => return y[i],
        Err(i) => {
            if i == 0 {
                0
            } else {
                i - 1
            }
        }
    };

    // Compute slopes (Fritsch-Carlson)
    let slopes: Vec<f64> = x
        .windows(2)
        .zip(y.windows(2))
        .map(|(xw, yw)| (yw[1] - yw[0]) / (xw[1] - xw[0]))
        .collect();

    // Tangents at each point
    let mut tangents = vec![0.0; n];
    tangents[0] = slopes[0];
    tangents[n - 1] = slopes[n - 2];
    for i in 1..n - 1 {
        if slopes[i - 1].signum() != slopes[i].signum() {
            tangents[i] = 0.0;
        } else {
            tangents[i] = (slopes[i - 1] + slopes[i]) / 2.0;
        }
    }

    // Hermite basis
    let h = x[k + 1] - x[k];
    let s = (t - x[k]) / h;
    let s2 = s * s;
    let s3 = s2 * s;

    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;

    h00 * y[k] + h10 * h * tangents[k] + h01 * y[k + 1] + h11 * h * tangents[k + 1]
}

/// Numerical gradient with uniform spacing using 5-point stencil (O(h⁴)).
///
/// Interior points use the 5-point central difference:
///   `g[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12*h)`
///
/// Near-boundary points use appropriate forward/backward formulas.
pub fn gradient_uniform(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0; n];
    if n < 2 {
        return g;
    }
    if n == 2 {
        g[0] = (y[1] - y[0]) / h;
        g[1] = (y[1] - y[0]) / h;
        return g;
    }
    if n == 3 {
        g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h);
        g[1] = (y[2] - y[0]) / (2.0 * h);
        g[2] = (y[0] - 4.0 * y[1] + 3.0 * y[2]) / (2.0 * h);
        return g;
    }
    if n == 4 {
        g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h);
        g[1] = (y[2] - y[0]) / (2.0 * h);
        g[2] = (y[3] - y[1]) / (2.0 * h);
        g[3] = (y[1] - 4.0 * y[2] + 3.0 * y[3]) / (2.0 * h);
        return g;
    }

    // n >= 5: use 5-point stencil for interior, 4-point formulas at boundaries
    // Left boundary: O(h³) forward formula
    g[0] = (-25.0 * y[0] + 48.0 * y[1] - 36.0 * y[2] + 16.0 * y[3] - 3.0 * y[4]) / (12.0 * h);
    g[1] = (-3.0 * y[0] - 10.0 * y[1] + 18.0 * y[2] - 6.0 * y[3] + y[4]) / (12.0 * h);

    // Interior: 5-point central difference O(h⁴)
    for i in 2..n - 2 {
        g[i] = (-y[i + 2] + 8.0 * y[i + 1] - 8.0 * y[i - 1] + y[i - 2]) / (12.0 * h);
    }

    // Right boundary: O(h³) backward formula
    g[n - 2] = (-y[n - 5] + 6.0 * y[n - 4] - 18.0 * y[n - 3] + 10.0 * y[n - 2] + 3.0 * y[n - 1])
        / (12.0 * h);
    g[n - 1] = (3.0 * y[n - 5] - 16.0 * y[n - 4] + 36.0 * y[n - 3] - 48.0 * y[n - 2]
        + 25.0 * y[n - 1])
        / (12.0 * h);
    g
}

/// Numerical gradient for non-uniform grids using 3-point Lagrange derivative.
///
/// At interior points uses the three-point formula:
///   `g[i] = y[i-1]*h_r/(-h_l*(h_l+h_r)) + y[i]*(h_r-h_l)/(h_l*h_r) + y[i+1]*h_l/(h_r*(h_l+h_r))`
/// where `h_l = t[i]-t[i-1]` and `h_r = t[i+1]-t[i]`.
///
/// Boundary points use forward/backward 3-point formulas.
pub fn gradient_nonuniform(y: &[f64], t: &[f64]) -> Vec<f64> {
    let n = y.len();
    assert_eq!(n, t.len(), "y and t must have the same length");
    let mut g = vec![0.0; n];
    if n < 2 {
        return g;
    }
    if n == 2 {
        let h = t[1] - t[0];
        if h.abs() < 1e-15 {
            return g;
        }
        g[0] = (y[1] - y[0]) / h;
        g[1] = g[0];
        return g;
    }

    // Left boundary: 3-point forward Lagrange derivative
    let h0 = t[1] - t[0];
    let h1 = t[2] - t[0];
    if h0.abs() > 1e-15 && h1.abs() > 1e-15 && (h1 - h0).abs() > 1e-15 {
        g[0] = y[0] * (-h1 - h0) / (h0 * h1) + y[1] * h1 / (h0 * (h1 - h0))
            - y[2] * h0 / (h1 * (h1 - h0));
    } else {
        g[0] = (y[1] - y[0]) / h0.max(1e-15);
    }

    // Interior: 3-point Lagrange central formula
    for i in 1..n - 1 {
        let h_l = t[i] - t[i - 1];
        let h_r = t[i + 1] - t[i];
        let h_sum = h_l + h_r;
        if h_l.abs() < 1e-15 || h_r.abs() < 1e-15 || h_sum.abs() < 1e-15 {
            g[i] = 0.0;
            continue;
        }
        g[i] = -y[i - 1] * h_r / (h_l * h_sum)
            + y[i] * (h_r - h_l) / (h_l * h_r)
            + y[i + 1] * h_l / (h_r * h_sum);
    }

    // Right boundary: 3-point backward Lagrange derivative
    let h_last = t[n - 1] - t[n - 2];
    let h_prev = t[n - 1] - t[n - 3];
    let h_mid = t[n - 2] - t[n - 3];
    if h_last.abs() > 1e-15 && h_prev.abs() > 1e-15 && h_mid.abs() > 1e-15 {
        g[n - 1] = y[n - 3] * h_last / (h_mid * h_prev) - y[n - 2] * h_prev / (h_mid * h_last)
            + y[n - 1] * (h_prev + h_last) / (h_prev * h_last);
    } else {
        g[n - 1] = (y[n - 1] - y[n - 2]) / h_last.max(1e-15);
    }

    g
}

/// Numerical gradient that auto-detects uniform vs non-uniform grids.
///
/// If the grid `t` is uniformly spaced (max|Δt_i − Δt_0| < ε), dispatches to
/// [`gradient_uniform`] for optimal accuracy. Otherwise falls back to
/// [`gradient_nonuniform`].
pub fn gradient(y: &[f64], t: &[f64]) -> Vec<f64> {
    let n = t.len();
    if n < 2 {
        return vec![0.0; y.len()];
    }

    let h0 = t[1] - t[0];
    let is_uniform = t
        .windows(2)
        .all(|w| ((w[1] - w[0]) - h0).abs() < 1e-12 * h0.abs().max(1.0));

    if is_uniform {
        gradient_uniform(y, h0)
    } else {
        gradient_nonuniform(y, t)
    }
}

/// Extrapolation policy controlling behavior when a query point falls
/// outside the domain of `argvals`.
///
/// Used with [`fdata_interpolate_with_policy`] to give callers explicit control
/// over out-of-range query handling instead of the silent boundary clamp that
/// [`fdata_interpolate`] applies.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExtrapolationPolicy {
    /// Clamp the query point to the nearest boundary value.
    ///
    /// A query at `t < t_min` returns the interpolated value at `t_min`;
    /// a query at `t > t_max` returns the interpolated value at `t_max`.
    Boundary,
    /// Return an error for any out-of-range query point.
    ///
    /// Returns `Err(FdarError::InvalidParameter { parameter: "new_argvals", .. })`
    /// when a query point lies outside `[argvals[0], argvals[m-1]]`.
    Exception,
    /// Fill out-of-range cells with a constant value.
    Fill(f64),
    /// Wrap query points modulo the domain length (periodic extension).
    ///
    /// A query at `t_min - delta` returns the same value as `t_max - delta`.
    /// Uses the guarded-modulo recipe `((t - t_min) % L + L) % L` to handle
    /// negative remainders for `t < t_min`.
    Periodic,
}

/// Interpolate functional data to a new grid with explicit extrapolation control.
///
/// Like [`fdata_interpolate`] but applies `policy` for any query point that falls
/// outside the domain `[argvals[0], argvals[m-1]]` instead of silently clamping.
/// In-range queries produce identical values to the [`fdata_interpolate`] path.
///
/// # Arguments
/// * `data`       — Functional data matrix (`n × m`)
/// * `argvals`    — Original evaluation points (length `m`, must be sorted)
/// * `new_argvals`— New evaluation points (length `m_new`)
/// * `method`     — Interpolation method for in-range (and `Boundary`/`Periodic`) points
/// * `policy`     — Extrapolation policy for out-of-range query points
///
/// # Returns
/// Interpolated matrix `(n × m_new)`
///
/// # Errors
/// * `FdarError::InvalidDimension` — `argvals.len() != data.ncols()`
/// * `FdarError::InvalidParameter` — a query point is out of range and `policy ==
///   ExtrapolationPolicy::Exception`
pub fn fdata_interpolate_with_policy(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    new_argvals: &[f64],
    method: InterpolationMethod,
    policy: ExtrapolationPolicy,
) -> Result<crate::matrix::FdMatrix, crate::FdarError> {
    let (n, m) = data.shape();
    if argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    let m_new = new_argvals.len();
    if n == 0 || m < 2 || m_new == 0 {
        return Ok(crate::matrix::FdMatrix::zeros(n.max(1), m_new.max(1)));
    }
    let t_min = argvals[0];
    let t_max = argvals[m - 1];
    let domain_len = t_max - t_min;

    // CR-01: Guard degenerate domain before the loop — Periodic wraps via modulo domain_len
    // and produces NaN when domain_len == 0 (IEEE 754: x % 0.0 = NaN).  Other policies do
    // not divide by domain_len so only Periodic needs this guard.
    if domain_len <= 0.0 && matches!(policy, ExtrapolationPolicy::Periodic) {
        return Err(crate::FdarError::InvalidParameter {
            parameter: "argvals",
            message: "Periodic extrapolation requires a positive domain length \
                      (argvals[0] < argvals[m-1])"
                .to_string(),
        });
    }

    let mut result = crate::matrix::FdMatrix::zeros(n, m_new);
    for i in 0..n {
        let y: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        for (j, &t) in new_argvals.iter().enumerate() {
            let in_range = t >= t_min && t <= t_max;
            result[(i, j)] = if in_range {
                match method {
                    InterpolationMethod::Linear => linear_interp(argvals, &y, t),
                    InterpolationMethod::CubicHermite => cubic_hermite_interp(argvals, &y, t),
                }
            } else {
                match &policy {
                    ExtrapolationPolicy::Boundary => {
                        let t_clamped = t.clamp(t_min, t_max);
                        match method {
                            InterpolationMethod::Linear => linear_interp(argvals, &y, t_clamped),
                            InterpolationMethod::CubicHermite => {
                                cubic_hermite_interp(argvals, &y, t_clamped)
                            }
                        }
                    }
                    ExtrapolationPolicy::Exception => {
                        return Err(crate::FdarError::InvalidParameter {
                            parameter: "new_argvals",
                            message: format!("query {t} is outside domain [{t_min}, {t_max}]"),
                        });
                    }
                    ExtrapolationPolicy::Fill(v) => *v,
                    ExtrapolationPolicy::Periodic => {
                        let wrapped = t_min + ((t - t_min) % domain_len + domain_len) % domain_len;
                        match method {
                            InterpolationMethod::Linear => linear_interp(argvals, &y, wrapped),
                            InterpolationMethod::CubicHermite => {
                                cubic_hermite_interp(argvals, &y, wrapped)
                            }
                        }
                    }
                }
            };
        }
    }
    Ok(result)
}

// ── FEAT-03: NaN imputation ────────────────────────────────────────────────

/// Strategy for in-grid NaN imputation in a regular `FdMatrix`.
///
/// Used with [`impute_missing_values`] to specify how NaN entries are replaced
/// in each curve.
///
/// Leading or trailing NaN values (no neighbor on one side) are filled with
/// the nearest valid value (boundary extension) for the `Linear` strategy.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImputationMethod {
    /// Linear interpolation between the nearest non-NaN neighbors.
    ///
    /// For gaps at the boundary (leading or trailing NaN), the nearest valid
    /// value is used as boundary extension.
    Linear,
    /// Replace each NaN with the curve's mean of its non-NaN values.
    Mean,
    /// Replace each NaN with a user-supplied constant value.
    Constant(f64),
}

/// Impute NaN values in a regular functional data matrix.
///
/// Returns a new `FdMatrix` with NaN entries replaced according to `method`.
/// Non-NaN entries are copied through unchanged.
///
/// For `Linear`, gaps at the curve boundary (no neighbor on one side) are
/// filled with the nearest valid value (boundary extension).
///
/// # Arguments
/// * `data`    — Functional data matrix (`n × m`) with possible NaN entries
/// * `argvals` — Evaluation points (length `m`, must be sorted)
/// * `method`  — Imputation strategy
///
/// # Errors
/// * `FdarError::InvalidDimension` if `argvals.len() != data.ncols()`
/// * `FdarError::InvalidParameter` if any curve consists entirely of NaN values
pub fn impute_missing_values(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    method: ImputationMethod,
) -> Result<crate::matrix::FdMatrix, crate::FdarError> {
    let (n, m) = data.shape();
    if argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    // WR-01: A zero-column matrix has no evaluation points and is degenerate.
    // Without this guard the per-curve loop would report "curve 0 contains only NaN values",
    // which is factually incorrect (the curve has no values at all).
    if m == 0 {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data",
            expected: "m >= 1".to_string(),
            actual: "m=0".to_string(),
        });
    }
    let mut out_data = vec![0.0_f64; n * m]; // column-major output buffer
    for i in 0..n {
        let row: Vec<f64> = data.row(i);
        let valid_count = row.iter().filter(|v| !v.is_nan()).count();
        if valid_count == 0 {
            return Err(crate::FdarError::InvalidParameter {
                parameter: "data",
                message: format!("curve {i} contains only NaN values"),
            });
        }
        let imputed = impute_row(&row, argvals, &method);
        for j in 0..m {
            out_data[i + j * n] = imputed[j]; // column-major write
        }
    }
    crate::matrix::FdMatrix::from_column_major(out_data, n, m)
}

/// Impute a single row (curve) of NaN values using the given strategy.
fn impute_row(row: &[f64], argvals: &[f64], method: &ImputationMethod) -> Vec<f64> {
    let mut result = row.to_vec();
    match method {
        ImputationMethod::Mean => {
            let sum: f64 = row.iter().filter(|v| !v.is_nan()).sum();
            let count = row.iter().filter(|v| !v.is_nan()).count();
            let mean = sum / count as f64;
            for v in &mut result {
                if v.is_nan() {
                    *v = mean;
                }
            }
        }
        ImputationMethod::Constant(c) => {
            for v in &mut result {
                if v.is_nan() {
                    *v = *c;
                }
            }
        }
        ImputationMethod::Linear => {
            let valid_idxs: Vec<usize> = (0..row.len()).filter(|&j| !row[j].is_nan()).collect();
            for j in 0..row.len() {
                if result[j].is_nan() {
                    let left = valid_idxs.iter().rev().find(|&&k| k < j).copied();
                    let right = valid_idxs.iter().find(|&&k| k > j).copied();
                    result[j] = match (left, right) {
                        (Some(l), Some(r)) => {
                            linear_interp(&[argvals[l], argvals[r]], &[row[l], row[r]], argvals[j])
                        }
                        (Some(l), None) => row[l], // boundary fill (trailing NaN)
                        (None, Some(r)) => row[r], // boundary fill (leading NaN)
                        (None, None) => unreachable!(), // all-NaN already rejected
                    };
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpsons_weights_uniform() {
        let argvals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let weights = simpsons_weights(&argvals);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_simpsons_weights_2d() {
        let argvals_s = vec![0.0, 0.5, 1.0];
        let argvals_t = vec![0.0, 0.5, 1.0];
        let weights = simpsons_weights_2d(&argvals_s, &argvals_t);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_extract_curves() {
        // Column-major data: 2 observations, 3 points
        // obs 0: [1, 2, 3], obs 1: [4, 5, 6]
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mat = crate::matrix::FdMatrix::from_column_major(data, 2, 3).unwrap();
        let curves = extract_curves(&mat);
        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(curves[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_l2_distance_identical() {
        let curve = vec![1.0, 2.0, 3.0];
        let weights = vec![0.25, 0.5, 0.25];
        let dist = l2_distance(&curve, &curve, &weights);
        assert!(dist.abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_l2_distance_different() {
        let curve1 = vec![0.0, 0.0, 0.0];
        let curve2 = vec![1.0, 1.0, 1.0];
        let weights = vec![0.25, 0.5, 0.25]; // sum = 1
        let dist = l2_distance(&curve1, &curve2, &weights);
        // dist^2 = 0.25*1 + 0.5*1 + 0.25*1 = 1.0, so dist = 1.0
        assert!((dist - 1.0).abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_n1_weights() {
        // Single point: fallback weight is 1.0 (degenerate case)
        let w = simpsons_weights(&[0.5]);
        assert_eq!(w.len(), 1);
        assert!((w[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_n2_weights() {
        let w = simpsons_weights(&[0.0, 1.0]);
        assert_eq!(w.len(), 2);
        // Trapezoidal: each weight should be 0.5
        assert!((w[0] - 0.5).abs() < 1e-12);
        assert!((w[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_mismatched_l2_distance() {
        // Mismatched lengths should not panic but may give garbage
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let w = vec![0.5, 0.5, 0.5];
        let d = l2_distance(&a, &b, &w);
        assert!(d.abs() < 1e-12, "Same vectors should have zero distance");
    }

    // ── trapz ──

    #[test]
    fn test_trapz_sine() {
        // ∫₀^π sin(x) dx = 2
        let m = 1000;
        let x: Vec<f64> = (0..m)
            .map(|i| std::f64::consts::PI * i as f64 / (m - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let result = trapz(&y, &x);
        assert!(
            (result - 2.0).abs() < 1e-4,
            "∫ sin(x) dx over [0,π] should be ~2, got {result}"
        );
    }

    // ── cumulative_trapz ──

    #[test]
    fn test_cumulative_trapz_matches_final() {
        let m = 100;
        let x: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
        let cum = cumulative_trapz(&y, &x);
        let total = trapz(&y, &x);
        assert!(
            (cum[m - 1] - total).abs() < 1e-12,
            "Final cumulative value should match trapz"
        );
    }

    // ── linear_interp ──

    #[test]
    fn test_linear_interp_boundary_clamp() {
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![10.0, 20.0, 30.0];
        assert!((linear_interp(&x, &y, -1.0) - 10.0).abs() < 1e-12);
        assert!((linear_interp(&x, &y, 2.0) - 30.0).abs() < 1e-12);
        assert!((linear_interp(&x, &y, 0.25) - 15.0).abs() < 1e-12);
    }

    // ── gradient_uniform ──

    #[test]
    fn test_gradient_uniform_linear() {
        // f(x) = 3x → f'(x) = 3 everywhere
        let m = 50;
        let h = 1.0 / (m - 1) as f64;
        let y: Vec<f64> = (0..m).map(|i| 3.0 * i as f64 * h).collect();
        let g = gradient_uniform(&y, h);
        for i in 0..m {
            assert!(
                (g[i] - 3.0).abs() < 1e-10,
                "gradient of 3x should be 3 at i={i}, got {}",
                g[i]
            );
        }
    }

    // ── fdata_interpolate ──

    #[test]
    fn test_gaussian_kernel() {
        assert!((gaussian_kernel(0.0, 1.0) - 1.0).abs() < 1e-12);
        assert!(gaussian_kernel(3.0, 1.0) < 0.02); // far from center
        assert!((gaussian_kernel(1.0, 0.0)).abs() < 1e-12); // zero bandwidth
    }

    #[test]
    fn test_bandwidth_candidates() {
        let n = 5;
        let mut dists = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                dists[i * n + j] = (i as f64 - j as f64).abs();
            }
        }
        let cands = bandwidth_candidates_from_dists(&dists, n, 10);
        assert!(!cands.is_empty());
        assert!(cands.iter().all(|&h| h > 0.0));
        // Should be sorted
        for w in cands.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn test_quantile_sorted() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile_sorted(&data, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile_sorted(&data, 1.0) - 5.0).abs() < 1e-12);
        assert!((quantile_sorted(&data, 0.5) - 3.0).abs() < 1e-12);
        assert!((quantile_sorted(&data, 0.25) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_r_squared_perfect() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let resid = vec![0.0, 0.0, 0.0, 0.0];
        assert!((r_squared(&y, &resid) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_r_squared_mean_model() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let mean = 2.5;
        let resid: Vec<f64> = y.iter().map(|&yi| yi - mean).collect();
        assert!(r_squared(&y, &resid).abs() < 1e-12); // R²=0 for mean model
    }

    #[test]
    fn test_aic_bic() {
        let a = aic(100, 50.0, 5);
        let b = bic(100, 50.0, 5);
        assert!(a.is_finite());
        assert!(b.is_finite());
        assert!(b > a); // BIC penalizes more for n > ~8
    }

    #[test]
    fn fdata_interpolate_linear_identity() {
        let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        let vals: Vec<f64> = t.iter().map(|&x| x.sin()).collect();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        let result = fdata_interpolate(&data, &t, &t, InterpolationMethod::Linear);
        for j in 0..20 {
            assert!((result[(0, j)] - data[(0, j)]).abs() < 1e-12);
        }
    }

    #[test]
    fn fdata_interpolate_cubic_hermite_smooth() {
        let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        let vals: Vec<f64> = t.iter().map(|&x| x.sin()).collect();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();

        let t_fine: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let result = fdata_interpolate(&data, &t, &t_fine, InterpolationMethod::CubicHermite);

        // Values should approximate sin(t) well
        for (j, &tj) in t_fine.iter().enumerate() {
            assert!(
                (result[(0, j)] - tj.sin()).abs() < 0.02,
                "at t={tj:.2}: got {:.4}, expected {:.4}",
                result[(0, j)],
                tj.sin()
            );
        }
    }

    #[test]
    fn fdata_interpolate_multiple_curves() {
        let t: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
        let n = 5;
        let m = 30;
        // Build column-major data: n curves, each sin((i+1)*x)
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                col_major[i + j * n] = ((i + 1) as f64 * t[j]).sin();
            }
        }
        let data = crate::matrix::FdMatrix::from_column_major(col_major, n, m).unwrap();

        let t_new: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let result = fdata_interpolate(&data, &t, &t_new, InterpolationMethod::Linear);
        assert_eq!(result.shape(), (n, 50));
        // All values should be finite
        for i in 0..n {
            for j in 0..50 {
                assert!(result[(i, j)].is_finite());
            }
        }
    }

    // ── spline_interpolate ──

    #[test]
    fn spline_interpolate_reproduces_argvals() {
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20);
        let vals: Vec<f64> = t.iter().map(|&x| x.powi(3)).collect();
        // column-major: 1 row, 20 columns
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        let result = spline_interpolate(&data, &t, &t, 4).unwrap();
        for j in 0..20 {
            assert!(
                (result[(0, j)] - data[(0, j)]).abs() < 1e-10,
                "at j={j}: got {}, expected {}",
                result[(0, j)],
                data[(0, j)]
            );
        }
    }

    #[test]
    fn spline_interpolate_cubic_offgrid() {
        // A cubic polynomial y = 2t^3 - t^2 + 0.5t - 0.1 lies exactly in the
        // order-4 B-spline space; an order-4 interpolant should reproduce it
        // within 1e-10 at off-grid midpoints.
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20); // 20 evaluation points in [0, 1]
        let poly = |x: f64| 2.0 * x.powi(3) - x.powi(2) + 0.5 * x - 0.1;
        let vals: Vec<f64> = t.iter().map(|&x| poly(x)).collect();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();

        // Query at off-grid midpoints between consecutive t values
        let q: Vec<f64> = t.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        let result = spline_interpolate(&data, &t, &q, 4).unwrap();

        for (j, &qj) in q.iter().enumerate() {
            let expected = poly(qj);
            let got = result[(0, j)];
            assert!(
                (got - expected).abs() < 1e-10,
                "off-grid at q={qj:.4}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn spline_interpolate_rejects_out_of_range() {
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20);
        let vals: Vec<f64> = t.to_vec();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();

        // Query point below argvals[0]
        let q_below = vec![-0.1_f64];
        let err = spline_interpolate(&data, &t, &q_below, 4).unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidParameter {
                    parameter: "query_points",
                    ..
                }
            ),
            "expected InvalidParameter for query below domain, got {err:?}"
        );

        // Query point above argvals[m-1]
        let q_above = vec![1.1_f64];
        let err2 = spline_interpolate(&data, &t, &q_above, 4).unwrap_err();
        assert!(
            matches!(
                err2,
                crate::FdarError::InvalidParameter {
                    parameter: "query_points",
                    ..
                }
            ),
            "expected InvalidParameter for query above domain, got {err2:?}"
        );
    }

    #[test]
    fn spline_interpolate_rejects_bad_order() {
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20);
        let vals: Vec<f64> = t.to_vec();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        let q = vec![0.5_f64];

        // order == 0
        let err = spline_interpolate(&data, &t, &q, 0).unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidParameter {
                    parameter: "order",
                    ..
                }
            ),
            "expected InvalidParameter for order=0, got {err:?}"
        );

        // order >= m (m=20)
        let err2 = spline_interpolate(&data, &t, &q, 20).unwrap_err();
        assert!(
            matches!(
                err2,
                crate::FdarError::InvalidParameter {
                    parameter: "order",
                    ..
                }
            ),
            "expected InvalidParameter for order=20 (>=m=20), got {err2:?}"
        );
    }

    #[test]
    fn spline_interpolate_rejects_dim_mismatch() {
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20);
        let vals: Vec<f64> = t.to_vec();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();

        // argvals.len() != data.ncols()
        let bad_argvals: Vec<f64> = (0..15).map(|i| i as f64 / 14.0).collect();
        let q = vec![0.5_f64];
        let err = spline_interpolate(&data, &bad_argvals, &q, 4).unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidDimension {
                    parameter: "argvals",
                    ..
                }
            ),
            "expected InvalidDimension for argvals mismatch, got {err:?}"
        );

        // empty query_points
        let err2 = spline_interpolate(&data, &t, &[], 4).unwrap_err();
        assert!(
            matches!(
                err2,
                crate::FdarError::InvalidDimension {
                    parameter: "query_points",
                    ..
                }
            ),
            "expected InvalidDimension for empty query_points, got {err2:?}"
        );
    }

    // ── spline_interpolate_with_policy tests ──────────────────────────────

    #[test]
    fn test_spline_with_policy_in_range_matches_spline() {
        // In-range queries must match spline_interpolate exactly regardless of policy.
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20);
        let poly = |x: f64| 2.0 * x.powi(3) - x.powi(2) + 0.5 * x - 0.1;
        let vals: Vec<f64> = t.iter().map(|&x| poly(x)).collect();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        // Off-grid midpoints (all in-range)
        let q: Vec<f64> = t.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        let expected = spline_interpolate(&data, &t, &q, 4).unwrap();
        let actual = spline_interpolate_with_policy(
            &data,
            &t,
            &q,
            4,
            ExtrapolationPolicy::Boundary,
        )
        .unwrap();
        for j in 0..q.len() {
            assert!(
                (actual[(0, j)] - expected[(0, j)]).abs() < 1e-10,
                "in-range mismatch at j={j}: policy={} vs plain={}",
                actual[(0, j)],
                expected[(0, j)]
            );
        }
    }

    #[test]
    fn test_spline_with_policy_boundary() {
        // OOB queries clamped to nearest boundary value.
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20); // [0, 1]
        let vals: Vec<f64> = t.iter().map(|&x| x.powi(2)).collect(); // y = x^2
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        // Query below (should clamp to t_min=0 → y=0) and above (clamp to t_max=1 → y≈1).
        let q = vec![-0.5_f64, 0.5, 1.5];
        let result = spline_interpolate_with_policy(
            &data,
            &t,
            &q,
            4,
            ExtrapolationPolicy::Boundary,
        )
        .unwrap();
        // Clamped to 0 → y = 0^2 = 0 (within spline tolerance)
        assert!(result[(0, 0)].abs() < 1e-9, "below boundary should clamp, got {}", result[(0, 0)]);
        // Clamped to 1 → y = 1^2 = 1 (within spline tolerance)
        assert!((result[(0, 2)] - 1.0).abs() < 1e-9, "above boundary should clamp, got {}", result[(0, 2)]);
        // In-range 0.5 → y ≈ 0.25
        assert!((result[(0, 1)] - 0.25).abs() < 1e-9, "in-range should be ~0.25, got {}", result[(0, 1)]);
    }

    #[test]
    fn test_spline_with_policy_exception() {
        // Exception policy errors on OOB, matches spline_interpolate behavior.
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20);
        let vals: Vec<f64> = t.to_vec();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        let q_oob = vec![1.5_f64];
        let err = spline_interpolate_with_policy(
            &data,
            &t,
            &q_oob,
            4,
            ExtrapolationPolicy::Exception,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidParameter {
                    parameter: "query_points",
                    ..
                }
            ),
            "Exception policy should error on OOB, got {err:?}"
        );
        // In-range queries with Exception policy should succeed.
        let q_ok = vec![0.0_f64, 0.5, 1.0];
        let ok = spline_interpolate_with_policy(
            &data,
            &t,
            &q_ok,
            4,
            ExtrapolationPolicy::Exception,
        );
        assert!(ok.is_ok(), "Exception policy should succeed for in-range queries");
    }

    #[test]
    fn test_spline_with_policy_fill() {
        // Fill policy: OOB cells get constant fill value; in-range cells use spline.
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20); // [0, 1]
        let vals: Vec<f64> = t.iter().map(|&x| x.powi(2)).collect();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        let fill_val = 42.0_f64;
        let q = vec![-0.5_f64, 0.5, 2.0];
        let result = spline_interpolate_with_policy(
            &data,
            &t,
            &q,
            4,
            ExtrapolationPolicy::Fill(fill_val),
        )
        .unwrap();
        assert!(
            (result[(0, 0)] - fill_val).abs() < 1e-10,
            "OOB below should be fill value, got {}",
            result[(0, 0)]
        );
        assert!(
            (result[(0, 2)] - fill_val).abs() < 1e-10,
            "OOB above should be fill value, got {}",
            result[(0, 2)]
        );
        // In-range: y ≈ 0.25
        assert!(
            (result[(0, 1)] - 0.25).abs() < 1e-9,
            "in-range should be ~0.25, got {}",
            result[(0, 1)]
        );
    }

    #[test]
    fn test_spline_with_policy_periodic() {
        // Periodic policy: OOB queries wrap modulo domain length.
        // Use y = x (linear) on [0, 1]; a query at 1.3 should wrap to 0.3.
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(20); // [0, 1]
        let vals: Vec<f64> = t.to_vec(); // y = x
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        let q = vec![1.3_f64];
        let result = spline_interpolate_with_policy(
            &data,
            &t,
            &q,
            4,
            ExtrapolationPolicy::Periodic,
        )
        .unwrap();
        // Expected: wrap 1.3 → 0.3, spline of y=x at 0.3 ≈ 0.3
        assert!(
            (result[(0, 0)] - 0.3).abs() < 1e-9,
            "Periodic wrap of 1.3 should give ~0.3, got {}",
            result[(0, 0)]
        );
        // t = -0.2 → wrap to 0.8
        let q2 = vec![-0.2_f64];
        let result2 = spline_interpolate_with_policy(
            &data,
            &t,
            &q2,
            4,
            ExtrapolationPolicy::Periodic,
        )
        .unwrap();
        assert!(
            (result2[(0, 0)] - 0.8).abs() < 1e-9,
            "Periodic wrap of -0.2 should give ~0.8, got {}",
            result2[(0, 0)]
        );
    }

    #[test]
    fn test_spline_with_policy_periodic_zero_length_domain_errors() {
        // Periodic + zero-length domain must error (same guard as fdata_interpolate_with_policy).
        let argvals = vec![3.0_f64, 3.0, 3.0];
        let vals = vec![1.0_f64, 1.0, 1.0];
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 3).unwrap();
        let q = vec![4.0_f64]; // OOB
        let err = spline_interpolate_with_policy(
            &data,
            &argvals,
            &q,
            1,
            ExtrapolationPolicy::Periodic,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidParameter {
                    parameter: "argvals",
                    ..
                }
            ),
            "Periodic + zero-length domain should error, got {err:?}"
        );
    }

    // ── ExtrapolationPolicy tests ──────────────────────────────────────────

    /// Build a 1-curve FdMatrix: y = x on [0, 1] with `n_pts` points.
    fn make_linear_curve(n_pts: usize) -> (crate::matrix::FdMatrix, Vec<f64>) {
        use crate::test_helpers::uniform_grid;
        let t = uniform_grid(n_pts);
        let vals: Vec<f64> = t.to_vec();
        let mat = crate::matrix::FdMatrix::from_column_major(vals, 1, n_pts).unwrap();
        (mat, t)
    }

    #[test]
    fn test_extrapolation_boundary() {
        let (data, t) = make_linear_curve(11); // y=x on [0,1]
                                               // Query at t=-0.2 (below) and t=1.3 (above)
        let q = vec![-0.2_f64, 0.5, 1.3];
        let result = fdata_interpolate_with_policy(
            &data,
            &t,
            &q,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Boundary,
        )
        .unwrap();
        // Clamped to t_min=0.0 → y=0.0
        assert!(
            (result[(0, 0)] - 0.0).abs() < 1e-10,
            "below boundary should clamp to 0"
        );
        // In-range: y=0.5
        assert!(
            (result[(0, 1)] - 0.5).abs() < 1e-10,
            "in-range should interpolate correctly"
        );
        // Clamped to t_max=1.0 → y=1.0
        assert!(
            (result[(0, 2)] - 1.0).abs() < 1e-10,
            "above boundary should clamp to 1"
        );
    }

    #[test]
    fn test_extrapolation_exception() {
        let (data, t) = make_linear_curve(11);
        let q_bad = vec![1.5_f64]; // out of range
        let err = fdata_interpolate_with_policy(
            &data,
            &t,
            &q_bad,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Exception,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidParameter {
                    parameter: "new_argvals",
                    ..
                }
            ),
            "expected InvalidParameter for OOB query, got {err:?}"
        );

        // In-range should still work with Exception policy
        let q_ok = vec![0.0_f64, 0.5, 1.0];
        let result = fdata_interpolate_with_policy(
            &data,
            &t,
            &q_ok,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Exception,
        )
        .unwrap();
        assert!((result[(0, 1)] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolation_fill() {
        let (data, t) = make_linear_curve(11);
        let fill_val = 99.0_f64;
        let q = vec![-0.5_f64, 0.5, 2.0];
        let result = fdata_interpolate_with_policy(
            &data,
            &t,
            &q,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Fill(fill_val),
        )
        .unwrap();
        assert!(
            (result[(0, 0)] - fill_val).abs() < 1e-10,
            "below range should be fill value"
        );
        assert!(
            (result[(0, 1)] - 0.5).abs() < 1e-10,
            "in-range should interpolate"
        );
        assert!(
            (result[(0, 2)] - fill_val).abs() < 1e-10,
            "above range should be fill value"
        );
    }

    #[test]
    fn test_extrapolation_periodic() {
        let (data, t) = make_linear_curve(11); // y=x on [0,1], domain_len=1
                                               // t = -0.1 should wrap to 0.9 (y=0.9)
        let q = vec![-0.1_f64, 0.5, 1.1];
        let result = fdata_interpolate_with_policy(
            &data,
            &t,
            &q,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Periodic,
        )
        .unwrap();
        // wrapped: ((−0.1 − 0) % 1 + 1) % 1 = (−0.1 + 1) % 1 = 0.9 % 1 = 0.9
        assert!(
            (result[(0, 0)] - 0.9).abs() < 1e-9,
            "t=-0.1 should wrap to 0.9, got {}",
            result[(0, 0)]
        );
        assert!(
            (result[(0, 1)] - 0.5).abs() < 1e-10,
            "in-range point unchanged"
        );
        // t=1.1 → ((1.1 − 0) % 1 + 1) % 1 = (0.1 + 1) % 1 = 0.1
        assert!(
            (result[(0, 2)] - 0.1).abs() < 1e-9,
            "t=1.1 should wrap to 0.1, got {}",
            result[(0, 2)]
        );
    }

    #[test]
    fn test_extrapolation_in_range_equivalence() {
        // In-range queries must match fdata_interpolate exactly
        let (data, t) = make_linear_curve(21);
        let q: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let expected = fdata_interpolate(&data, &t, &q, InterpolationMethod::Linear);
        let actual = fdata_interpolate_with_policy(
            &data,
            &t,
            &q,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Boundary,
        )
        .unwrap();
        let (_, m_new) = actual.shape();
        for j in 0..m_new {
            assert!(
                (actual[(0, j)] - expected[(0, j)]).abs() < 1e-12,
                "in-range mismatch at j={j}: policy={} vs plain={}",
                actual[(0, j)],
                expected[(0, j)]
            );
        }
    }

    #[test]
    fn test_extrapolation_policy_dim_guard() {
        let (data, _t) = make_linear_curve(11);
        let bad_argvals: Vec<f64> = (0..5).map(|i| i as f64 / 4.0).collect(); // len=5, ncols=11
        let q = vec![0.5_f64];
        let err = fdata_interpolate_with_policy(
            &data,
            &bad_argvals,
            &q,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Boundary,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidDimension {
                    parameter: "argvals",
                    ..
                }
            ),
            "expected InvalidDimension for argvals mismatch, got {err:?}"
        );
    }

    // ── ImputationMethod / impute_missing_values tests ────────────────────

    /// Build a 1-curve FdMatrix from given values and a uniform grid.
    fn make_curve_with_vals(vals: Vec<f64>) -> (crate::matrix::FdMatrix, Vec<f64>) {
        let m = vals.len();
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let mat = crate::matrix::FdMatrix::from_column_major(vals, 1, m).unwrap();
        (mat, argvals)
    }

    #[test]
    fn test_impute_linear() {
        // Curve: [0.0, NaN, 1.0] on argvals [0.0, 0.5, 1.0]
        // Linear between (0,0.0) and (1.0,1.0): at t=0.5 → 0.5
        let (data, argvals) = make_curve_with_vals(vec![0.0_f64, f64::NAN, 1.0]);
        let result = impute_missing_values(&data, &argvals, ImputationMethod::Linear).unwrap();
        // Hand-computed: linear_interp([0.0,1.0],[0.0,1.0],0.5) = 0.5
        assert!(
            (result[(0, 1)] - 0.5).abs() < 1e-10,
            "linear imputation should give 0.5, got {}",
            result[(0, 1)]
        );
        // Non-NaN entries unchanged
        assert!((result[(0, 0)] - 0.0).abs() < 1e-10);
        assert!((result[(0, 2)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_mean() {
        // Curve: [1.0, NaN, 3.0] → mean of non-NaN = 2.0
        let (data, argvals) = make_curve_with_vals(vec![1.0_f64, f64::NAN, 3.0]);
        let result = impute_missing_values(&data, &argvals, ImputationMethod::Mean).unwrap();
        assert!(
            (result[(0, 1)] - 2.0).abs() < 1e-10,
            "mean imputation should give 2.0, got {}",
            result[(0, 1)]
        );
    }

    #[test]
    fn test_impute_constant() {
        // Curve: [1.0, NaN, 3.0] → constant 99.0
        let (data, argvals) = make_curve_with_vals(vec![1.0_f64, f64::NAN, 3.0]);
        let result =
            impute_missing_values(&data, &argvals, ImputationMethod::Constant(99.0)).unwrap();
        assert!(
            (result[(0, 1)] - 99.0).abs() < 1e-10,
            "constant imputation should give 99.0, got {}",
            result[(0, 1)]
        );
        // Non-NaN entries unchanged
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(0, 2)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_impute_all_nan() {
        // An all-NaN curve should return Err(InvalidParameter)
        let (data, argvals) = make_curve_with_vals(vec![f64::NAN, f64::NAN, f64::NAN]);
        let err = impute_missing_values(&data, &argvals, ImputationMethod::Linear).unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidParameter {
                    parameter: "data",
                    ..
                }
            ),
            "expected InvalidParameter for all-NaN curve, got {err:?}"
        );
    }

    #[test]
    fn test_impute_boundary_nan() {
        // Curve: [NaN, 0.5, 1.0] → leading NaN → boundary fill with 0.5
        let (data, argvals) = make_curve_with_vals(vec![f64::NAN, 0.5_f64, 1.0]);
        let result = impute_missing_values(&data, &argvals, ImputationMethod::Linear).unwrap();
        assert!(
            (result[(0, 0)] - 0.5).abs() < 1e-10,
            "leading NaN should be filled with nearest valid (0.5), got {}",
            result[(0, 0)]
        );

        // Curve: [0.0, 0.5, NaN] → trailing NaN → boundary fill with 0.5
        let (data2, argvals2) = make_curve_with_vals(vec![0.0_f64, 0.5, f64::NAN]);
        let result2 = impute_missing_values(&data2, &argvals2, ImputationMethod::Linear).unwrap();
        assert!(
            (result2[(0, 2)] - 0.5).abs() < 1e-10,
            "trailing NaN should be filled with nearest valid (0.5), got {}",
            result2[(0, 2)]
        );
    }

    // ── CR-01: Periodic + zero-length domain must error (not produce NaN) ────

    #[test]
    fn test_extrapolation_periodic_zero_length_domain_errors() {
        // Domain [5.0, 5.0] has length 0 — Periodic would compute x % 0.0 = NaN without the guard.
        let degenerate_argvals = vec![5.0_f64, 5.0, 5.0];
        let vals = vec![1.0_f64, 1.0, 1.0];
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 3).unwrap();
        // Any OOB query with Periodic on a zero-length domain must return Err.
        let q = vec![6.0_f64]; // outside [5.0, 5.0]
        let err = fdata_interpolate_with_policy(
            &data,
            &degenerate_argvals,
            &q,
            InterpolationMethod::Linear,
            ExtrapolationPolicy::Periodic,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidParameter {
                    parameter: "argvals",
                    ..
                }
            ),
            "expected InvalidParameter for zero-length domain + Periodic, got {err:?}"
        );
    }

    // ── WR-01: m=0 guard in impute_missing_values ─────────────────────────

    #[test]
    fn test_impute_zero_columns_errors() {
        // A matrix with m=0 columns is degenerate; should return InvalidDimension, not "all-NaN".
        let data = crate::matrix::FdMatrix::zeros(2, 0);
        let argvals: Vec<f64> = vec![];
        let err = impute_missing_values(&data, &argvals, ImputationMethod::Linear).unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidDimension {
                    parameter: "data",
                    ..
                }
            ),
            "expected InvalidDimension for m=0 matrix, got {err:?}"
        );
    }

    #[test]
    fn test_impute_dim_mismatch() {
        // argvals length != ncols
        let (data, _argvals) = make_curve_with_vals(vec![1.0, 2.0, 3.0]);
        let bad_argvals = vec![0.0_f64, 1.0]; // len=2, ncols=3
        let err = impute_missing_values(&data, &bad_argvals, ImputationMethod::Linear).unwrap_err();
        assert!(
            matches!(
                err,
                crate::FdarError::InvalidDimension {
                    parameter: "argvals",
                    ..
                }
            ),
            "expected InvalidDimension for argvals mismatch, got {err:?}"
        );
    }
}
