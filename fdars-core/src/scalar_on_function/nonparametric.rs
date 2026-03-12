use super::*;

// ---------------------------------------------------------------------------
// Nonparametric kernel regression
// ---------------------------------------------------------------------------

/// Gaussian kernel: K(d, h) = exp(-d² / (2h²))
pub(super) fn gaussian_kernel(d: f64, h: f64) -> f64 {
    (-d * d / (2.0 * h * h)).exp()
}

/// Compute symmetric pairwise distance matrix (flat n×n).
pub(super) fn compute_pairwise_distances(data: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
    let n = data.nrows();
    let weights = crate::helpers::simpsons_weights(argvals);
    let mut dists = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = crate::helpers::l2_distance(&data.row(i), &data.row(j), &weights);
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Compute pairwise Euclidean distance matrix for scalar covariates.
pub(super) fn compute_scalar_distances(sc: &FdMatrix) -> Vec<f64> {
    let n = sc.nrows();
    let p = sc.ncols();
    let mut dists = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for k in 0..p {
                let diff = sc[(i, k)] - sc[(j, k)];
                d2 += diff * diff;
            }
            let d = d2.sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Nadaraya-Watson LOO prediction for one observation.
fn nw_loo_predict(
    i: usize,
    n: usize,
    y: &[f64],
    func_dists: &[f64],
    scalar_dists: &[f64],
    h_func: f64,
    h_scalar: f64,
    has_scalar: bool,
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for j in 0..n {
        if i == j {
            continue;
        }
        let kf = gaussian_kernel(func_dists[i * n + j], h_func);
        let ks = if has_scalar {
            gaussian_kernel(scalar_dists[i * n + j], h_scalar)
        } else {
            1.0
        };
        let w = kf * ks;
        num += w * y[j];
        den += w;
    }
    if den > 1e-15 {
        num / den
    } else {
        y[i]
    }
}

/// LOO-CV error for Nadaraya-Watson with a single bandwidth.
fn loo_cv_error(dists: &[f64], y: &[f64], n: usize, h: f64) -> f64 {
    (0..n)
        .map(|i| {
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let w = gaussian_kernel(dists[i * n + j], h);
                num += w * y[j];
                den += w;
            }
            let yhat = if den > 1e-15 { num / den } else { y[i] };
            (y[i] - yhat).powi(2)
        })
        .sum::<f64>()
        / n as f64
}

/// Select bandwidth by minimizing LOO-CV error on a grid of distance quantiles.
pub(super) fn select_bandwidth_loo(
    dists: &[f64],
    y: &[f64],
    n: usize,
    _other_dists: Option<&[f64]>,
) -> f64 {
    let mut nonzero_dists: Vec<f64> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| dists[i * n + j]))
        .filter(|&d| d > 0.0)
        .collect();
    crate::helpers::sort_nan_safe(&mut nonzero_dists);

    if nonzero_dists.is_empty() {
        return 1.0;
    }

    let n_cand = 20;
    let mut best_h = nonzero_dists[nonzero_dists.len() / 2];
    let mut best_cv = f64::INFINITY;

    for qi in 1..=n_cand {
        let q = f64::from(qi) / f64::from(n_cand + 1);
        let idx = ((nonzero_dists.len() as f64 * q) as usize).min(nonzero_dists.len() - 1);
        let h = nonzero_dists[idx].max(1e-10);
        let cv = loo_cv_error(dists, y, n, h);
        if cv < best_cv {
            best_cv = cv;
            best_h = h;
        }
    }
    best_h
}

/// Nonparametric kernel regression with mixed functional and scalar predictors.
///
/// Uses product kernels:
/// ```text
/// ŷ(x, z) = Σᵢ K_func(Xᵢ, x) · K_scalar(zᵢ, z) · yᵢ / Σᵢ K_func(Xᵢ, x) · K_scalar(zᵢ, z)
/// ```
///
/// Bandwidths are selected via leave-one-out CV if set to 0.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector
/// * `argvals` - Grid points for integration (length m)
/// * `scalar_covariates` - Optional scalar covariates (n × p)
/// * `h_func` - Bandwidth for functional kernel (0 for automatic)
/// * `h_scalar` - Bandwidth for scalar kernel (0 for automatic)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, `y.len() != n`, or `argvals.len() != m`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_np_mixed(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    h_func: f64,
    h_scalar: f64,
) -> Result<FregreNpResult, FdarError> {
    let n = data.nrows();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n}"),
        });
    }
    if data.ncols() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column".to_string(),
            actual: "0".to_string(),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if argvals.len() != data.ncols() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{}", data.ncols()),
            actual: format!("{}", argvals.len()),
        });
    }

    let func_dists = compute_pairwise_distances(data, argvals);
    let has_scalar = scalar_covariates.is_some();
    let scalar_dists = scalar_covariates
        .map(compute_scalar_distances)
        .unwrap_or_default();

    let h_func = if h_func <= 0.0 {
        select_bandwidth_loo(&func_dists, y, n, None)
    } else {
        h_func
    };

    let h_scalar = if has_scalar && h_scalar <= 0.0 {
        select_bandwidth_loo(&scalar_dists, y, n, Some(&func_dists))
    } else {
        h_scalar
    };

    let mut fitted_values = vec![0.0; n];
    let mut cv_error = 0.0;
    for i in 0..n {
        fitted_values[i] = nw_loo_predict(
            i,
            n,
            y,
            &func_dists,
            &scalar_dists,
            h_func,
            h_scalar,
            has_scalar,
        );
        cv_error += (y[i] - fitted_values[i]).powi(2);
    }
    cv_error /= n as f64;

    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let (r_squared, _) = compute_r_squared(y, &residuals, 1);

    Ok(FregreNpResult {
        fitted_values,
        residuals,
        r_squared,
        h_func,
        h_scalar,
        cv_error,
    })
}

/// Predict new responses using a fitted nonparametric model.
pub fn predict_fregre_np(
    train_data: &FdMatrix,
    y: &[f64],
    train_scalar: Option<&FdMatrix>,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
    argvals: &[f64],
    h_func: f64,
    h_scalar: f64,
) -> Vec<f64> {
    let n_train = train_data.nrows();
    let n_new = new_data.nrows();
    let weights = crate::helpers::simpsons_weights(argvals);

    (0..n_new)
        .map(|i| {
            let new_row = new_data.row(i);
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..n_train {
                let d_func = crate::helpers::l2_distance(&new_row, &train_data.row(j), &weights);
                let kf = gaussian_kernel(d_func, h_func);
                let ks = match (new_scalar, train_scalar) {
                    (Some(ns), Some(ts)) => {
                        let d2: f64 = (0..ns.ncols())
                            .map(|k| (ns[(i, k)] - ts[(j, k)]).powi(2))
                            .sum();
                        gaussian_kernel(d2.sqrt(), h_scalar)
                    }
                    _ => 1.0,
                };
                let w = kf * ks;
                num += w * y[j];
                den += w;
            }
            if den > 1e-15 {
                num / den
            } else {
                0.0
            }
        })
        .collect()
}
