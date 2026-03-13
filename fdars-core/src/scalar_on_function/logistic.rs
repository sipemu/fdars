use super::*;

// ---------------------------------------------------------------------------
// Functional logistic regression
// ---------------------------------------------------------------------------

/// One IRLS step: compute working response and solve weighted least squares.
/// Returns updated beta or None if system is singular.
fn irls_step(design: &FdMatrix, y: &[f64], beta: &[f64]) -> Option<Vec<f64>> {
    let (n, p) = design.shape();

    // Linear predictor η = Xβ, probabilities μ = sigmoid(η)
    let eta: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| design[(i, j)] * beta[j]).sum())
        .collect();
    let mu: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();
    let w: Vec<f64> = mu.iter().map(|&p| (p * (1.0 - p)).max(1e-10)).collect();
    let z_work: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / w[i]).collect();

    // Weighted normal equations: (X'WX) β = X'Wz
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let mut s = 0.0;
            for i in 0..n {
                s += design[(i, k)] * w[i] * design[(i, j)];
            }
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }

    let mut xtwz = vec![0.0; p];
    for k in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += design[(i, k)] * w[i] * z_work[i];
        }
        xtwz[k] = s;
    }

    cholesky_solve(&xtwx, &xtwz, p).ok()
}

/// Compute log-likelihood of logistic model.
fn logistic_log_likelihood(probabilities: &[f64], y: &[f64]) -> f64 {
    probabilities
        .iter()
        .zip(y)
        .map(|(&p, &yi)| {
            let p = p.clamp(1e-15, 1.0 - 1e-15);
            yi * p.ln() + (1.0 - yi) * (1.0 - p).ln()
        })
        .sum()
}

/// Run IRLS iteration loop and return (beta, iterations).
fn irls_loop(design: &FdMatrix, y: &[f64], max_iter: usize, tol: f64) -> (Vec<f64>, usize) {
    let p_total = design.ncols();
    let mut beta = vec![0.0; p_total];
    let mut iterations = 0;
    for iter in 0..max_iter {
        iterations = iter + 1;
        let beta_new = match irls_step(design, y, &beta) {
            Some(b) => b,
            None => break,
        };
        let change: f64 = beta_new
            .iter()
            .zip(&beta)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        beta = beta_new;
        if change < tol {
            break;
        }
    }
    (beta, iterations)
}

/// Build logistic result from converged beta.
fn build_logistic_result(
    design: &FdMatrix,
    beta: Vec<f64>,
    y: &[f64],
    fpca: FpcaResult,
    ncomp: usize,
    m: usize,
    iterations: usize,
) -> FunctionalLogisticResult {
    let (n, p) = design.shape();
    let eta = compute_fitted(design, &beta);
    let probabilities: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();
    let predicted_classes: Vec<usize> = probabilities
        .iter()
        .map(|&p| if p >= 0.5 { 1 } else { 0 })
        .collect();
    let correct: usize = predicted_classes
        .iter()
        .zip(y)
        .filter(|(&pred, &actual)| pred as f64 == actual)
        .count();
    let beta_t = recover_beta_t(&beta[1..1 + ncomp], &fpca.rotation, m);
    let gamma: Vec<f64> = beta[1 + ncomp..].to_vec();

    // Compute coefficient SEs from Fisher information matrix (X'WX)^{-1}
    let w: Vec<f64> = probabilities
        .iter()
        .map(|&mu| (mu * (1.0 - mu)).max(1e-10))
        .collect();
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let mut s = 0.0;
            for i in 0..n {
                s += design[(i, k)] * w[i] * design[(i, j)];
            }
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }
    let std_errors = cholesky_factor(&xtwx, p).map_or_else(
        |_| vec![f64::NAN; p],
        |l| compute_ols_std_errors(&l, p, 1.0),
    );
    let beta_se = compute_beta_se(&std_errors[1..1 + ncomp], &fpca.rotation, m);

    let ll = logistic_log_likelihood(&probabilities, y);
    let deviance = -2.0 * ll;
    let nf = n as f64;
    let pf = p as f64;
    let aic = deviance + 2.0 * pf;
    let bic = deviance + nf.ln() * pf;

    FunctionalLogisticResult {
        intercept: beta[0],
        beta_t,
        beta_se,
        gamma,
        accuracy: correct as f64 / nf,
        log_likelihood: ll,
        probabilities,
        predicted_classes,
        ncomp,
        std_errors,
        coefficients: beta,
        iterations,
        fpca,
        aic,
        bic,
    }
}

/// Functional logistic regression for binary outcomes.
///
/// Fits: `log(P(Y=1)/P(Y=0)) = α + ∫β(t)X(t)dt + γᵀz`
/// using IRLS (iteratively reweighted least squares) on FPC scores.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Binary response vector (0.0 or 1.0, length n)
/// * `scalar_covariates` - Optional scalar covariates (n × p)
/// * `ncomp` - Number of FPC components
/// * `max_iter` - Maximum IRLS iterations (default: 25)
/// * `tol` - Convergence tolerance (default: 1e-6)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if any element of `y` is not `0.0`
/// or `1.0`.
/// Returns [`FdarError::ComputationFailed`] if the SVD inside FPCA fails to
/// extract components.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::functional_logistic;
///
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(),
///     10, 20,
/// ).unwrap();
/// let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
/// let fit = functional_logistic(&data, &y, None, 3, 25, 1e-6).unwrap();
/// assert_eq!(fit.probabilities.len(), 10);
/// assert!(fit.probabilities.iter().all(|&p| p >= 0.0 && p <= 1.0));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_logistic(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    max_iter: usize,
    tol: f64,
) -> Result<FunctionalLogisticResult, FdarError> {
    let (n, m) = data.shape();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n}"),
        });
    }
    if m == 0 {
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
    if y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: "all values must be 0.0 or 1.0 for binary classification".to_string(),
        });
    }

    let ncomp = ncomp.min(n - 1).min(m);
    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let design = build_design_matrix(&fpca.scores, ncomp, scalar_covariates, n);

    let max_iter = if max_iter == 0 { 25 } else { max_iter };
    let tol = if tol <= 0.0 { 1e-6 } else { tol };

    let (beta, iterations) = irls_loop(&design, y, max_iter, tol);
    Ok(build_logistic_result(
        &design, beta, y, fpca, ncomp, m, iterations,
    ))
}

/// Predict probabilities P(Y=1) for new data using a fitted functional logistic model.
///
/// Projects new curves through the stored FPCA, computes linear predictor,
/// and applies sigmoid.
///
/// # Arguments
/// * `fit` — A fitted [`FunctionalLogisticResult`]
/// * `new_data` — New functional predictor matrix (n_new × m)
/// * `new_scalar` — Optional new scalar covariates (n_new × p)
pub fn predict_functional_logistic(
    fit: &FunctionalLogisticResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Vec<f64> {
    let (n_new, m) = new_data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();

    (0..n_new)
        .map(|i| {
            let mut eta = fit.coefficients[0]; // intercept
            for k in 0..ncomp {
                let mut s = 0.0;
                for j in 0..m {
                    s += (new_data[(i, j)] - fit.fpca.mean[j]) * fit.fpca.rotation[(j, k)];
                }
                eta += fit.coefficients[1 + k] * s;
            }
            if let Some(sc) = new_scalar {
                for j in 0..p_scalar {
                    eta += fit.gamma[j] * sc[(i, j)];
                }
            }
            sigmoid(eta)
        })
        .collect()
}
