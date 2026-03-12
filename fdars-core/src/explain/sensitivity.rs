//! Sobol sensitivity indices, functional saliency, and domain selection.

use super::helpers::*;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};
use rand::prelude::*;

// ===========================================================================
// Sobol Sensitivity Indices
// ===========================================================================

/// Sobol first-order and total-order sensitivity indices.
pub struct SobolIndicesResult {
    /// First-order indices S_k, length ncomp.
    pub first_order: Vec<f64>,
    /// Total-order indices ST_k, length ncomp.
    pub total_order: Vec<f64>,
    /// Total variance of Y.
    pub var_y: f64,
    /// Per-component variance contribution, length ncomp.
    pub component_variance: Vec<f64>,
}

/// Exact Sobol sensitivity indices for a linear functional regression model.
///
/// For an additive model with orthogonal FPC predictors, first-order = total-order.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 2 rows, its
/// column count does not match `fit.fpca.mean`, or `y.len()` does not match the
/// row count.
/// Returns [`FdarError::InvalidParameter`] if `fit.ncomp` is zero.
/// Returns [`FdarError::ComputationFailed`] if the variance of `y` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn sobol_indices(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
) -> Result<SobolIndicesResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">=2 rows".into(),
            actual: format!("{n}"),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} (matching data rows)"),
            actual: format!("{}", y.len()),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let _ = scalar_covariates; // not needed for variance decomposition
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }

    let score_var = compute_score_variance(&fit.fpca.scores, n, ncomp);

    let component_variance: Vec<f64> = (0..ncomp)
        .map(|k| fit.coefficients[1 + k].powi(2) * score_var[k])
        .collect();

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let var_y: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    if var_y == 0.0 {
        return Err(FdarError::ComputationFailed {
            operation: "sobol_indices",
            detail: "variance of y is zero".into(),
        });
    }

    let first_order: Vec<f64> = component_variance.iter().map(|&cv| cv / var_y).collect();
    let total_order = first_order.clone(); // additive + orthogonal -> S_k = ST_k

    Ok(SobolIndicesResult {
        first_order,
        total_order,
        var_y,
        component_variance,
    })
}

/// Sobol sensitivity indices for a functional logistic regression model (Saltelli MC).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 2 rows or its
/// column count does not match `fit.fpca.mean`.
/// Returns [`FdarError::InvalidParameter`] if `n_samples` is zero or `fit.ncomp`
/// is zero.
/// Returns [`FdarError::ComputationFailed`] if the variance of predictions is
/// near zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn sobol_indices_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<SobolIndicesResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">=2 rows".into(),
            actual: format!("{n}"),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    if n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "must be > 0".into(),
        });
    }
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let eval_model = |s: &[f64]| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * s[k];
        }
        for j in 0..p_scalar {
            eta += fit.gamma[j] * mean_z[j];
        }
        sigmoid(eta)
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let (mat_a, mat_b) = generate_sobol_matrices(&scores, n, ncomp, n_samples, &mut rng);

    let f_a: Vec<f64> = mat_a.iter().map(|s| eval_model(s)).collect();
    let f_b: Vec<f64> = mat_b.iter().map(|s| eval_model(s)).collect();

    let mean_fa = f_a.iter().sum::<f64>() / n_samples as f64;
    let var_fa = f_a.iter().map(|&v| (v - mean_fa).powi(2)).sum::<f64>() / n_samples as f64;

    if var_fa < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "sobol_indices_logistic",
            detail: "variance of predictions is near zero".into(),
        });
    }

    let mut first_order = vec![0.0; ncomp];
    let mut total_order = vec![0.0; ncomp];
    let mut component_variance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let (s_k, st_k) = compute_sobol_component(
            &mat_a,
            &mat_b,
            &f_a,
            &f_b,
            var_fa,
            k,
            n_samples,
            &eval_model,
        );
        first_order[k] = s_k;
        total_order[k] = st_k;
        component_variance[k] = s_k * var_fa;
    }

    Ok(SobolIndicesResult {
        first_order,
        total_order,
        var_y: var_fa,
        component_variance,
    })
}

// ===========================================================================
// Functional Saliency Maps
// ===========================================================================

/// Functional saliency map result.
pub struct FunctionalSaliencyResult {
    /// Saliency map (n x m).
    pub saliency_map: FdMatrix,
    /// Mean absolute saliency at each grid point (length m).
    pub mean_absolute_saliency: Vec<f64>,
}

/// Functional saliency maps for a linear functional regression model.
///
/// Lifts FPC-level SHAP attributions to the function domain via the rotation matrix.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its column
/// count does not match `fit.fpca.mean`.
/// Returns [`FdarError::InvalidParameter`] if `fit.ncomp` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_saliency(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<FunctionalSaliencyResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_scores = compute_column_means(&scores, ncomp);

    let weights: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let saliency_map = compute_saliency_map(
        &scores,
        &mean_scores,
        &weights,
        &fit.fpca.rotation,
        n,
        m,
        ncomp,
    );
    let mean_absolute_saliency = mean_absolute_column(&saliency_map, n, m);

    Ok(FunctionalSaliencyResult {
        saliency_map,
        mean_absolute_saliency,
    })
}

/// Functional saliency maps for a functional logistic regression model (gradient-based).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `fit.probabilities` is empty or
/// `fit.beta_t` has zero length.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_saliency_logistic(
    fit: &FunctionalLogisticResult,
) -> Result<FunctionalSaliencyResult, FdarError> {
    let m = fit.beta_t.len();
    let n = fit.probabilities.len();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "probabilities",
            expected: ">0 length".into(),
            actual: "0".into(),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "beta_t",
            expected: ">0 length".into(),
            actual: "0".into(),
        });
    }

    // saliency[(i,j)] = p_i * (1 - p_i) * beta_t[j]
    let mut saliency_map = FdMatrix::zeros(n, m);
    for i in 0..n {
        let pi = fit.probabilities[i];
        let w = pi * (1.0 - pi);
        for j in 0..m {
            saliency_map[(i, j)] = w * fit.beta_t[j];
        }
    }

    let mut mean_absolute_saliency = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            mean_absolute_saliency[j] += saliency_map[(i, j)].abs();
        }
        mean_absolute_saliency[j] /= n as f64;
    }

    Ok(FunctionalSaliencyResult {
        saliency_map,
        mean_absolute_saliency,
    })
}

// ===========================================================================
// Domain Selection / Interval Importance
// ===========================================================================

/// An important interval in the function domain.
pub struct ImportantInterval {
    /// Start index (inclusive).
    pub start_idx: usize,
    /// End index (inclusive).
    pub end_idx: usize,
    /// Summed importance of the interval.
    pub importance: f64,
}

/// Result of domain selection analysis.
pub struct DomainSelectionResult {
    /// Pointwise importance: |beta(t)|^2, length m.
    pub pointwise_importance: Vec<f64>,
    /// Important intervals sorted by importance descending.
    pub intervals: Vec<ImportantInterval>,
    /// Sliding window width used.
    pub window_width: usize,
    /// Threshold used.
    pub threshold: f64,
}

/// Domain selection for a linear functional regression model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `beta_t`, `window_width`, or
/// `threshold` are invalid (e.g., empty `beta_t`, zero `window_width`, or
/// `window_width` exceeding `beta_t` length).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn domain_selection(
    fit: &FregreLmResult,
    window_width: usize,
    threshold: f64,
) -> Result<DomainSelectionResult, FdarError> {
    compute_domain_selection(&fit.beta_t, window_width, threshold).ok_or_else(|| {
        FdarError::InvalidParameter {
            parameter: "domain_selection",
            message: "invalid beta_t, window_width, or threshold".into(),
        }
    })
}

/// Domain selection for a functional logistic regression model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `beta_t`, `window_width`, or
/// `threshold` are invalid (e.g., empty `beta_t`, zero `window_width`, or
/// `window_width` exceeding `beta_t` length).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn domain_selection_logistic(
    fit: &FunctionalLogisticResult,
    window_width: usize,
    threshold: f64,
) -> Result<DomainSelectionResult, FdarError> {
    compute_domain_selection(&fit.beta_t, window_width, threshold).ok_or_else(|| {
        FdarError::InvalidParameter {
            parameter: "domain_selection_logistic",
            message: "invalid beta_t, window_width, or threshold".into(),
        }
    })
}
