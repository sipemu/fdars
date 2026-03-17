//! Cross-validation for the elastic alignment regularisation parameter lambda.

use super::karcher::karcher_mean;
use super::pairwise::elastic_distance;
use crate::cv::{create_folds, fold_indices, subset_rows};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

// ─── Config / Result ─────────────────────────────────────────────────────────

/// Configuration for lambda cross-validation.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaCvConfig {
    /// Candidate lambda values to evaluate.
    pub lambdas: Vec<f64>,
    /// Number of folds (0 = leave-one-out).
    pub n_folds: usize,
    /// Maximum Karcher iterations per fold.
    pub max_iter: usize,
    /// Karcher convergence tolerance.
    pub tol: f64,
    /// RNG seed for fold assignment.
    pub seed: u64,
}

impl Default for LambdaCvConfig {
    fn default() -> Self {
        Self {
            lambdas: vec![0.0, 0.01, 0.1, 1.0, 10.0],
            n_folds: 5,
            max_iter: 15,
            tol: 1e-3,
            seed: 42,
        }
    }
}

/// Result of lambda cross-validation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LambdaCvResult {
    /// Lambda with the lowest mean CV score.
    pub best_lambda: f64,
    /// Mean CV score for each candidate lambda (same order as `lambdas`).
    pub cv_scores: Vec<f64>,
    /// Candidate lambda values (copied from config).
    pub lambdas: Vec<f64>,
}

// ─── Cross-validation ────────────────────────────────────────────────────────

/// Select the best elastic-alignment regularisation parameter via K-fold
/// cross-validation.
///
/// For each candidate lambda the data are split into K folds. A Karcher mean
/// is computed on the training set and every held-out curve is scored by its
/// elastic distance to that mean. The lambda with the lowest average
/// held-out distance wins.
///
/// # Arguments
/// * `data`    — Functional data matrix (n x m).
/// * `argvals` — Evaluation grid (length m).
/// * `config`  — Cross-validation settings (lambdas, folds, iterations, …).
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if `data` has fewer than 4 rows
/// or `argvals` length does not match `data.ncols()`.
/// Returns `FdarError::InvalidParameter` if any lambda is negative or
/// `n_folds` is 1.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn lambda_cv(
    data: &FdMatrix,
    argvals: &[f64],
    config: &LambdaCvConfig,
) -> Result<LambdaCvResult, FdarError> {
    let n = data.nrows();
    let m = data.ncols();

    // ── Validation ──────────────────────────────────────────────────────
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 4 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if config.lambdas.iter().any(|&l| l < 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "lambdas",
            message: "all lambda values must be >= 0".to_string(),
        });
    }
    if config.n_folds == 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_folds",
            message: "n_folds must be > 1 or 0 (leave-one-out)".to_string(),
        });
    }

    let actual_folds = if config.n_folds == 0 {
        n
    } else {
        config.n_folds
    };
    let folds = create_folds(n, actual_folds, config.seed);

    // Number of distinct fold labels actually produced.
    let k_max = *folds.iter().max().unwrap_or(&0) + 1;

    // ── Evaluate each lambda ────────────────────────────────────────────
    let mut cv_scores = Vec::with_capacity(config.lambdas.len());

    for &lambda in &config.lambdas {
        let mut fold_scores = Vec::with_capacity(k_max);

        for k in 0..k_max {
            let (train_idx, test_idx) = fold_indices(&folds, k);
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }

            let train_data = subset_rows(data, &train_idx);
            let km = karcher_mean(&train_data, argvals, config.max_iter, config.tol, lambda);

            let fold_dist: f64 = test_idx
                .iter()
                .map(|&idx| {
                    let test_curve = data.row(idx);
                    elastic_distance(&test_curve, &km.mean, argvals, lambda)
                })
                .sum::<f64>()
                / test_idx.len() as f64;

            fold_scores.push(fold_dist);
        }

        let mean_score = if fold_scores.is_empty() {
            f64::INFINITY
        } else {
            fold_scores.iter().sum::<f64>() / fold_scores.len() as f64
        };
        cv_scores.push(mean_score);
    }

    // ── Pick best lambda ────────────────────────────────────────────────
    let best_idx = cv_scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(LambdaCvResult {
        best_lambda: config.lambdas[best_idx],
        cv_scores,
        lambdas: config.lambdas.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;

    fn make_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let data = sim_fundata(n, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
        (data, t)
    }

    #[test]
    fn lambda_cv_default_config() {
        let (data, t) = make_test_data(8, 30);
        let config = LambdaCvConfig {
            max_iter: 5,
            tol: 1e-2,
            ..LambdaCvConfig::default()
        };
        let result = lambda_cv(&data, &t, &config).unwrap();
        assert_eq!(result.cv_scores.len(), config.lambdas.len());
        assert!(result.best_lambda >= 0.0);
        assert!(result.cv_scores.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn lambda_cv_loo() {
        let (data, t) = make_test_data(6, 25);
        let config = LambdaCvConfig {
            lambdas: vec![0.0, 1.0],
            n_folds: 0,
            max_iter: 3,
            tol: 1e-2,
            seed: 7,
        };
        let result = lambda_cv(&data, &t, &config).unwrap();
        assert_eq!(result.cv_scores.len(), 2);
    }

    #[test]
    fn lambda_cv_rejects_too_few_rows() {
        let t = uniform_grid(10);
        let data = sim_fundata(3, &t, 2, EFunType::Fourier, EValType::Exponential, Some(0));
        let config = LambdaCvConfig::default();
        assert!(lambda_cv(&data, &t, &config).is_err());
    }

    #[test]
    fn lambda_cv_rejects_negative_lambda() {
        let (data, t) = make_test_data(8, 20);
        let config = LambdaCvConfig {
            lambdas: vec![-1.0, 0.0],
            ..LambdaCvConfig::default()
        };
        assert!(lambda_cv(&data, &t, &config).is_err());
    }

    #[test]
    fn lambda_cv_rejects_one_fold() {
        let (data, t) = make_test_data(8, 20);
        let config = LambdaCvConfig {
            n_folds: 1,
            ..LambdaCvConfig::default()
        };
        assert!(lambda_cv(&data, &t, &config).is_err());
    }

    #[test]
    fn lambda_cv_rejects_argval_mismatch() {
        let (data, _) = make_test_data(8, 20);
        let bad_t = uniform_grid(15);
        let config = LambdaCvConfig::default();
        assert!(lambda_cv(&data, &bad_t, &config).is_err());
    }
}
