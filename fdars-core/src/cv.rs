//! Cross-validation utilities and unified CV framework.
//!
//! This module provides:
//! - Shared fold assignment utilities used across all CV functions
//! - [`cv_fdata`]: Generic k-fold + repeated CV framework (R's `cv.fdata`)

use crate::matrix::FdMatrix;
use rand::prelude::*;
use std::any::Any;
use std::collections::HashMap;

// ─── Fold Utilities ─────────────────────────────────────────────────────────

/// Assign observations to folds (deterministic given seed).
///
/// Returns a vector of length `n` where element `i` is the fold index (0..n_folds)
/// that observation `i` belongs to.
pub fn create_folds(n: usize, n_folds: usize, seed: u64) -> Vec<usize> {
    let n_folds = n_folds.max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let mut folds = vec![0usize; n];
    for (rank, &idx) in indices.iter().enumerate() {
        folds[idx] = rank % n_folds;
    }
    folds
}

/// Assign observations to stratified folds (classification).
///
/// Ensures each fold has approximately the same class distribution.
pub fn create_stratified_folds(n: usize, y: &[usize], n_folds: usize, seed: u64) -> Vec<usize> {
    let n_folds = n_folds.max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let n_classes = y.iter().copied().max().unwrap_or(0) + 1;

    let mut folds = vec![0usize; n];

    // Group indices by class
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for i in 0..n {
        if y[i] < n_classes {
            class_indices[y[i]].push(i);
        }
    }

    // Shuffle within each class, then assign folds round-robin
    for indices in &mut class_indices {
        indices.shuffle(&mut rng);
        for (rank, &idx) in indices.iter().enumerate() {
            folds[idx] = rank % n_folds;
        }
    }

    folds
}

/// Split indices into train and test sets for a given fold.
///
/// Returns `(train_indices, test_indices)`.
pub fn fold_indices(folds: &[usize], fold: usize) -> (Vec<usize>, Vec<usize>) {
    let train: Vec<usize> = (0..folds.len()).filter(|&i| folds[i] != fold).collect();
    let test: Vec<usize> = (0..folds.len()).filter(|&i| folds[i] == fold).collect();
    (train, test)
}

/// Extract a sub-matrix from an FdMatrix by selecting specific row indices.
pub fn subset_rows(data: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let m = data.ncols();
    let n_sub = indices.len();
    let mut sub = FdMatrix::zeros(n_sub, m);
    for (new_i, &orig_i) in indices.iter().enumerate() {
        for j in 0..m {
            sub[(new_i, j)] = data[(orig_i, j)];
        }
    }
    sub
}

/// Extract elements from a slice by indices.
pub fn subset_vec(v: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| v[i]).collect()
}

// ─── CV Metrics ─────────────────────────────────────────────────────────────

/// Type of cross-validation task.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CvType {
    Regression,
    Classification,
}

/// Cross-validation metrics.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CvMetrics {
    /// Regression metrics.
    Regression { rmse: f64, mae: f64, r_squared: f64 },
    /// Classification metrics.
    Classification {
        accuracy: f64,
        confusion: Vec<Vec<usize>>,
    },
}

/// A named metric function: `(name, fn(y_true, y_pred) -> f64)`.
pub type MetricFn = (&'static str, fn(&[f64], &[f64]) -> f64);

// ─── Built-in Regression Metrics ────────────────────────────────────────────

/// Root Mean Squared Error.
pub fn metric_rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().min(y_pred.len());
    if n == 0 {
        return f64::NAN;
    }
    let mse: f64 = (0..n).map(|i| (y_true[i] - y_pred[i]).powi(2)).sum::<f64>() / n as f64;
    mse.sqrt()
}

/// Mean Absolute Error.
pub fn metric_mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().min(y_pred.len());
    if n == 0 {
        return f64::NAN;
    }
    (0..n).map(|i| (y_true[i] - y_pred[i]).abs()).sum::<f64>() / n as f64
}

/// Coefficient of determination (R-squared).
pub fn metric_r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().min(y_pred.len());
    if n == 0 {
        return f64::NAN;
    }
    let mean = y_true.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = (0..n).map(|i| (y_true[i] - y_pred[i]).powi(2)).sum();
    let ss_tot: f64 = (0..n).map(|i| (y_true[i] - mean).powi(2)).sum();
    if ss_tot > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Default regression metric set: RMSE, MAE, R-squared.
pub fn regression_metrics() -> Vec<MetricFn> {
    vec![
        ("rmse", metric_rmse as fn(&[f64], &[f64]) -> f64),
        ("mae", metric_mae),
        ("r_squared", metric_r_squared),
    ]
}

// ─── Built-in Classification Metrics ────────────────────────────────────────

/// Classification accuracy.
pub fn metric_accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().min(y_pred.len());
    if n == 0 {
        return f64::NAN;
    }
    let correct = (0..n)
        .filter(|&i| (y_true[i] as usize) == (y_pred[i].round() as usize))
        .count();
    correct as f64 / n as f64
}

/// Macro (binary) precision: TP / (TP + FP).
pub fn metric_precision(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().min(y_pred.len());
    let mut tp = 0usize;
    let mut fp = 0usize;
    for i in 0..n {
        let pred = y_pred[i].round() as usize;
        let true_c = y_true[i] as usize;
        if pred == 1 {
            if true_c == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    }
}

/// Macro (binary) recall: TP / (TP + FN).
pub fn metric_recall(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().min(y_pred.len());
    let mut tp = 0usize;
    let mut fn_ = 0usize;
    for i in 0..n {
        let pred = y_pred[i].round() as usize;
        let true_c = y_true[i] as usize;
        if true_c == 1 {
            if pred == 1 {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }
    if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    }
}

/// F1 score (harmonic mean of precision and recall).
pub fn metric_f1(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let p = metric_precision(y_true, y_pred);
    let r = metric_recall(y_true, y_pred);
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

/// Default classification metric set: accuracy, precision, recall, F1.
pub fn classification_metrics() -> Vec<MetricFn> {
    vec![
        ("accuracy", metric_accuracy as fn(&[f64], &[f64]) -> f64),
        ("precision", metric_precision),
        ("recall", metric_recall),
        ("f1", metric_f1),
    ]
}

/// Evaluate a set of metric functions on (y_true, y_pred).
fn evaluate_metrics(
    y_true: &[f64],
    y_pred: &[f64],
    metric_fns: &[MetricFn],
) -> HashMap<String, f64> {
    metric_fns
        .iter()
        .map(|(name, f)| ((*name).to_string(), f(y_true, y_pred)))
        .collect()
}

/// Result of unified cross-validation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CvFdataResult {
    /// Out-of-fold predictions (length n); for repeated CV, averaged across reps.
    pub oof_predictions: Vec<f64>,
    /// Overall metrics (built-in).
    pub metrics: CvMetrics,
    /// Per-fold metrics (built-in).
    pub fold_metrics: Vec<CvMetrics>,
    /// Fold assignments from the last (or only) repetition.
    pub folds: Vec<usize>,
    /// Type of CV task.
    pub cv_type: CvType,
    /// Number of repetitions.
    pub nrep: usize,
    /// Standard deviation of OOF predictions across repetitions (only when nrep > 1).
    pub oof_sd: Option<Vec<f64>>,
    /// Per-repetition overall metrics (only when nrep > 1).
    pub rep_metrics: Option<Vec<CvMetrics>>,
    /// Custom metrics evaluated on OOF predictions (name -> value).
    pub custom_metrics: HashMap<String, f64>,
    /// Per-fold custom metrics.
    pub fold_custom_metrics: Vec<HashMap<String, f64>>,
}

// ─── Unified CV Framework ────────────────────────────────────────────────────

/// Create CV folds based on strategy (stratified or random).
fn create_cv_folds(
    n: usize,
    y: &[f64],
    n_folds: usize,
    cv_type: CvType,
    stratified: bool,
    seed: u64,
) -> Vec<usize> {
    if stratified {
        match cv_type {
            CvType::Classification => {
                let y_class: Vec<usize> = y
                    .iter()
                    .map(|&v| crate::utility::f64_to_usize_clamped(v))
                    .collect();
                create_stratified_folds(n, &y_class, n_folds, seed)
            }
            CvType::Regression => {
                let mut sorted_y: Vec<(usize, f64)> = y.iter().copied().enumerate().collect();
                sorted_y.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let n_bins = n_folds.min(n);
                let bin_labels: Vec<usize> = {
                    let mut labels = vec![0usize; n];
                    for (rank, &(orig_i, _)) in sorted_y.iter().enumerate() {
                        labels[orig_i] = (rank * n_bins / n).min(n_bins - 1);
                    }
                    labels
                };
                create_stratified_folds(n, &bin_labels, n_folds, seed)
            }
        }
    } else {
        create_folds(n, n_folds, seed)
    }
}

/// Aggregate out-of-fold predictions across repetitions (mean and SD).
fn aggregate_oof_predictions(all_oof: Vec<Vec<f64>>, n: usize) -> (Vec<f64>, Option<Vec<f64>>) {
    let nrep = all_oof.len();
    if nrep == 1 {
        return (
            all_oof.into_iter().next().expect("non-empty iterator"),
            None,
        );
    }
    let mut mean_oof = vec![0.0; n];
    for oof in &all_oof {
        for i in 0..n {
            mean_oof[i] += oof[i];
        }
    }
    for v in &mut mean_oof {
        *v /= nrep as f64;
    }

    let mut sd_oof = vec![0.0; n];
    for oof in &all_oof {
        for i in 0..n {
            let diff = oof[i] - mean_oof[i];
            sd_oof[i] += diff * diff;
        }
    }
    for v in &mut sd_oof {
        *v = (*v / (nrep as f64 - 1.0).max(1.0)).sqrt();
    }

    (mean_oof, Some(sd_oof))
}

/// Generic k-fold + repeated cross-validation framework (R's `cv.fdata`).
///
/// Uses built-in metrics (RMSE/MAE/R² for regression, accuracy/confusion for
/// classification). For custom metrics, use [`cv_fdata_with_metrics`].
pub fn cv_fdata<F, P>(
    data: &FdMatrix,
    y: &[f64],
    fit_fn: F,
    predict_fn: P,
    n_folds: usize,
    nrep: usize,
    cv_type: CvType,
    stratified: bool,
    seed: u64,
) -> CvFdataResult
where
    F: Fn(&FdMatrix, &[f64]) -> Box<dyn Any>,
    P: Fn(&dyn Any, &FdMatrix) -> Vec<f64>,
{
    cv_fdata_with_metrics(
        data,
        y,
        fit_fn,
        predict_fn,
        n_folds,
        nrep,
        cv_type,
        stratified,
        seed,
        &[],
    )
}

/// Generic k-fold + repeated CV with user-defined metrics.
///
/// Same as [`cv_fdata`] but accepts a slice of [`MetricFn`] that are evaluated
/// on each fold's (y_true, y_pred) and on the overall OOF predictions.
///
/// # Examples
///
/// ```
/// use fdars_core::cv::*;
/// use fdars_core::matrix::FdMatrix;
/// use std::any::Any;
///
/// let data = FdMatrix::zeros(20, 5);
/// let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
///
/// // Custom metric: median absolute error
/// fn median_ae(y_true: &[f64], y_pred: &[f64]) -> f64 {
///     let mut errs: Vec<f64> = y_true.iter().zip(y_pred)
///         .map(|(&a, &b)| (a - b).abs()).collect();
///     errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
///     errs[errs.len() / 2]
/// }
///
/// let mut metrics = regression_metrics();
/// metrics.push(("median_ae", median_ae));
///
/// let result = cv_fdata_with_metrics(
///     &data, &y,
///     |_d, y| Box::new(y.iter().sum::<f64>() / y.len() as f64),
///     |m, td| { let v = *m.downcast_ref::<f64>().unwrap(); vec![v; td.nrows()] },
///     5, 1, CvType::Regression, false, 42,
///     &metrics,
/// );
/// assert!(result.custom_metrics.contains_key("rmse"));
/// assert!(result.custom_metrics.contains_key("median_ae"));
/// ```
pub fn cv_fdata_with_metrics<F, P>(
    data: &FdMatrix,
    y: &[f64],
    fit_fn: F,
    predict_fn: P,
    n_folds: usize,
    nrep: usize,
    cv_type: CvType,
    stratified: bool,
    seed: u64,
    metric_fns: &[MetricFn],
) -> CvFdataResult
where
    F: Fn(&FdMatrix, &[f64]) -> Box<dyn Any>,
    P: Fn(&dyn Any, &FdMatrix) -> Vec<f64>,
{
    let n = data.nrows();
    let nrep = nrep.max(1);
    let n_folds = n_folds.max(2).min(n);

    let mut all_oof: Vec<Vec<f64>> = Vec::with_capacity(nrep);
    let mut all_rep_metrics: Vec<CvMetrics> = Vec::with_capacity(nrep);
    let mut last_folds = vec![0usize; n];
    let mut last_fold_metrics = Vec::new();
    let mut last_fold_custom = Vec::new();

    for r in 0..nrep {
        let rep_seed = seed.wrapping_add(r as u64);
        let folds = create_cv_folds(n, y, n_folds, cv_type, stratified, rep_seed);

        let mut oof_preds = vec![0.0; n];
        let mut fold_metrics = Vec::with_capacity(n_folds);
        let mut fold_custom = Vec::with_capacity(n_folds);

        for fold in 0..n_folds {
            let (train_idx, test_idx) = fold_indices(&folds, fold);
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }

            let train_data = subset_rows(data, &train_idx);
            let train_y = subset_vec(y, &train_idx);
            let test_data = subset_rows(data, &test_idx);
            let test_y = subset_vec(y, &test_idx);

            let model = fit_fn(&train_data, &train_y);
            let preds = predict_fn(&*model, &test_data);

            for (local_i, &orig_i) in test_idx.iter().enumerate() {
                if local_i < preds.len() {
                    oof_preds[orig_i] = preds[local_i];
                }
            }

            fold_metrics.push(compute_metrics(&test_y, &preds, cv_type));
            if !metric_fns.is_empty() {
                fold_custom.push(evaluate_metrics(&test_y, &preds, metric_fns));
            }
        }

        let rep_metric = compute_metrics(y, &oof_preds, cv_type);
        all_oof.push(oof_preds);
        all_rep_metrics.push(rep_metric);
        last_folds = folds;
        last_fold_metrics = fold_metrics;
        last_fold_custom = fold_custom;
    }

    let (final_oof, oof_sd) = aggregate_oof_predictions(all_oof, n);
    let overall_metrics = compute_metrics(y, &final_oof, cv_type);
    let custom_metrics = if metric_fns.is_empty() {
        HashMap::new()
    } else {
        evaluate_metrics(y, &final_oof, metric_fns)
    };

    CvFdataResult {
        oof_predictions: final_oof,
        metrics: overall_metrics,
        fold_metrics: last_fold_metrics,
        folds: last_folds,
        cv_type,
        nrep,
        oof_sd,
        rep_metrics: if nrep > 1 {
            Some(all_rep_metrics)
        } else {
            None
        },
        custom_metrics,
        fold_custom_metrics: last_fold_custom,
    }
}

/// Compute metrics from true and predicted values.
fn compute_metrics(y_true: &[f64], y_pred: &[f64], cv_type: CvType) -> CvMetrics {
    let n = y_true.len().min(y_pred.len());
    if n == 0 {
        return match cv_type {
            CvType::Regression => CvMetrics::Regression {
                rmse: f64::NAN,
                mae: f64::NAN,
                r_squared: f64::NAN,
            },
            CvType::Classification => CvMetrics::Classification {
                accuracy: 0.0,
                confusion: Vec::new(),
            },
        };
    }

    match cv_type {
        CvType::Regression => {
            let mean_y = y_true.iter().sum::<f64>() / n as f64;
            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;
            let mut mae_sum = 0.0;
            for i in 0..n {
                let resid = y_true[i] - y_pred[i];
                ss_res += resid * resid;
                ss_tot += (y_true[i] - mean_y).powi(2);
                mae_sum += resid.abs();
            }
            let rmse = (ss_res / n as f64).sqrt();
            let mae = mae_sum / n as f64;
            let r_squared = if ss_tot > 1e-15 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };
            CvMetrics::Regression {
                rmse,
                mae,
                r_squared,
            }
        }
        CvType::Classification => {
            let n_classes = y_true
                .iter()
                .chain(y_pred.iter())
                .map(|&v| v as usize)
                .max()
                .unwrap_or(0)
                + 1;
            let mut confusion = vec![vec![0usize; n_classes]; n_classes];
            let mut correct = 0usize;
            for i in 0..n {
                let true_c = y_true[i] as usize;
                let pred_c = y_pred[i].round() as usize;
                if true_c < n_classes && pred_c < n_classes {
                    confusion[true_c][pred_c] += 1;
                }
                if true_c == pred_c {
                    correct += 1;
                }
            }
            let accuracy = correct as f64 / n as f64;
            CvMetrics::Classification {
                accuracy,
                confusion,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FdarError;

    #[test]
    fn test_create_folds_basic() {
        let folds = create_folds(10, 5, 42);
        assert_eq!(folds.len(), 10);
        // Each fold should have 2 members
        for f in 0..5 {
            let count = folds.iter().filter(|&&x| x == f).count();
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_create_folds_deterministic() {
        let f1 = create_folds(20, 5, 123);
        let f2 = create_folds(20, 5, 123);
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_stratified_folds() {
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let folds = create_stratified_folds(10, &y, 5, 42);
        assert_eq!(folds.len(), 10);
        // Each fold should have 1 from each class
        for f in 0..5 {
            let class0_count = (0..10).filter(|&i| folds[i] == f && y[i] == 0).count();
            let class1_count = (0..10).filter(|&i| folds[i] == f && y[i] == 1).count();
            assert_eq!(class0_count, 1);
            assert_eq!(class1_count, 1);
        }
    }

    #[test]
    fn test_fold_indices() {
        let folds = vec![0, 1, 2, 0, 1, 2];
        let (train, test) = fold_indices(&folds, 1);
        assert_eq!(test, vec![1, 4]);
        assert_eq!(train, vec![0, 2, 3, 5]);
    }

    #[test]
    fn test_subset_rows() {
        let mut data = FdMatrix::zeros(4, 3);
        for i in 0..4 {
            for j in 0..3 {
                data[(i, j)] = (i * 10 + j) as f64;
            }
        }
        let sub = subset_rows(&data, &[1, 3]);
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 3);
        assert!((sub[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((sub[(1, 0)] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_cv_fdata_regression() -> Result<(), FdarError> {
        // Simple test: predict mean
        let n = 20;
        let m = 5;
        let mut data = FdMatrix::zeros(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = y[i] + j as f64 * 0.1;
            }
        }

        let result = cv_fdata(
            &data,
            &y,
            |_train_data, train_y| {
                let mean = train_y.iter().sum::<f64>() / train_y.len() as f64;
                Box::new(mean)
            },
            |model, test_data| {
                let mean = model.downcast_ref::<f64>().unwrap();
                vec![*mean; test_data.nrows()]
            },
            5,
            1,
            CvType::Regression,
            false,
            42,
        );

        assert_eq!(result.oof_predictions.len(), n);
        assert_eq!(result.nrep, 1);
        assert!(result.oof_sd.is_none());
        match &result.metrics {
            CvMetrics::Regression { rmse, .. } => assert!(*rmse > 0.0),
            _ => {
                return Err(FdarError::ComputationFailed {
                    operation: "cv_fdata_regression",
                    detail: "expected regression metrics".into(),
                });
            }
        }
        Ok(())
    }

    #[test]
    fn test_cv_fdata_repeated() {
        let n = 20;
        let m = 3;
        let data = FdMatrix::zeros(n, m);
        let y: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();

        let result = cv_fdata(
            &data,
            &y,
            |_d, _y| Box::new(0.5_f64),
            |_model, test_data| vec![0.5; test_data.nrows()],
            5,
            3,
            CvType::Regression,
            false,
            42,
        );

        assert_eq!(result.nrep, 3);
        assert!(result.oof_sd.is_some());
        assert!(result.rep_metrics.is_some());
        assert_eq!(result.rep_metrics.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_custom_metrics() {
        let n = 20;
        let m = 3;
        let data = FdMatrix::zeros(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let metrics = regression_metrics();
        let result = cv_fdata_with_metrics(
            &data,
            &y,
            |_d, train_y| {
                let mean = train_y.iter().sum::<f64>() / train_y.len() as f64;
                Box::new(mean)
            },
            |model, test_data| {
                let mean = model.downcast_ref::<f64>().unwrap();
                vec![*mean; test_data.nrows()]
            },
            5,
            1,
            CvType::Regression,
            false,
            42,
            &metrics,
        );

        assert!(result.custom_metrics.contains_key("rmse"));
        assert!(result.custom_metrics.contains_key("mae"));
        assert!(result.custom_metrics.contains_key("r_squared"));
        assert!(*result.custom_metrics.get("rmse").unwrap() > 0.0);
        assert_eq!(result.fold_custom_metrics.len(), 5);
    }

    #[test]
    fn test_classification_metrics_standalone() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let y_pred = vec![0.0, 1.0, 1.0, 1.0, 0.0];
        assert!((metric_accuracy(&y_true, &y_pred) - 0.6).abs() < 1e-10);
        assert!((metric_precision(&y_true, &y_pred) - 2.0 / 3.0).abs() < 1e-10); // TP=2, FP=1
        assert!((metric_recall(&y_true, &y_pred) - 2.0 / 3.0).abs() < 1e-10); // TP=2, FN=1
        let f1 = metric_f1(&y_true, &y_pred);
        assert!((f1 - 2.0 / 3.0).abs() < 1e-10); // P=R => F1=P=R
    }

    #[test]
    fn test_compute_metrics_classification() -> Result<(), FdarError> {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 1.0, 1.0, 1.0]; // 1 misclassification
        let m = compute_metrics(&y_true, &y_pred, CvType::Classification);
        match m {
            CvMetrics::Classification {
                accuracy,
                confusion,
            } => {
                assert!((accuracy - 0.75).abs() < 1e-10);
                assert_eq!(confusion[0][0], 1); // true 0, pred 0
                assert_eq!(confusion[0][1], 1); // true 0, pred 1
                assert_eq!(confusion[1][1], 2); // true 1, pred 1
            }
            _ => {
                return Err(FdarError::ComputationFailed {
                    operation: "compute_metrics_classification",
                    detail: "expected classification metrics".into(),
                });
            }
        }
        Ok(())
    }
}
