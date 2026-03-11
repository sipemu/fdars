//! Cross-validation utilities and unified CV framework.
//!
//! This module provides:
//! - Shared fold assignment utilities used across all CV functions
//! - [`cv_fdata`]: Generic k-fold + repeated CV framework (R's `cv.fdata`)

use crate::matrix::FdMatrix;
use rand::prelude::*;
use std::any::Any;

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
pub enum CvType {
    Regression,
    Classification,
}

/// Cross-validation metrics.
#[derive(Debug, Clone)]
pub enum CvMetrics {
    /// Regression metrics.
    Regression { rmse: f64, mae: f64, r_squared: f64 },
    /// Classification metrics.
    Classification {
        accuracy: f64,
        confusion: Vec<Vec<usize>>,
    },
}

/// Result of unified cross-validation.
#[derive(Debug, Clone)]
pub struct CvFdataResult {
    /// Out-of-fold predictions (length n); for repeated CV, averaged across reps.
    pub oof_predictions: Vec<f64>,
    /// Overall metrics.
    pub metrics: CvMetrics,
    /// Per-fold metrics.
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
}

// ─── Unified CV Framework ────────────────────────────────────────────────────

/// Generic k-fold + repeated cross-validation framework (R's `cv.fdata`).
///
/// The user provides fit/predict closures so this works with any model.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `y` — Response vector (length n); for classification, should be 0, 1, 2, …
/// * `fit_fn` — Closure that fits a model on training data and returns a boxed model
/// * `predict_fn` — Closure that predicts from a model on test data
/// * `n_folds` — Number of CV folds
/// * `nrep` — Number of repetitions (1 = single CV, >1 = repeated)
/// * `cv_type` — Whether this is regression or classification
/// * `stratified` — Whether to stratify folds
/// * `seed` — Random seed for fold assignment
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
    let n = data.nrows();
    let nrep = nrep.max(1);
    let n_folds = n_folds.max(2).min(n);

    // Storage for repeated CV
    let mut all_oof: Vec<Vec<f64>> = Vec::with_capacity(nrep);
    let mut all_rep_metrics: Vec<CvMetrics> = Vec::with_capacity(nrep);
    let mut last_folds = vec![0usize; n];
    let mut last_fold_metrics = Vec::new();

    for r in 0..nrep {
        let rep_seed = seed.wrapping_add(r as u64);

        // Create folds
        let folds = if stratified {
            match cv_type {
                CvType::Classification => {
                    let y_class: Vec<usize> = y.iter().map(|&v| v as usize).collect();
                    create_stratified_folds(n, &y_class, n_folds, rep_seed)
                }
                CvType::Regression => {
                    // Stratify by quantile bin
                    let mut sorted_y: Vec<(usize, f64)> = y.iter().copied().enumerate().collect();
                    sorted_y
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    let n_bins = n_folds.min(n);
                    let bin_labels: Vec<usize> = {
                        let mut labels = vec![0usize; n];
                        for (rank, &(orig_i, _)) in sorted_y.iter().enumerate() {
                            labels[orig_i] = (rank * n_bins / n).min(n_bins - 1);
                        }
                        labels
                    };
                    create_stratified_folds(n, &bin_labels, n_folds, rep_seed)
                }
            }
        } else {
            create_folds(n, n_folds, rep_seed)
        };

        let mut oof_preds = vec![0.0; n];
        let mut fold_metrics = Vec::with_capacity(n_folds);

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
        }

        let rep_metric = compute_metrics(y, &oof_preds, cv_type);
        all_oof.push(oof_preds);
        all_rep_metrics.push(rep_metric);
        last_folds = folds;
        last_fold_metrics = fold_metrics;
    }

    // Aggregate across repetitions
    let (final_oof, oof_sd) = if nrep == 1 {
        (all_oof.into_iter().next().unwrap(), None)
    } else {
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
    };

    let overall_metrics = compute_metrics(y, &final_oof, cv_type);

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
    fn test_cv_fdata_regression() {
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
            _ => panic!("Expected regression metrics"),
        }
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
    fn test_compute_metrics_classification() {
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
            _ => panic!("Expected classification metrics"),
        }
    }
}
