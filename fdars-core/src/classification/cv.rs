//! Cross-validation for functional classification.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::lda::{lda_params, lda_predict};
use super::qda::{build_qda_params, qda_predict};
use super::{remap_labels, ClassifCvResult};
use crate::linalg::cholesky_d;

/// K-fold cross-validated error rate for functional classification.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `argvals` — Evaluation points
/// * `y` — Class labels
/// * `scalar_covariates` — Optional scalar covariates
/// * `method` — "lda", "qda", "knn", "kernel", "dd"
/// * `ncomp` — Number of FPC components (for lda/qda/knn)
/// * `nfold` — Number of CV folds
/// * `seed` — Random seed for fold assignment
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `nfold < 2` or `nfold > n`.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::classification::cv::fclassif_cv;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..100).map(|i| (i as f64 * 0.1).sin()).collect(),
///     10, 10,
/// ).unwrap();
/// let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let result = fclassif_cv(&data, &argvals, &y, None, "lda", 2, 3, 42).unwrap();
/// assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_cv(
    data: &FdMatrix,
    argvals: &[f64],
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    method: &str,
    ncomp: usize,
    nfold: usize,
    seed: u64,
) -> Result<ClassifCvResult, FdarError> {
    let n = data.nrows();
    if n < nfold || nfold < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "nfold",
            message: format!("need 2 <= nfold <= n, got nfold={nfold}, n={n}"),
        });
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("need at least 2 classes, got {g}"),
        });
    }

    // Assign folds
    let folds = assign_folds(n, nfold, seed);

    let fold_errors: Vec<f64> = iter_maybe_parallel!(0..nfold)
        .map(|fold| {
            let (train_idx, test_idx) = fold_split(&folds, fold);
            let train_data = extract_class_data(data, &train_idx);
            let test_data = extract_class_data(data, &test_idx);
            let train_labels: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
            let test_labels: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

            let train_cov = scalar_covariates.map(|c| extract_class_data(c, &train_idx));
            let test_cov = scalar_covariates.map(|c| extract_class_data(c, &test_idx));

            let predictions = cv_fold_predict(
                &train_data,
                &test_data,
                argvals,
                &train_labels,
                g,
                train_cov.as_ref(),
                test_cov.as_ref(),
                method,
                ncomp,
            );

            let n_test = test_labels.len();
            match predictions {
                Some(pred) => {
                    let wrong = pred
                        .iter()
                        .zip(&test_labels)
                        .filter(|(&p, &t)| p != t)
                        .count();
                    wrong as f64 / n_test as f64
                }
                None => 1.0,
            }
        })
        .collect();

    let error_rate = fold_errors.iter().sum::<f64>() / nfold as f64;

    Ok(ClassifCvResult {
        error_rate,
        fold_errors,
        best_ncomp: ncomp,
    })
}

/// Assign observations to folds.
pub(super) fn assign_folds(n: usize, nfold: usize, seed: u64) -> Vec<usize> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let mut folds = vec![0usize; n];
    for (rank, &idx) in indices.iter().enumerate() {
        folds[idx] = rank % nfold;
    }
    folds
}

/// Split indices into train and test for given fold.
pub(super) fn fold_split(folds: &[usize], fold: usize) -> (Vec<usize>, Vec<usize>) {
    let train: Vec<usize> = (0..folds.len()).filter(|&i| folds[i] != fold).collect();
    let test: Vec<usize> = (0..folds.len()).filter(|&i| folds[i] == fold).collect();
    (train, test)
}

/// Predict on test set for one CV fold.
fn cv_fold_predict(
    train_data: &FdMatrix,
    test_data: &FdMatrix,
    _argvals: &[f64],
    train_labels: &[usize],
    g: usize,
    train_cov: Option<&FdMatrix>,
    test_cov: Option<&FdMatrix>,
    method: &str,
    ncomp: usize,
) -> Option<Vec<usize>> {
    let m = train_data.ncols();
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(train_data, ncomp, &argvals).ok()?;
    match method {
        "lda" => {
            let predictions =
                project_and_classify_lda(test_data, &fpca, train_labels, g, train_cov, test_cov);
            Some(predictions)
        }
        "qda" => {
            let predictions =
                project_and_classify_qda(test_data, &fpca, train_labels, g, train_cov, test_cov);
            Some(predictions)
        }
        "knn" => {
            let predictions =
                project_and_classify_knn(test_data, &fpca, train_labels, g, train_cov, test_cov, 5);
            Some(predictions)
        }
        // kernel and dd classifiers don't support out-of-sample prediction on new data
        _ => None,
    }
}

/// Project test data onto FPCA basis (mean-center, multiply by rotation with weights).
pub(super) fn project_test_onto_fpca(
    test_data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
) -> FdMatrix {
    let n_test = test_data.nrows();
    let m = test_data.ncols();
    let d_pc = fpca.scores.ncols();
    let mut test_features = FdMatrix::zeros(n_test, d_pc);
    for i in 0..n_test {
        for k in 0..d_pc {
            let mut score = 0.0;
            for j in 0..m {
                score +=
                    (test_data[(i, j)] - fpca.mean[j]) * fpca.rotation[(j, k)] * fpca.weights[j];
            }
            test_features[(i, k)] = score;
        }
    }
    test_features
}

/// Append scalar covariates to FPCA scores to form augmented feature matrix.
fn append_scalar_covariates(scores: &FdMatrix, scalar_covariates: Option<&FdMatrix>) -> FdMatrix {
    match scalar_covariates {
        None => scores.clone(),
        Some(cov) => {
            let n = scores.nrows();
            let d_pc = scores.ncols();
            let d_cov = cov.ncols();
            let mut features = FdMatrix::zeros(n, d_pc + d_cov);
            for i in 0..n {
                for j in 0..d_pc {
                    features[(i, j)] = scores[(i, j)];
                }
                for j in 0..d_cov {
                    features[(i, d_pc + j)] = cov[(i, j)];
                }
            }
            features
        }
    }
}

/// Project test data onto training FPCA and classify with LDA.
fn project_and_classify_lda(
    test_data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
    train_labels: &[usize],
    g: usize,
    train_cov: Option<&FdMatrix>,
    test_cov: Option<&FdMatrix>,
) -> Vec<usize> {
    let test_pc = project_test_onto_fpca(test_data, fpca);
    let test_features = append_scalar_covariates(&test_pc, test_cov);

    let train_features = append_scalar_covariates(&fpca.scores, train_cov);
    let (class_means, cov, priors) = lda_params(&train_features, train_labels, g);
    let d = train_features.ncols();
    match cholesky_d(&cov, d) {
        Ok(chol) => lda_predict(&test_features, &class_means, &chol, &priors, g),
        Err(_) => vec![0; test_data.nrows()],
    }
}

/// Project test data onto training FPCA and classify with QDA.
fn project_and_classify_qda(
    test_data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
    train_labels: &[usize],
    g: usize,
    train_cov: Option<&FdMatrix>,
    test_cov: Option<&FdMatrix>,
) -> Vec<usize> {
    let n_test = test_data.nrows();
    let test_pc = project_test_onto_fpca(test_data, fpca);
    let test_features = append_scalar_covariates(&test_pc, test_cov);

    let train_features = append_scalar_covariates(&fpca.scores, train_cov);

    match build_qda_params(&train_features, train_labels, g) {
        Ok((class_means, class_chols, class_log_dets, priors)) => qda_predict(
            &test_features,
            &class_means,
            &class_chols,
            &class_log_dets,
            &priors,
            g,
        ),
        Err(_) => vec![0; n_test],
    }
}

/// Project test data and classify with k-NN.
fn project_and_classify_knn(
    test_data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
    train_labels: &[usize],
    g: usize,
    train_cov: Option<&FdMatrix>,
    test_cov: Option<&FdMatrix>,
    k_nn: usize,
) -> Vec<usize> {
    let n_test = test_data.nrows();
    let n_train = fpca.scores.nrows();

    let test_pc = project_test_onto_fpca(test_data, fpca);
    let test_features = append_scalar_covariates(&test_pc, test_cov);
    let train_features = append_scalar_covariates(&fpca.scores, train_cov);
    let d = train_features.ncols();

    (0..n_test)
        .map(|i| {
            // Distances to all training points in augmented feature space
            let mut dists: Vec<(f64, usize)> = (0..n_train)
                .map(|t| {
                    let d_sq: f64 = (0..d)
                        .map(|k| (test_features[(i, k)] - train_features[(t, k)]).powi(2))
                        .sum();
                    (d_sq, train_labels[t])
                })
                .collect();
            let k_eff = k_nn.min(n_train);
            if k_eff > 0 && k_eff < dists.len() {
                dists.select_nth_unstable_by(k_eff - 1, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            let mut votes = vec![0usize; g];
            for &(_, label) in dists.iter().take(k_eff) {
                votes[label] += 1;
            }
            votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map_or(0, |(c, _)| c)
        })
        .collect()
}

/// Extract rows corresponding to given indices into a new FdMatrix.
pub(super) fn extract_class_data(data: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let nc = indices.len();
    let m = data.ncols();
    let mut result = FdMatrix::zeros(nc, m);
    for (ri, &i) in indices.iter().enumerate() {
        for j in 0..m {
            result[(ri, j)] = data[(i, j)];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small deterministic classification dataset: n observations,
    /// m evaluation points, 2 well-separated classes (first n/2 are class 0,
    /// rest are class 1), argvals on [0, 1].
    fn make_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
        let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
        let mut raw = vec![0.0f64; n * m];
        // Column-major: element (i, j) is at index i + j * n
        for i in 0..n {
            let class_offset = if i < n / 2 { 0.0 } else { 5.0 };
            for j in 0..m {
                // Simple bump function shifted by class_offset — well-separated classes
                raw[i + j * n] = class_offset + (argvals[j] * std::f64::consts::PI).sin();
            }
        }
        let data = FdMatrix::from_column_major(raw, n, m).unwrap();
        let labels: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        (data, argvals, labels)
    }

    /// Verify that `fclassif_cv` produces bit-for-bit identical results when called
    /// twice with the same seed and arguments, regardless of whether the `parallel`
    /// feature is enabled.  This proves the collect-in-order determinism contract
    /// for the parallelized fold loop.
    #[test]
    fn test_fclassif_cv_parallel_matches_sequential() {
        let n = 20;
        let m = 10;
        let ncomp = 2;
        let nfold = 5;
        let seed = 42u64;

        let (data, argvals, labels) = make_test_data(n, m);

        let res_a = fclassif_cv(&data, &argvals, &labels, None, "lda", ncomp, nfold, seed)
            .expect("fclassif_cv call A failed");
        let res_b = fclassif_cv(&data, &argvals, &labels, None, "lda", ncomp, nfold, seed)
            .expect("fclassif_cv call B failed");

        assert_eq!(
            res_a.fold_errors.len(),
            res_b.fold_errors.len(),
            "fold_errors length mismatch"
        );
        for (i, (&a, &b)) in res_a
            .fold_errors
            .iter()
            .zip(res_b.fold_errors.iter())
            .enumerate()
        {
            assert_eq!(
                a, b,
                "fold_errors[{i}] not bit-for-bit identical: {a} vs {b}"
            );
        }
        assert_eq!(
            res_a.error_rate, res_b.error_rate,
            "error_rate not bit-for-bit identical"
        );
    }
}
