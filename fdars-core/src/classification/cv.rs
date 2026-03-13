//! Cross-validation for functional classification.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

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

    let mut fold_errors = Vec::with_capacity(nfold);

    for fold in 0..nfold {
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
        let errors = match predictions {
            Some(pred) => {
                let wrong = pred
                    .iter()
                    .zip(&test_labels)
                    .filter(|(&p, &t)| p != t)
                    .count();
                wrong as f64 / n_test as f64
            }
            None => 1.0,
        };
        fold_errors.push(errors);
    }

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
    let fpca = fdata_to_pc_1d(train_data, ncomp).ok()?;
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
        "kernel" | "dd" => None,
        _ => None,
    }
}

/// Project test data onto FPCA basis (mean-center, multiply by rotation).
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
                score += (test_data[(i, j)] - fpca.mean[j]) * fpca.rotation[(j, k)];
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
