//! k-NN classifier internals.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    build_feature_matrix, compute_accuracy, confusion_matrix, remap_labels, ClassifResult,
};

/// FPC + k-NN classification.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Class labels
/// * `scalar_covariates` — Optional scalar covariates
/// * `ncomp` — Number of FPC components
/// * `k_nn` — Number of nearest neighbors
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::InvalidParameter`] if `k_nn` is zero.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
/// Returns [`FdarError::ComputationFailed`] if the SVD decomposition in FPCA fails.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::classification::knn::fclassif_knn;
///
/// let data = FdMatrix::from_column_major(
///     (0..100).map(|i| (i as f64 * 0.1).sin()).collect(),
///     10, 10,
/// ).unwrap();
/// let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let result = fclassif_knn(&data, &y, None, 3, 3).unwrap();
/// assert_eq!(result.predicted.len(), 10);
/// assert_eq!(result.n_classes, 2);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_knn(
    data: &FdMatrix,
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    k_nn: usize,
) -> Result<ClassifResult, FdarError> {
    let n = data.nrows();
    if n == 0 || y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y",
            expected: "n > 0 and y.len() == n".to_string(),
            actual: format!("n={}, y.len()={}", n, y.len()),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".to_string(),
        });
    }
    if k_nn == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k_nn",
            message: "must be > 0".to_string(),
        });
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("need at least 2 classes, got {g}"),
        });
    }

    let (features, _mean, _rotation, _weights) =
        build_feature_matrix(data, scalar_covariates, ncomp)?;
    let d = features.ncols();

    let predicted = knn_predict_loo(&features, &labels, g, d, k_nn);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Ok(ClassifResult {
        predicted,
        probabilities: None,
        accuracy,
        confusion,
        n_classes: g,
        ncomp: d.min(ncomp),
    })
}

/// k-NN classification from a precomputed distance matrix.
///
/// Works with **any** distance matrix (elastic, DTW, Lp, or custom).
/// Labels are 0-indexed class indices.
///
/// # Arguments
/// * `dist_mat` — Symmetric n × n distance matrix
/// * `y` — Class labels (length n, 0-indexed)
/// * `k_nn` — Number of nearest neighbors
///
/// # Errors
/// Returns errors if `dist_mat` is not square, `y.len() != n`, `k_nn == 0`, or fewer than 2 classes.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn knn_classify_from_distances(
    dist_mat: &FdMatrix,
    y: &[usize],
    k_nn: usize,
) -> Result<ClassifResult, FdarError> {
    let n = dist_mat.nrows();
    if dist_mat.ncols() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "dist_mat",
            expected: format!("{n} x {n} (square)"),
            actual: format!("{} x {}", n, dist_mat.ncols()),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if k_nn == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k_nn",
            message: "must be > 0".to_string(),
        });
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("need at least 2 classes, got {g}"),
        });
    }

    let k_nn = k_nn.min(n - 1);
    let predicted: Vec<usize> = (0..n)
        .map(|i| {
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (dist_mat[(i, j)], labels[j]))
                .collect();
            if k_nn > 0 && k_nn < dists.len() {
                dists.select_nth_unstable_by(k_nn - 1, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let mut votes = vec![0usize; g];
            for &(_, label) in dists.iter().take(k_nn) {
                votes[label] += 1;
            }
            votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map_or(0, |(c, _)| c)
        })
        .collect();

    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Ok(ClassifResult {
        predicted,
        probabilities: None,
        accuracy,
        confusion,
        n_classes: g,
        ncomp: 0,
    })
}

/// Leave-one-out k-NN prediction.
pub(crate) fn knn_predict_loo(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
    d: usize,
    k_nn: usize,
) -> Vec<usize> {
    let n = features.nrows();
    let k_nn = k_nn.min(n - 1);

    (0..n)
        .map(|i| {
            let xi: Vec<f64> = (0..d).map(|j| features[(i, j)]).collect();
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let xj: Vec<f64> = (0..d).map(|jj| features[(j, jj)]).collect();
                    let d_sq: f64 = xi.iter().zip(&xj).map(|(&a, &b)| (a - b).powi(2)).sum();
                    (d_sq, labels[j])
                })
                .collect();
            if k_nn > 0 && k_nn < dists.len() {
                dists.select_nth_unstable_by(k_nn - 1, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            // Majority vote among k nearest
            let mut votes = vec![0usize; g];
            for &(_, label) in dists.iter().take(k_nn) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    #[test]
    fn knn_from_distances_smoke() {
        // 6 points: 3 in class 0 (close together), 3 in class 1 (close together)
        let mut dist = FdMatrix::zeros(6, 6);
        // Within class 0 (indices 0,1,2): small distances
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    dist[(i, j)] = 0.1;
                }
            }
        }
        // Within class 1 (indices 3,4,5): small distances
        for i in 3..6 {
            for j in 3..6 {
                if i != j {
                    dist[(i, j)] = 0.1;
                }
            }
        }
        // Between classes: large distances
        for i in 0..3 {
            for j in 3..6 {
                dist[(i, j)] = 5.0;
                dist[(j, i)] = 5.0;
            }
        }

        let y = vec![0, 0, 0, 1, 1, 1];
        let result = knn_classify_from_distances(&dist, &y, 3).unwrap();
        assert_eq!(result.predicted, vec![0, 0, 0, 1, 1, 1]);
        assert!((result.accuracy - 1.0).abs() < 1e-10);
    }
}
