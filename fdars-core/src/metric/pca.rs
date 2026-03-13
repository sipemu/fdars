//! PCA-based semimetric for functional data.
//!
//! Computes distances between functional observations in the space of
//! principal component scores, retaining only the first `ncomp` components.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use nalgebra::SVD;

use super::{cross_distance_matrix, self_distance_matrix};

/// Compute column means and return centered data as a nalgebra `DMatrix`.
fn center_data(data: &FdMatrix) -> (nalgebra::DMatrix<f64>, Vec<f64>) {
    let n = data.nrows();
    let m = data.ncols();
    let mut means = vec![0.0; m];
    for j in 0..m {
        let mut sum = 0.0;
        for i in 0..n {
            sum += data[(i, j)];
        }
        means[j] = sum / n as f64;
    }
    let mut centered = nalgebra::DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            centered[(i, j)] = data[(i, j)] - means[j];
        }
    }
    (centered, means)
}

/// Project rows of `data` onto the first `ncomp` right-singular vectors of `v_t`.
///
/// Returns an `n x ncomp` score matrix stored as `Vec<Vec<f64>>`.
fn project_scores(
    data: &FdMatrix,
    means: &[f64],
    v_t: &nalgebra::DMatrix<f64>,
    ncomp: usize,
) -> Vec<Vec<f64>> {
    let n = data.nrows();
    let m = data.ncols();
    (0..n)
        .map(|i| {
            (0..ncomp)
                .map(|k| {
                    let mut s = 0.0;
                    for j in 0..m {
                        // v_t row k = k-th right singular vector
                        s += (data[(i, j)] - means[j]) * v_t[(k, j)];
                    }
                    s
                })
                .collect()
        })
        .collect()
}

/// Euclidean distance between two score vectors.
fn score_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the PCA-based semimetric self-distance matrix.
///
/// Each curve is projected onto the first `ncomp` principal components
/// (derived from the SVD of the centered data matrix), and the Euclidean
/// distance in score space is returned.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero or exceeds
/// `min(n, m)`.
///
/// Returns [`FdarError::InvalidDimension`] if data has fewer than 2 rows.
///
/// Returns [`FdarError::ComputationFailed`] if the SVD fails to produce
/// the required matrices.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::pca_self_1d;
///
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 10,
/// ).unwrap();
/// let dist = pca_self_1d(&data, 2).unwrap();
/// assert_eq!(dist.shape(), (5, 5));
/// assert!(dist[(0, 0)].abs() < 1e-10);
/// ```
pub fn pca_self_1d(data: &FdMatrix, ncomp: usize) -> Result<FdMatrix, FdarError> {
    let n = data.nrows();
    let m = data.ncols();

    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if ncomp == 0 || ncomp > n.min(m) {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("must be in 1..={}, got {ncomp}", n.min(m)),
        });
    }

    let (centered, _means) = center_data(data);
    let svd = SVD::new(centered, true, true);
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "SVD",
            detail: "failed to compute V^T matrix".to_string(),
        })?;
    let u = svd.u.as_ref().ok_or_else(|| FdarError::ComputationFailed {
        operation: "SVD",
        detail: "failed to compute U matrix".to_string(),
    })?;

    // Scores = U * Sigma (first ncomp columns)
    let scores: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..ncomp)
                .map(|k| u[(i, k)] * svd.singular_values[k])
                .collect()
        })
        .collect();

    // Suppress unused variable warning — v_t is validated above but scores
    // are computed from U * Sigma for efficiency.
    let _ = v_t;

    Ok(self_distance_matrix(n, |i, j| {
        score_distance(&scores[i], &scores[j])
    }))
}

/// Compute the PCA-based semimetric cross-distance matrix.
///
/// Principal components are derived from `data1`. Both datasets are then
/// projected onto those components, and pairwise Euclidean distances in
/// score space form the returned `n1 x n2` matrix.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero or exceeds
/// `min(n1, m)`.
///
/// Returns [`FdarError::InvalidDimension`] if `data1` has fewer than 2 rows,
/// or if the two datasets have different numbers of columns.
///
/// Returns [`FdarError::ComputationFailed`] if the SVD fails.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::pca_cross_1d;
///
/// let data1 = FdMatrix::from_column_major(
///     (0..30).map(|i| (i as f64 * 0.1).sin()).collect(), 3, 10,
/// ).unwrap();
/// let data2 = FdMatrix::from_column_major(
///     (0..20).map(|i| (i as f64 * 0.2).cos()).collect(), 2, 10,
/// ).unwrap();
/// let dist = pca_cross_1d(&data1, &data2, 2).unwrap();
/// assert_eq!(dist.shape(), (3, 2));
/// ```
pub fn pca_cross_1d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    ncomp: usize,
) -> Result<FdMatrix, FdarError> {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();

    if n1 < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data1",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n1} rows"),
        });
    }
    if data2.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data2",
            expected: format!("{m} columns (matching data1)"),
            actual: format!("{} columns", data2.ncols()),
        });
    }
    if ncomp == 0 || ncomp > n1.min(m) {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("must be in 1..={}, got {ncomp}", n1.min(m)),
        });
    }

    let (centered, means) = center_data(data1);
    let svd = SVD::new(centered, true, true);
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "SVD",
            detail: "failed to compute V^T matrix".to_string(),
        })?;

    // Scores for data1: project centered data1 onto PCs
    let scores1 = project_scores(data1, &means, v_t, ncomp);
    // Scores for data2: project centered (using data1 means) data2 onto data1 PCs
    let scores2 = project_scores(data2, &means, v_t, ncomp);

    Ok(cross_distance_matrix(n1, n2, |i, j| {
        score_distance(&scores1[i], &scores2[j])
    }))
}
