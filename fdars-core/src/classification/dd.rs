//! Depth-based DD-classifier.

use crate::depth::fraiman_muniz_1d;
use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::cv::extract_class_data;
use super::kernel::{argmax_class, scalar_depth_for_obs};
use super::{compute_accuracy, confusion_matrix, remap_labels, ClassifResult};

/// Compute depth of all observations w.r.t. each class.
fn compute_class_depths(data: &FdMatrix, class_indices: &[Vec<usize>], n: usize) -> FdMatrix {
    let g = class_indices.len();
    let mut depth_scores = FdMatrix::zeros(n, g);
    for c in 0..g {
        if class_indices[c].is_empty() {
            continue;
        }
        let class_data = extract_class_data(data, &class_indices[c]);
        let depths = fraiman_muniz_1d(data, &class_data, true);
        for i in 0..n {
            depth_scores[(i, c)] = depths[i];
        }
    }
    depth_scores
}

/// Blend functional depth scores with scalar rank depth from covariates.
pub(super) fn blend_scalar_depths(
    depth_scores: &mut FdMatrix,
    cov: &FdMatrix,
    class_indices: &[Vec<usize>],
    n: usize,
) {
    let g = class_indices.len();
    let p = cov.ncols();
    for c in 0..g {
        for i in 0..n {
            let sd = scalar_depth_for_obs(cov, i, &class_indices[c], p);
            depth_scores[(i, c)] = 0.7 * depth_scores[(i, c)] + 0.3 * sd;
        }
    }
}

/// Depth-based DD-classifier for functional data.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_dd(
    data: &FdMatrix,
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
) -> Result<ClassifResult, FdarError> {
    let n = data.nrows();
    if n == 0 || y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y",
            expected: "n > 0 and y.len() == n".to_string(),
            actual: format!("n={}, y.len()={}", n, y.len()),
        });
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("need at least 2 classes, got {g}"),
        });
    }

    let class_indices: Vec<Vec<usize>> = (0..g)
        .map(|c| (0..n).filter(|&i| labels[i] == c).collect())
        .collect();

    let mut depth_scores = compute_class_depths(data, &class_indices, n);

    if let Some(cov) = scalar_covariates {
        blend_scalar_depths(&mut depth_scores, cov, &class_indices, n);
    }

    let predicted: Vec<usize> = (0..n)
        .map(|i| {
            let scores: Vec<f64> = (0..g).map(|c| depth_scores[(i, c)]).collect();
            argmax_class(&scores)
        })
        .collect();

    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Ok(ClassifResult {
        predicted,
        probabilities: Some(depth_scores),
        accuracy,
        confusion,
        n_classes: g,
        ncomp: 0,
    })
}
