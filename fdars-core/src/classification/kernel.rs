//! Nonparametric kernel classifier with mixed predictors.

use crate::error::FdarError;
use crate::helpers::{l2_distance, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;

use super::{compute_accuracy, confusion_matrix, remap_labels, ClassifResult};

/// Find class with maximum score.
pub(super) fn argmax_class(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(c, _)| c)
}

/// Compute marginal rank-based scalar depth of observation i w.r.t. class c.
pub(super) fn scalar_depth_for_obs(
    cov: &FdMatrix,
    i: usize,
    class_indices: &[usize],
    p: usize,
) -> f64 {
    let nc = class_indices.len() as f64;
    if nc < 1.0 || p == 0 {
        return 0.0;
    }
    let mut depth = 0.0;
    for j in 0..p {
        let val = cov[(i, j)];
        let rank = class_indices
            .iter()
            .filter(|&&k| cov[(k, j)] <= val)
            .count() as f64;
        let u = rank / nc.max(1.0);
        depth += u.min(1.0 - u).min(0.5);
    }
    depth / p as f64
}

/// Generate bandwidth candidates from distance percentiles.
pub(super) fn bandwidth_candidates(dists: &[f64], n: usize) -> Vec<f64> {
    let mut all_dists: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            all_dists.push(dists[i * n + j]);
        }
    }
    crate::helpers::sort_nan_safe(&mut all_dists);

    (1..=20)
        .map(|p| {
            let idx = (f64::from(p) / 20.0 * (all_dists.len() - 1) as f64) as usize;
            all_dists[idx.min(all_dists.len() - 1)]
        })
        .filter(|&h| h > 1e-15)
        .collect()
}

/// LOO classification accuracy for a single bandwidth.
fn loo_accuracy_for_bandwidth(dists: &[f64], labels: &[usize], g: usize, n: usize, h: f64) -> f64 {
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let correct = iter_maybe_parallel!(0..n)
        .filter(|&i| {
            let mut votes = vec![0.0; g];
            for j in 0..n {
                if j != i {
                    votes[labels[j]] += gaussian_kernel(dists[i * n + j], h);
                }
            }
            argmax_class(&votes) == labels[i]
        })
        .count();
    correct as f64 / n as f64
}

/// Gaussian kernel: exp(-d²/(2h²)).
pub(super) fn gaussian_kernel(dist: f64, h: f64) -> f64 {
    if h < 1e-15 {
        return 0.0;
    }
    (-dist * dist / (2.0 * h * h)).exp()
}

/// Nonparametric kernel classifier for functional data with optional scalar covariates.
///
/// Uses product kernel: K_func × K_scalar. Bandwidth selected by LOO-CV.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Class labels
/// * `argvals` — Evaluation points
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `h_func` — Functional bandwidth (0 = auto via LOO-CV)
/// * `h_scalar` — Scalar bandwidth (0 = auto)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, `y.len() != n`,
/// or `argvals.len() != m`.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_kernel(
    data: &FdMatrix,
    y: &[usize],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    h_func: f64,
    h_scalar: f64,
) -> Result<ClassifResult, FdarError> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || y.len() != n || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y/argvals",
            expected: "n > 0, y.len() == n, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, y.len()={}, m={}, argvals.len()={}",
                n,
                y.len(),
                m,
                argvals.len()
            ),
        });
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("need at least 2 classes, got {g}"),
        });
    }

    let weights = simpsons_weights(argvals);

    // Compute pairwise functional distances
    let func_dists = compute_pairwise_l2(data, &weights);

    // Compute pairwise scalar distances if covariates exist
    let scalar_dists = scalar_covariates.map(compute_pairwise_scalar);

    // Select bandwidths via LOO if needed
    let h_f = if h_func > 0.0 {
        h_func
    } else {
        select_bandwidth_loo(&func_dists, &labels, g, n, true)
    };
    let h_s = match &scalar_dists {
        Some(sd) if h_scalar <= 0.0 => select_bandwidth_loo(sd, &labels, g, n, false),
        _ => h_scalar,
    };

    let predicted = kernel_classify_loo(
        &func_dists,
        scalar_dists.as_deref(),
        &labels,
        g,
        n,
        h_f,
        h_s,
    );
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

/// Compute pairwise L2 distances between curves.
fn compute_pairwise_l2(data: &FdMatrix, weights: &[f64]) -> Vec<f64> {
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let n = data.nrows();
    // Build upper-triangle pair list, compute distances in parallel, then scatter.
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let pair_dists: Vec<(usize, usize, f64)> = iter_maybe_parallel!(pairs)
        .map(|(i, j)| {
            let ri = data.row(i);
            let rj = data.row(j);
            (i, j, l2_distance(&ri, &rj, weights))
        })
        .collect();
    let mut dists = vec![0.0; n * n];
    for (i, j, d) in pair_dists {
        dists[i * n + j] = d;
        dists[j * n + i] = d;
    }
    dists
}

/// Compute pairwise Euclidean distances between scalar covariate vectors.
pub(super) fn compute_pairwise_scalar(scalar_covariates: &FdMatrix) -> Vec<f64> {
    let n = scalar_covariates.nrows();
    let p = scalar_covariates.ncols();
    let mut dists = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d_sq = 0.0;
            for k in 0..p {
                d_sq += (scalar_covariates[(i, k)] - scalar_covariates[(j, k)]).powi(2);
            }
            let d = d_sq.sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Select bandwidth by LOO classification accuracy.
pub(super) fn select_bandwidth_loo(
    dists: &[f64],
    labels: &[usize],
    g: usize,
    n: usize,
    is_func: bool,
) -> f64 {
    let candidates = bandwidth_candidates(dists, n);
    if candidates.is_empty() {
        return if is_func { 1.0 } else { 0.5 };
    }

    let mut best_h = candidates[0];
    let mut best_acc = 0.0;
    for &h in &candidates {
        let acc = loo_accuracy_for_bandwidth(dists, labels, g, n, h);
        if acc > best_acc {
            best_acc = acc;
            best_h = h;
        }
    }
    best_h
}

/// LOO kernel classification with product kernel.
fn kernel_classify_loo(
    func_dists: &[f64],
    scalar_dists: Option<&[f64]>,
    labels: &[usize],
    g: usize,
    n: usize,
    h_func: f64,
    h_scalar: f64,
) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let mut votes = vec![0.0; g];
            for j in 0..n {
                if j == i {
                    continue;
                }
                let kf = gaussian_kernel(func_dists[i * n + j], h_func);
                let ks = match scalar_dists {
                    Some(sd) if h_scalar > 1e-15 => gaussian_kernel(sd[i * n + j], h_scalar),
                    _ => 1.0,
                };
                votes[labels[j]] += kf * ks;
            }
            argmax_class(&votes)
        })
        .collect()
}
