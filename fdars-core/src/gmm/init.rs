//! Feature extraction and k-means++ initialization for GMM.

use super::covariance::compute_covariances;
use super::CovType;
use crate::basis::fdata_to_basis_1d;
use crate::matrix::FdMatrix;
use rand::prelude::*;

// ---------------------------------------------------------------------------
// Feature extraction: basis coefficients + optional covariates
// ---------------------------------------------------------------------------

/// Build feature matrix: project curves onto basis, optionally append covariates.
/// Returns (feature_matrix as Vec<Vec<f64>>, dimension d).
pub(super) fn build_features(
    data: &FdMatrix,
    argvals: &[f64],
    covariates: Option<&FdMatrix>,
    nbasis: usize,
    basis_type: i32,
    cov_weight: f64,
) -> Option<(Vec<Vec<f64>>, usize)> {
    let n = data.nrows();
    let proj = fdata_to_basis_1d(data, argvals, nbasis, basis_type)?;
    let coef = &proj.coefficients;
    let d_basis = coef.ncols();

    let d_cov = covariates.map_or(0, super::super::matrix::FdMatrix::ncols);
    let d = d_basis + d_cov;

    let mut features = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(d);
        for j in 0..d_basis {
            row.push(coef[(i, j)]);
        }
        if let Some(cov) = covariates {
            for j in 0..d_cov {
                row.push(cov[(i, j)] * cov_weight);
            }
        }
        features.push(row);
    }

    Some((features, d))
}

// ---------------------------------------------------------------------------
// GMM initialization: k-means++ on feature vectors
// ---------------------------------------------------------------------------

/// Euclidean distance squared between two feature vectors.
fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

/// Sample an index proportional to weights using cumulative distribution.
fn weighted_sample(weights: &[f64], rng: &mut StdRng) -> usize {
    let total: f64 = weights.iter().sum();
    if total < 1e-15 {
        return rng.gen_range(0..weights.len());
    }
    let r = rng.gen::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= r {
            return i;
        }
    }
    weights.len() - 1
}

/// K-means++ initialization on feature vectors. Returns initial means.
fn kmeans_pp_init(features: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let n = features.len();
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    centers.push(features[rng.gen_range(0..n)].clone());

    let mut min_dists = vec![f64::INFINITY; n];
    for c_idx in 1..k {
        let last = &centers[c_idx - 1];
        for i in 0..n {
            min_dists[i] = min_dists[i].min(dist_sq(&features[i], last));
        }
        let chosen = weighted_sample(&min_dists, rng);
        centers.push(features[chosen].clone());
    }
    centers
}

/// Assign each feature vector to its nearest center.
fn assign_nearest(features: &[Vec<f64>], centers: &[Vec<f64>]) -> Vec<usize> {
    features
        .iter()
        .map(|f| {
            centers
                .iter()
                .enumerate()
                .map(|(c, ctr)| (c, dist_sq(f, ctr)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(c, _)| c)
        })
        .collect()
}

/// Recompute centers from assignments; keep old center if cluster is empty.
fn update_centers(
    features: &[Vec<f64>],
    assignments: &[usize],
    old_centers: &[Vec<f64>],
    k: usize,
) -> Vec<Vec<f64>> {
    let d = features[0].len();
    let mut counts = vec![0usize; k];
    let mut new_centers = vec![vec![0.0; d]; k];
    for (i, &c) in assignments.iter().enumerate() {
        counts[c] += 1;
        for j in 0..d {
            new_centers[c][j] += features[i][j];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d {
                new_centers[c][j] /= counts[c] as f64;
            }
        } else {
            new_centers[c] = old_centers[c].clone();
        }
    }
    new_centers
}

/// Run a few k-means iterations to get initial cluster assignments.
pub(super) fn kmeans_init_assignments(
    features: &[Vec<f64>],
    k: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let mut centers = kmeans_pp_init(features, k, rng);
    let mut assignments = vec![0usize; features.len()];
    for _ in 0..10 {
        assignments = assign_nearest(features, &centers);
        centers = update_centers(features, &assignments, &centers, k);
    }
    assignments
}

/// Initialize GMM parameters from k-means assignments.
pub(super) fn init_params_from_assignments(
    features: &[Vec<f64>],
    assignments: &[usize],
    k: usize,
    d: usize,
    cov_type: CovType,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = features.len();
    let mut counts = vec![0usize; k];
    let mut means = vec![vec![0.0; d]; k];

    for i in 0..n {
        let c = assignments[i];
        counts[c] += 1;
        for j in 0..d {
            means[c][j] += features[i][j];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d {
                means[c][j] /= counts[c] as f64;
            }
        }
    }

    let reg = 1e-6; // regularization
    let covariances = compute_covariances(features, assignments, &means, k, d, cov_type, reg);

    let weights: Vec<f64> = counts.iter().map(|&c| c.max(1) as f64 / n as f64).collect();

    (means, covariances, weights)
}
